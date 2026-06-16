"""RPO (Robust Policy Optimization) training on RobotCoverageEnv.

CleanRL-style single-file implementation using the RPO continuous-action
algorithm with NatureCNN + MLP encoder for the Dict observation space.
"""

from __future__ import annotations

import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Allow running this file as a script from src/reinforce/ or src/reinforce/v2/.
_HERE = Path(__file__).resolve().parent
_REINFORCE = _HERE.parent
if str(_REINFORCE) not in sys.path:
    sys.path.insert(0, str(_REINFORCE))

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from robot_env import RobotCoverageEnv
from torch.utils.tensorboard import SummaryWriter

from v2.agent import Agent


@dataclass
class Args:
    exp_name: str = "rpo_robot"
    """name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, torch.backends.cudnn.deterministic=True"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = ""
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos (rgb_array envs get recorded for the first sub-env)"""

    # Env
    env_a: float = 2.0
    env_b: float = 1.0

    # Algorithm (CleanRL RPO defaults)
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    num_envs: int = 8
    num_steps: int = 2048
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    rpo_alpha: float = 0.5

    # Checkpointing / eval
    save_dir: str = "./models/v2"
    log_dir: str = "./logs/v2"
    eval_episodes: int = 5
    eval_every_iters: int = 25
    save_every_iters: int = 25

    # to be filled in runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def make_train_env(a: float, b: float, capture_video: bool, run_name: str, idx: int):
    def thunk():
        if capture_video and idx == 0:
            env = RobotCoverageEnv(a=a, b=b, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = RobotCoverageEnv(a=a, b=b, render_mode=None)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def make_eval_env(a: float, b: float, render_mode: str = "human"):
    return RobotCoverageEnv(a=a, b=b, render_mode=render_mode)


def evaluate(
    agent: Agent,
    device: torch.device,
    sensor_dim: int,
    a: float,
    b: float,
    episodes: int,
    max_steps: int,
    render_mode: str = "human",
) -> tuple[float, float]:
    """Run deterministic evaluation episodes. Returns (mean_return, mean_coverage)."""
    env = make_eval_env(a=a, b=b, render_mode=render_mode)
    returns: list[float] = []
    coverages: list[int] = []
    try:
        for _ in range(episodes):
            obs, _ = env.reset()
            obs_dict = _to_tensor(obs, device)
            ep_return = 0.0
            for _ in range(max_steps):
                with torch.no_grad():
                    action, _, _, _ = agent.get_action_and_value(obs_dict)
                action_np = action.cpu().numpy()[0]
                obs, reward, terminated, truncated, info = env.step(action_np)
                ep_return += float(reward)
                obs_dict = _to_tensor(obs, device)
                if render_mode == "human":
                    env.render()
                if terminated or truncated:
                    break
            returns.append(ep_return)
            coverages.append(info.get("coverage_cells", 0))
    finally:
        env.close()
    return float(np.mean(returns)), float(np.mean(coverages))


def _to_tensor(obs: dict, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "visual": torch.as_tensor(obs["visual"], device=device).unsqueeze(0),
        "sensors": torch.as_tensor(obs["sensors"], device=device).unsqueeze(0).float(),
    }


def main():
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.exp_name}__{args.seed}__{timestamp}"

    repo_root = Path(__file__).resolve().parents[2]
    save_dir = (
        (repo_root / args.save_dir)
        if not os.path.isabs(args.save_dir)
        else Path(args.save_dir)
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / f"{args.exp_name}_best.pt"
    final_path = save_dir / f"{args.exp_name}_final.pt"

    log_root = (
        (repo_root / args.log_dir)
        if not os.path.isabs(args.log_dir)
        else Path(args.log_dir)
    )
    log_root.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_root / run_name)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n"
        + "\n".join(f"|{k}|{v}|" for k, v in vars(args).items()),
    )

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity or None,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"device: {device}")

    envs = gym.vector.SyncVectorEnv(
        [
            make_train_env(args.env_a, args.env_b, args.capture_video, run_name, i)
            for i in range(args.num_envs)
        ]
    )

    sample_obs, _ = envs.reset(seed=args.seed)
    visual_shape = sample_obs["visual"].shape[1:]  # (3, 64, 64)
    sensor_dim = int(np.prod(sample_obs["sensors"].shape[1:]))
    action_dim = int(np.prod(envs.single_action_space.shape))
    print(
        f"obs visual: {visual_shape}, sensor_dim: {sensor_dim}, action_dim: {action_dim}, "
        f"num_envs: {args.num_envs}, num_steps: {args.num_steps}, "
        f"batch: {args.batch_size}, minibatch: {args.minibatch_size}, "
        f"iterations: {args.num_iterations}"
    )

    agent = Agent(
        sensor_dim=sensor_dim, action_dim=action_dim, rpo_alpha=args.rpo_alpha
    ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs_buf = torch.zeros(
        (args.num_steps, args.num_envs, *visual_shape), dtype=torch.uint8, device=device
    )
    sensor_buf = torch.zeros(
        (args.num_steps, args.num_envs, sensor_dim), dtype=torch.float32, device=device
    )
    actions = torch.zeros((args.num_steps, args.num_envs, action_dim), device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    terminations = torch.zeros((args.num_steps, args.num_envs), device=device)
    truncations = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_terminated = torch.zeros(args.num_envs, device=device)

    inflight_length = np.zeros(args.num_envs, dtype=np.int64)
    inflight_return = np.zeros(args.num_envs, dtype=np.float64)

    best_eval_return = -float("inf")

    for update in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        rollout_returns: list[float] = []
        rollout_lengths: list[int] = []
        inflight_lengths_at_end: list[int] = []
        inflight_returns_at_end: list[float] = []

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs_buf[step] = torch.as_tensor(
                next_obs["visual"], device=device, dtype=torch.uint8
            )
            sensor_buf[step] = torch.as_tensor(
                next_obs["sensors"], device=device, dtype=torch.float32
            )
            terminations[step] = next_terminated

            with torch.no_grad():
                obs_dict = {"visual": obs_buf[step], "sensors": sensor_buf[step]}
                action, logprob, _, value = agent.get_action_and_value(obs_dict)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, term_np, trunc_np, infos = envs.step(action.cpu().numpy())
            rewards[step] = torch.as_tensor(
                reward, device=device, dtype=torch.float32
            ).view(-1)
            term_t = torch.as_tensor(term_np, device=device, dtype=torch.float32)
            trunc_t = torch.as_tensor(trunc_np, device=device, dtype=torch.float32)
            next_terminated = term_t

            inflight_length += 1
            inflight_return += reward
            done = np.logical_or(term_np, trunc_np)
            for i, d in enumerate(done):
                if d:
                    inflight_length[i] = 0
                    inflight_return[i] = 0.0

            if "final_info" in infos:
                for i, info in enumerate(infos["final_info"]):
                    if info and "episode" in info:
                        r = float(info["episode"]["r"])
                        l = int(info["episode"]["l"])
                        rollout_returns.append(r)
                        rollout_lengths.append(l)
                        writer.add_scalar("charts/episodic_return", r, global_step)
                        writer.add_scalar("charts/episodic_length", l, global_step)

        for i in range(args.num_envs):
            if inflight_length[i] > 0:
                inflight_lengths_at_end.append(int(inflight_length[i]))
                inflight_returns_at_end.append(float(inflight_return[i]))

        with torch.no_grad():
            next_obs_dict = {
                "visual": torch.as_tensor(
                    next_obs["visual"], device=device, dtype=torch.uint8
                ),
                "sensors": torch.as_tensor(
                    next_obs["sensors"], device=device, dtype=torch.float32
                ),
            }
            next_value = agent.get_value(next_obs_dict).reshape(1, -1)
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0.0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_terminated
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - terminations[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                )
                lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
                advantages[t] = lastgaelam
            returns = advantages + values

        b_obs = obs_buf.reshape((-1, *visual_shape))
        b_sensors = sensor_buf.reshape((-1, sensor_dim))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1, action_dim))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs: list[float] = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                mb_obs_dict = {
                    "visual": b_obs[mb_inds],
                    "sensors": b_sensors[mb_inds],
                }
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    mb_obs_dict, b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    )

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef
                    )
                    v_loss = (
                        0.5
                        * torch.max(
                            v_loss_unclipped, (v_clipped - b_returns[mb_inds]) ** 2
                        ).mean()
                    )
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl.item() > args.target_kl:
                break

        y_pred = b_values.cpu().numpy()
        y_true = b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = (
            float("nan") if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        )

        sps = int(global_step / (time.time() - start_time))
        all_returns = rollout_returns + inflight_returns_at_end
        all_lengths = rollout_lengths + inflight_lengths_at_end
        if all_returns:
            rollout_mean_return = float(np.mean(all_returns))
            rollout_mean_length = float(np.mean(all_lengths))
        else:
            rollout_mean_return = float("nan")
            rollout_mean_length = float("nan")
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar(
            "charts/rollout_mean_return", rollout_mean_return, global_step
        )
        writer.add_scalar(
            "charts/rollout_mean_episode_length", rollout_mean_length, global_step
        )
        writer.add_scalar(
            "charts/rollout_n_completed_eps", len(rollout_returns), global_step
        )
        writer.add_scalar(
            "charts/rollout_n_inflight_eps", len(inflight_lengths_at_end), global_step
        )
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", float(np.mean(clipfracs)), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", sps, global_step)
        print(
            f"iter={update}/{args.num_iterations} step={global_step} sps={sps} "
            f"ep_return(mean)={rollout_mean_return:.1f} ep_len(mean)={rollout_mean_length:.1f} "
            f"n_eps={len(rollout_returns)}+{len(inflight_lengths_at_end)}inflight "
            f"pg={pg_loss.item():.3f} v={v_loss.item():.3f} ent={entropy_loss.item():.3f}"
        )

        if update % args.eval_every_iters == 0 or update == args.num_iterations:
            eval_return, eval_coverage = evaluate(
                agent,
                device,
                sensor_dim=sensor_dim,
                a=args.env_a,
                b=args.env_b,
                episodes=args.eval_episodes,
                max_steps=10_000,
                render_mode=None,
            )
            writer.add_scalar("eval/mean_return", eval_return, global_step)
            writer.add_scalar("eval/mean_coverage_cells", eval_coverage, global_step)
            print(
                f"  eval @ step {global_step}: return={eval_return:.1f} coverage={eval_coverage:.0f}"
            )
            if eval_return > best_eval_return:
                best_eval_return = eval_return
                torch.save(
                    {
                        "agent_state_dict": agent.state_dict(),
                        "args": vars(args),
                        "sensor_dim": sensor_dim,
                        "action_dim": action_dim,
                        "global_step": global_step,
                        "eval_return": eval_return,
                    },
                    best_path,
                )
                print(f"  saved new best to {best_path}")

        if update % args.save_every_iters == 0 or update == args.num_iterations:
            torch.save(
                {
                    "agent_state_dict": agent.state_dict(),
                    "args": vars(args),
                    "sensor_dim": sensor_dim,
                    "action_dim": action_dim,
                    "global_step": global_step,
                },
                final_path,
            )

    envs.close()
    writer.close()
    print(f"done. final model: {final_path}  best model: {best_path}")


if __name__ == "__main__":
    main()
