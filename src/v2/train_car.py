"""Distributed RLlib PPO training script for the autonomous car.

Trains continuous non-holonomic Ackermann navigation policy using the 156-float
360-degree spatial perception observation space across the Ray cluster.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Sanitize sys.path to prevent foreign python version site-packages (e.g. Nix ROS python3.14 leaking into Python 3.12)
_curr_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
sys.path = [p for p in sys.path if "/lib/python" not in p or f"/lib/{_curr_ver}" in p]

# Ensure root workspace is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from typing import Any, Final

import numpy as np
import ray  # noqa: E402
from ray.rllib.algorithms.ppo import PPOConfig  # noqa: E402
from ray.tune.registry import register_env  # noqa: E402

from src.sim.car_gazebo_env import CarGazeboEnv  # noqa: E402
from src.v2.cluster_config import HEAD_ADDRESS, HEAD_TMPDIR  # noqa: E402

ENV_NAME = "CarGazeboEnv-v0"
CHECKPOINTS_DIR = BASE_DIR / "models" / "car_rllib"
VIDEO_FPS: Final[float] = 20.0


def env_creator(env_config: dict) -> CarGazeboEnv:
    return CarGazeboEnv(**env_config)


def evaluate_and_record(
    algo: Any,
    iter_idx: int,
    video_dir: Path,
    num_episodes: int = 1,
    num_obstacles: int = 6,
) -> list[dict[str, Any]]:
    """Runs evaluation episodes on the head node and writes rendered MP4 videos."""
    import cv2
    import torch

    video_dir.mkdir(parents=True, exist_ok=True)
    eval_results: list[dict[str, Any]] = []
    module = algo.get_module()

    for ep in range(num_episodes):
        env = CarGazeboEnv(
            sim_mode="mock",
            num_obstacles=num_obstacles,
            render_mode="rgb_array",
        )
        obs, info = env.reset()
        frames: list[np.ndarray] = [env.render()]
        total_reward = 0.0
        step = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                fwd_out = module.forward_inference({"obs": obs_tensor})
                # For continuous Gaussian policy, first 2 outputs are the mean actions
                action = fwd_out["action_dist_inputs"][0, :2].cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            frames.append(env.render())

        env.close()

        if info.get("mission_completed"):
            status = "completed"
        elif info.get("collision"):
            status = "collision"
        elif truncated:
            status = "timeout"
        else:
            status = "terminated"

        video_path = (
            video_dir
            / f"eval_iter_{iter_idx:03d}_ep_{ep:02d}_{status}_rew_{int(total_reward):+05d}.mp4"
        )

        h, w, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(video_path), fourcc, VIDEO_FPS, (w, h))
        for frame in frames:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(bgr)
        out.release()

        eval_summary = {
            "episode": ep,
            "steps": step,
            "reward": total_reward,
            "status": status,
            "video_path": str(video_path),
        }
        eval_results.append(eval_summary)
        print(
            f"  [Eval Video] Iter {iter_idx:2d} | Ep {ep+1}/{num_episodes} | "
            f"Reward: {total_reward:6.1f} | Steps: {step:3d} | Status: {status} -> {video_path.name}"
        )

    return eval_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train autonomous car with RLlib PPO")
    parser.add_argument(
        "--cluster",
        action="store_true",
        help="Connect to distributed Ray cluster head",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=5,
        help="Number of PPO training iterations",
    )
    parser.add_argument(
        "--runners",
        type=int,
        default=None,
        help="Number of distributed environment runners",
    )
    parser.add_argument(
        "--sim-mode",
        type=str,
        default="mock",
        choices=["mock", "gazebo"],
        help="Simulation mode: 'mock' (analytical CPU) or 'gazebo' (live ROS 2 Gazebo simulation)",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=2,
        help="Interval of training iterations to run evaluation and record demo video (0 to disable)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=1,
        help="Number of evaluation episodes to record per eval interval",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="logs/videos",
        help="Directory to save evaluation demo videos",
    )
    args = parser.parse_args()

    os.environ["TMPDIR"] = HEAD_TMPDIR
    os.environ["RAY_ENABLE_UV_RUNTIME_ENV"] = "0"

    runtime_excludes = [
        "best_model.zip",
        "yolov8s.pt",
        str(BASE_DIR / "best_model.zip"),
        str(BASE_DIR / "yolov8s.pt"),
        "models/**",
        "datasets/**",
        "logs/**",
        "*.zip",
        "*.pt",
        ".venv/**",
        "src/v2/.venv/**",
    ]

    if args.cluster:
        print(f"[train] Connecting to Ray cluster at {HEAD_ADDRESS}...")
        ray.init(
            address=HEAD_ADDRESS,
            runtime_env={
                "working_dir": str(BASE_DIR),
                "excludes": runtime_excludes,
            },
            ignore_reinit_error=True,
        )
        active_nodes = [n for n in ray.nodes() if n["Alive"]]
        print(f"[train] Connected to cluster with {len(active_nodes)} active nodes.")
        num_runners = args.runners if args.runners is not None else max(1, len(active_nodes) - 1)
    else:
        print("[train] Running locally (single-node mode)...")
        os.environ.pop("RAY_ADDRESS", None)
        ray.init(
            address="local",
            runtime_env={
                "working_dir": str(BASE_DIR),
                "env_vars": {"PYTHONPATH": str(BASE_DIR)},
                "excludes": runtime_excludes,
            },
            ignore_reinit_error=True,
        )
        if args.sim_mode == "gazebo":
            num_runners = args.runners if args.runners is not None else 1
            print(f"[train] Gazebo mode: using {num_runners} environment runner(s) for live ROS 2 simulation.")
        else:
            num_runners = args.runners if args.runners is not None else 2

    register_env(ENV_NAME, env_creator)
    print(f"[train] Configuring PPO with {num_runners} environment runners...")

    config = (
        PPOConfig()
        .environment(
            ENV_NAME,
            env_config={"sim_mode": args.sim_mode, "num_obstacles": 6},
        )
        .framework("torch")
        .training(
            train_batch_size=1000,
            minibatch_size=128,
            num_epochs=3,
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_loss_coeff=0.5,
            entropy_coeff=0.01,
        )
        .env_runners(
            num_env_runners=num_runners,
            rollout_fragment_length="auto",
        )
        .resources(num_gpus=0)
    )

    algo = config.build_algo()
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[train] Starting PPO training loop...")
    video_dir = BASE_DIR / args.video_dir
    for i in range(args.iters):
        result = algo.train()
        env_runners_stats = result.get("env_runners", {})
        mean_return = env_runners_stats.get("episode_return_mean", float("nan"))
        episodes = env_runners_stats.get("episodes_timesteps_total", 0)
        iter_num = i + 1
        print(
            f"  [Iter {iter_num:2d}/{args.iters}] Return Mean: {mean_return:6.2f} | Total Steps: {episodes}"
        )

        if args.eval_interval > 0 and (
            iter_num % args.eval_interval == 0 or iter_num == args.iters
        ):
            evaluate_and_record(
                algo=algo,
                iter_idx=iter_num,
                video_dir=video_dir,
                num_episodes=args.eval_episodes,
                num_obstacles=6,
            )

    ckpt_path = algo.save_checkpoint(str(CHECKPOINTS_DIR / "final_checkpoint"))
    print(f"\n[train] Checkpoint successfully saved to: {ckpt_path}")
    algo.stop()
    ray.shutdown()
    print("[train] Training completed successfully!")


if __name__ == "__main__":
    main()
