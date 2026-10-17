"""Distributed RLlib PPO training script for the autonomous car.

Trains continuous Ackermann navigation policy using the 154-float spatial
perception observation space across the Ray cluster.
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

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from src.sim.car_gazebo_env import CarGazeboEnv  # noqa: E402
from src.v2.cluster_config import HEAD_ADDRESS, HEAD_TMPDIR  # noqa: E402

ENV_NAME = "CarGazeboEnv-v0"
CHECKPOINTS_DIR = BASE_DIR / "models" / "car_rllib"


def env_creator(env_config: dict) -> CarGazeboEnv:
    return CarGazeboEnv(**env_config)


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
    args = parser.parse_args()

    os.environ["TMPDIR"] = HEAD_TMPDIR
    os.environ["RAY_ENABLE_UV_RUNTIME_ENV"] = "0"

    if args.cluster:
        print(f"[train] Connecting to Ray cluster at {HEAD_ADDRESS}...")
        ray.init(
            address=HEAD_ADDRESS,
            runtime_env={
                "working_dir": str(BASE_DIR),
                "excludes": [
                    "best_model.zip",
                    "yolov8s.pt",
                    "models/**",
                    "datasets/**",
                    "*.zip",
                    "*.pt",
                ],
            },
            ignore_reinit_error=True,
        )
        active_nodes = [n for n in ray.nodes() if n["Alive"]]
        print(f"[train] Connected to cluster with {len(active_nodes)} active nodes.")
        num_runners = args.runners if args.runners is not None else max(1, len(active_nodes) - 1)
    else:
        print("[train] Running locally (single-node mode)...")
        ray.init(ignore_reinit_error=True)
        num_runners = args.runners if args.runners is not None else 2

    register_env(ENV_NAME, env_creator)
    print(f"[train] Configuring PPO with {num_runners} environment runners...")

    config = (
        PPOConfig()
        .environment(
            ENV_NAME,
            env_config={"sim_mode": "mock", "num_obstacles": 6},
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
    for i in range(args.iters):
        result = algo.train()
        env_runners_stats = result.get("env_runners", {})
        mean_return = env_runners_stats.get("episode_return_mean", float("nan"))
        episodes = env_runners_stats.get("episodes_timesteps_total", 0)
        print(
            f"  [Iter {i+1:2d}/{args.iters}] Return Mean: {mean_return:6.2f} | Total Steps: {episodes}"
        )

    ckpt_path = algo.save_checkpoint(str(CHECKPOINTS_DIR / "final_checkpoint"))
    print(f"\n[train] Checkpoint successfully saved to: {ckpt_path}")
    algo.stop()
    ray.shutdown()
    print("[train] Training completed successfully!")


if __name__ == "__main__":
    main()
