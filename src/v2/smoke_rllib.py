"""Distributed Ray RLlib smoke test: PPO on CartPole across cluster workers."""

import os
from pathlib import Path

import ray
from ray.rllib.algorithms.ppo import PPOConfig

from src.v2.cluster_config import HEAD_ADDRESS, HEAD_TMPDIR


def main() -> None:
    os.environ["TMPDIR"] = HEAD_TMPDIR
    os.environ["RAY_ENABLE_UV_RUNTIME_ENV"] = "0"
    base_dir = Path(__file__).resolve().parent

    print(f"Connecting to Ray cluster at {HEAD_ADDRESS}...")
    ray.init(
        address=HEAD_ADDRESS,
        runtime_env={"working_dir": str(base_dir)},
    )

    nodes = [n for n in ray.nodes() if n["Alive"]]
    print(f"Connected to cluster with {len(nodes)} active nodes.")

    num_runners = min(6, len(nodes) - 1) if len(nodes) > 1 else 1
    print(f"Configuring PPO with {num_runners} distributed env runners...")

    config = (
        PPOConfig()
        .environment("CartPole-v1")
        .training(train_batch_size=400, minibatch_size=100, num_epochs=1)
        .env_runners(num_env_runners=num_runners)
        .resources(num_gpus=0)
    )
    algo = config.build_algo()
    for i in range(3):
        result = algo.train()
        reward = result["env_runners"]["episode_return_mean"]
        print(f"iter={i} episode_return_mean={reward:.2f}")
    algo.stop()
    ray.shutdown()
    print("DISTRIBUTED RLLIB SMOKE OK")


if __name__ == "__main__":
    main()
