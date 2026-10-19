import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import ray  # noqa: E402
from ray.rllib.algorithms.ppo import PPOConfig  # noqa: E402

from src.v2.cluster_config import HEAD_ADDRESS, HEAD_TMPDIR  # noqa: E402


def main() -> None:
    os.environ["TMPDIR"] = HEAD_TMPDIR
    os.environ["RAY_ENABLE_UV_RUNTIME_ENV"] = "0"
    os.environ["RAY_AUTH_MODE"] = "disabled"

    print(f"Connecting to Ray cluster at {HEAD_ADDRESS}...")
    ray.init(
        address=HEAD_ADDRESS,
        runtime_env={
            "working_dir": str(BASE_DIR),
            "excludes": [
                "logs/**",
                "models/**",
                ".venv/**",
                "src/v2/.venv/**",
                "*.mp4",
                "*.zip",
                "*.pt",
            ],
        },
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
