import argparse
import os

from robot_env import PHASES, RobotCoverageEnv
from stable_baselines3 import SAC

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models", "v2")
NUM_EPISODES = 20
MAX_STEPS = 15000


def main():
    parser = argparse.ArgumentParser(description="Visualize trained v2 policy")
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(MODEL_DIR, "sac_v2_final.zip"),
        help="Path to model zip",
    )
    parser.add_argument(
        "--phase",
        type=int,
        default=1,
        choices=list(PHASES.keys()),
        help="Phase to test",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=NUM_EPISODES,
        help=f"Number of episodes (default: {NUM_EPISODES})",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"No model found at {args.model}")
        print("Train first with: python train.py")
        return

    model = SAC.load(args.model)
    env = RobotCoverageEnv(render_mode="human", phase=args.phase)

    print(f"Testing phase {args.phase} for {args.episodes} episodes")

    try:
        for ep in range(args.episodes):
            obs, info = env.reset()
            total_reward = 0.0
            steps = 0

            for _ in range(MAX_STEPS):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                env.render()
                total_reward += reward
                steps += 1

                if terminated or truncated:
                    break

            cov_pct = info.get("coverage_percent", 0.0)
            coll = info.get("num_collisions", 0)
            print(
                f"Episode {ep + 1}: {steps} steps | "
                f"{cov_pct:.1%} coverage | "
                f"{coll} collisions | "
                f"reward {total_reward:.1f}"
            )

            env.close_display()
    finally:
        env.close()


if __name__ == "__main__":
    main()
