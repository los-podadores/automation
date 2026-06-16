import argparse
import os

from robot_env import RobotCoverageEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "ppo_robot_final.zip"
)
NUM_EPISODES = 20
MAX_STEPS = 10000
N_STACK = 4


def main():
    parser = argparse.ArgumentParser(description="Visualize trained policy")
    parser.add_argument(
        "--phase",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Phase to test (1, 2, or 3). Default: 1",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=NUM_EPISODES,
        help=f"Number of episodes to run (default: {NUM_EPISODES})",
    )
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        print(f"No model found at {MODEL_PATH}")
        print("Train first with: python train.py")
        return

    model = PPO.load(MODEL_PATH)
    env = DummyVecEnv(
        [lambda: RobotCoverageEnv(a=2.0, b=1.0, render_mode="human", phase=args.phase)]
    )
    env = VecFrameStack(env, n_stack=N_STACK)

    print(f"Testing phase {args.phase} for {args.episodes} episodes")

    try:
        for ep in range(args.episodes):
            obs = env.reset()
            total_reward = 0.0
            steps = 0

            for _ in range(MAX_STEPS):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                env.render()
                total_reward += reward[0]
                steps += 1

                if done[0]:
                    break

            coverage = int(env.envs[0].unwrapped.visited_grid.sum())
            print(
                f"Episode {ep + 1}: {steps} steps | {coverage} cells covered | reward {total_reward:.1f}"
            )

            env.envs[0].unwrapped.close_display()
    finally:
        env.close()


if __name__ == "__main__":
    main()
