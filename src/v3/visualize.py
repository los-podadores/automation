import argparse

from stable_baselines3 import PPO

from robot_env import RobotCoverageEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--phase", type=int, default=1, help="Phase to evaluate")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    args = parser.parse_args()

    model = PPO.load(args.model)

    env = RobotCoverageEnv(render_mode="human", phase=args.phase)

    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

            env.render()

        print(
            f"Episode {ep + 1}: {steps} steps, "
            f"coverage={info['coverage_percent']:.2%}, "
            f"collisions={info['num_collisions']}, "
            f"total_reward={total_reward:.2f}"
        )

    env.close()


if __name__ == "__main__":
    main()
