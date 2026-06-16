"""Visualize a trained RPO agent on RobotCoverageEnv using a torch state_dict."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_REINFORCE = _HERE.parent
if str(_REINFORCE) not in sys.path:
    sys.path.insert(0, str(_REINFORCE))

from robot_env import RobotCoverageEnv

from v2.agent import Agent

DEFAULT_MODEL_PATH = os.path.join(
    _HERE, "..", "..", "models", "v2", "rpo_robot_final.pt"
)
NUM_EPISODES = 20
MAX_STEPS = 10_000


def _to_tensor(obs: dict, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "visual": torch.as_tensor(obs["visual"], device=device).unsqueeze(0),
        "sensors": torch.as_tensor(obs["sensors"], device=device).unsqueeze(0).float(),
    }


def main(model_path: str = DEFAULT_MODEL_PATH):
    if not os.path.exists(model_path):
        print(f"No model found at {model_path}")
        print("Train first with: python -m v2.train")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    sensor_dim = ckpt["sensor_dim"]
    action_dim = ckpt["action_dim"]
    rpo_alpha = ckpt.get("args", {}).get("rpo_alpha", 0.5)

    agent = Agent(sensor_dim=sensor_dim, action_dim=action_dim, rpo_alpha=rpo_alpha).to(
        device
    )
    agent.load_state_dict(ckpt["agent_state_dict"])
    agent.eval()

    env = RobotCoverageEnv(a=2.0, b=1.0, render_mode="human")

    try:
        for ep in range(NUM_EPISODES):
            obs, _ = env.reset()
            obs_dict = _to_tensor(obs, device)
            total_reward = 0.0
            steps = 0

            for _ in range(MAX_STEPS):
                with torch.no_grad():
                    action, _, _, _ = agent.get_action_and_value(obs_dict)
                obs, reward, terminated, truncated, _ = env.step(
                    action.cpu().numpy()[0]
                )
                env.render()
                total_reward += float(reward)
                steps += 1
                obs_dict = _to_tensor(obs, device)
                if terminated or truncated:
                    break

            coverage = env.unwrapped.visited_grid
            print(
                f"Episode {ep + 1}: {steps} steps | {len(coverage)} cells covered | reward {total_reward:.1f}"
            )

            env.close_display()
    finally:
        env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    main(args.model)
