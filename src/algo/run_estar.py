#!/usr/bin/env python3
"""
Run the ε⋆ controller on RobotCoverageEnvNoLimit for testing and visualization.

Usage:
    uv run python src/algo/run_estar.py                  # phase 1
    uv run python src/algo/run_estar.py --phase 3        # phase 3
    uv run python src/algo/run_estar.py --render human   # with pygame display
    uv run python src/algo/run_estar.py --episodes 5     # run 5 episodes
"""

import argparse
import logging
import sys
import time

import numpy as np

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)

sys.path.insert(0, "src/v2")
sys.path.insert(0, "src/algo")

from robot_env_nolimit import RobotCoverageEnvNoLimit
from estar_controller import EStarController


def run_episode(env, controller, render=False, max_steps=50000):
    obs, info = env.reset()
    controller.reset()

    total_reward = 0.0
    start = time.time()

    for step in range(max_steps):
        action = controller.step(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if render:
            env.render()

        if step % 500 == 0 and step > 0:
            print(
                f"  step={step:5d}  cov={info['coverage_percent']:.1%}  "
                f"coll={info['num_collisions']}",
                flush=True,
            )

        if terminated or truncated:
            break

    elapsed = time.time() - start
    cov = info.get("coverage_percent", 0.0)
    collisions = info.get("num_collisions", 0)

    return {
        "steps": step + 1,
        "coverage": cov,
        "collisions": collisions,
        "reward": total_reward,
        "time_s": elapsed,
        "terminated": terminated,
        "truncated": truncated,
    }


def main():
    parser = argparse.ArgumentParser(description="Run ε⋆ controller")
    parser.add_argument("--phase", type=int, default=1, help="Environment phase (1-8)")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes")
    parser.add_argument(
        "--render", choices=["none", "human", "rgb"], default="none"
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=50000)
    args = parser.parse_args()

    render_mode = None if args.render == "none" else args.render

    env = RobotCoverageEnvNoLimit(
        render_mode=render_mode, phase=args.phase, max_steps=args.max_steps
    )
    controller = EStarController(env)

    if args.seed is not None:
        np.random.seed(args.seed)

    results = []
    for ep in range(args.episodes):
        r = run_episode(
            env, controller,
            render=(render_mode is not None),
            max_steps=args.max_steps,
        )
        results.append(r)
        print(
            f"Episode {ep+1}: steps={r['steps']}, coverage={r['coverage']:.1%}, "
            f"collisions={r['collisions']}, time={r['time_s']:.2f}s, "
            f"{'done' if r['terminated'] else 'safety-limit'}"
        )

    if len(results) > 1:
        avg_cov = np.mean([r["coverage"] for r in results])
        avg_steps = np.mean([r["steps"] for r in results])
        avg_coll = np.mean([r["collisions"] for r in results])
        avg_time = np.mean([r["time_s"] for r in results])
        print(f"\n--- Averages over {len(results)} episodes ---")
        print(f"  Coverage:   {avg_cov:.1%}")
        print(f"  Steps:      {avg_steps:.0f}")
        print(f"  Collisions: {avg_coll:.1f}")
        print(f"  Time:       {avg_time:.2f}s")

    env.close()


if __name__ == "__main__":
    main()
