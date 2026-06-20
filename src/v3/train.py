"""Training script for PPO agent with curriculum learning."""

from __future__ import annotations

import argparse
import os
import pickle
from collections import deque
from typing import Any

import numpy as np
from architectures import StackedMapFeaturesExtractor
from robot_env import (
    CELLS_MISSED_THRESHOLD,
    MAP_SIZE,
    NUM_MAPS,
    SENSOR_DIM,
    RobotCoverageEnv,
)
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

LEARNING_RATE: float = 3e-4
TOTAL_TIMESTEPS: int = 16_000_000
CNN_DIMS: int = 256
SUCCESS_WINDOW: int = 50
SUCCESS_THRESHOLD: float = 0.8
NUM_ENVS: int = 20
N_STEPS: int = 2048
SAVE_FREQ: int = 200_000 // NUM_ENVS
EVAL_FREQ: int = 200_000 // NUM_ENVS
N_EPOCHS: int = 4
GAMMA: float = 0.98
GAE_LAMBDA: float = 0.95
CLIP_RANGE: float = 0.2
ENT_COEF: float = 0.01
VF_COEF: float = 0.5
MAX_GRAD_NORM: float = 0.5
BATCH_SIZE: int = 512


class CurriculumCallback(BaseCallback):
    """Advance the curriculum phase when the agent achieves sufficient success."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.success_window: deque[bool] = deque(maxlen=SUCCESS_WINDOW)
        self.current_phase: int = 1

    def _on_step(self) -> bool:
        missed_cells: list[int] = []
        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                info = self.locals["infos"][i]
                cells_missed = info.get("cells_missed", 999)
                missed_cells.append(cells_missed)
                self.success_window.append(cells_missed < CELLS_MISSED_THRESHOLD)

        if missed_cells:
            self.logger.record("field/cells_missed_mean", float(np.mean(missed_cells)))
            self.logger.record(
                "curriculum/success_rate",
                sum(self.success_window) / len(self.success_window),
            )

        if len(self.success_window) >= SUCCESS_WINDOW:
            rate = sum(self.success_window) / len(self.success_window)
            if rate >= SUCCESS_THRESHOLD and self.current_phase < 8:
                self.current_phase += 1
                self.training_env.env_method("set_phase", self.current_phase)
                self.success_window.clear()
                if self.verbose > 0:
                    print(f"Curriculum: advancing to phase {self.current_phase}")

        self.logger.record("curriculum/phase", self.current_phase)
        return True


def load_model(
    model_path: str,
    env: SubprocVecEnv,
    log_dir: str,
) -> tuple[PPO, int]:
    """Load a saved PPO model and its curriculum phase."""
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path, env=env, tensorboard_log=log_dir)

    phase_path = model_path.replace(".zip", "_phase.pkl")
    initial_phase = 1
    if os.path.exists(phase_path):
        with open(phase_path, "rb") as f:
            initial_phase = pickle.load(f)  # noqa: S301
        print(f"Resuming curriculum from phase {initial_phase}")

    return model, initial_phase


def save_model(model: PPO, path: str, phase: int) -> None:
    """Save a PPO model and its current curriculum phase."""
    model.save(path)
    phase_path = path.replace(".zip", "_phase.pkl")
    with open(phase_path, "wb") as f:
        pickle.dump(phase, f)


def make_env(phase: int = 1, render_mode: str | None = None) -> callable[[], Monitor]:
    """Return a callable that creates a wrapped ``RobotCoverageEnv``."""

    def _init() -> Monitor:
        env = RobotCoverageEnv(render_mode=render_mode, phase=phase)
        return Monitor(
            env, info_keywords=("coverage_percent", "num_collisions", "phase")
        )

    return _init


def main() -> None:
    """Entry point for PPO curriculum training."""
    parser = argparse.ArgumentParser(description="Train PPO model for robot coverage")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a saved model checkpoint to resume training from",
    )
    args = parser.parse_args()

    log_dir = "./logs/v3/"
    model_dir = "./models/v3/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    policy_kwargs: dict[str, Any] = dict(
        features_extractor_class=StackedMapFeaturesExtractor,
        features_extractor_kwargs=dict(
            features_dim=CNN_DIMS,
            map_size=MAP_SIZE,
            num_maps=NUM_MAPS,
            sensor_dim=SENSOR_DIM,
            num_map_types=3,
        ),
        net_arch=dict(pi=[CNN_DIMS, CNN_DIMS], vf=[CNN_DIMS, CNN_DIMS]),
    )

    initial_phase = 1
    if args.resume:
        env = SubprocVecEnv([make_env(phase=1) for _ in range(NUM_ENVS)])
        check_env(RobotCoverageEnv(phase=1), warn=True)
        model, initial_phase = load_model(args.resume, env, log_dir)
    else:
        print("Initializing training environments...")
        env = SubprocVecEnv([make_env(phase=1) for _ in range(NUM_ENVS)])
        check_env(RobotCoverageEnv(phase=1), warn=True)
        model = PPO(
            policy="MultiInputPolicy",
            env=env,
            learning_rate=LEARNING_RATE,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            clip_range=CLIP_RANGE,
            ent_coef=ENT_COEF,
            vf_coef=VF_COEF,
            max_grad_norm=MAX_GRAD_NORM,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=log_dir,
        )

    env.env_method("set_phase", initial_phase)

    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=model_dir,
        name_prefix="ppo_v3",
    )

    eval_env = DummyVecEnv([make_env(phase=1, render_mode="rgb_array")])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=EVAL_FREQ,
        deterministic=True,
        n_eval_episodes=5,
    )

    curriculum_callback = CurriculumCallback(verbose=1)
    curriculum_callback.current_phase = initial_phase

    callback_list = CallbackList(
        [checkpoint_callback, eval_callback, curriculum_callback],
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback_list)
    save_model(
        model,
        os.path.join(model_dir, "ppo_v3_final"),
        curriculum_callback.current_phase,
    )
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
