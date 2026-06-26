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
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

LEARNING_RATE: float = 1e-4
TOTAL_TIMESTEPS: int = 16_000_000
CNN_DIMS: int = 256
SUCCESS_WINDOW: int = 50
SUCCESS_THRESHOLD: float = 0.8
NUM_ENVS: int = 20
NUM_EVAL_ENVS: int = 10
N_STEPS: int = 2048
N_EVAL_EPISODES: int = 50
SAVE_FREQ: int = 200_000 // NUM_ENVS
EVAL_FREQ: int = 200_000 // NUM_ENVS
N_EPOCHS: int = 4
GAMMA: float = 0.98
GAE_LAMBDA: float = 0.95
CLIP_RANGE: float = 0.2
ENT_COEF: float = 0.03
VF_COEF: float = 0.5
MAX_GRAD_NORM: float = 0.5
BATCH_SIZE: int = 512


class CurriculumCallback(BaseCallback):
    """Advance the curriculum phase when the agent achieves sufficient success."""

    def __init__(
        self,
        eval_env: VecNormalize,
        eval_callback: CustomEvalCallback | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.success_window: deque[bool] = deque(maxlen=SUCCESS_WINDOW)
        self.current_phase: int = 1
        self.eval_env = eval_env
        self.eval_callback = eval_callback

    def _on_step(self) -> bool:
        if (
            self.eval_callback is not None
            and self.eval_callback.last_eval_step == self.num_timesteps
            and self.eval_callback.last_eval_success_rate is not None
            and self.eval_callback.last_eval_success_rate >= SUCCESS_THRESHOLD
            and self.current_phase < 8
        ):
            self.current_phase += 1
            self.training_env.env_method("set_phase", self.current_phase)
            self.eval_env.env_method("set_phase", self.current_phase)
            if self.verbose > 0:
                print(f"Curriculum: advancing to phase {self.current_phase}")

        self.logger.record("curriculum/phase", self.current_phase)
        return True


class CustomEvalCallback(EvalCallback):
    """Custom EvalCallback that also logs cells_missed, success_rate, and phase."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.eval_cells_missed: list[float] = []
        self.eval_phases: list[float] = []
        self.eval_cells_missed_results: list[float] = []
        self.eval_phases_results: list[float] = []
        self.eval_success_rate_results: list[float] = []
        self.last_eval_success_rate: float | None = None
        self.last_eval_step: int | None = None

    def _log_success_callback(
        self, locals_: dict[str, Any], globals_: dict[str, Any]
    ) -> None:
        super()._log_success_callback(locals_, globals_)

        info = locals_["info"]
        if locals_["done"]:
            cells_missed = info.get("cells_missed")
            if cells_missed is not None:
                self.eval_cells_missed.append(float(cells_missed))

            phase = info.get("phase")
            if phase is not None:
                self.eval_phases.append(float(phase))

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.eval_cells_missed = []
            self.eval_phases = []
            self._is_success_buffer = []

            original_dump = self.logger.dump

            def custom_dump(step: int | None = None) -> None:
                if self.eval_cells_missed:
                    mean_cells_missed = float(np.mean(self.eval_cells_missed))
                    self.logger.record("eval/cells_missed", mean_cells_missed)
                    self.eval_cells_missed_results.append(mean_cells_missed)
                    if self.verbose >= 1:
                        print(f"Eval cells missed: {mean_cells_missed:.2f}")

                if self.eval_phases:
                    mean_phase = float(np.mean(self.eval_phases))
                    self.logger.record("eval/phase", mean_phase)
                    self.eval_phases_results.append(mean_phase)
                    if self.verbose >= 1:
                        print(f"Eval phase: {mean_phase:.2f}")

                if self._is_success_buffer:
                    success_rate = float(np.mean(self._is_success_buffer))
                    self.eval_success_rate_results.append(success_rate)

                original_dump(step)

            self.logger.dump = custom_dump
            try:
                result = super()._on_step()
                if len(self._is_success_buffer) > 0:
                    self.last_eval_success_rate = float(
                        np.mean(self._is_success_buffer)
                    )
                self.last_eval_step = self.num_timesteps
            finally:
                self.logger.dump = original_dump
            return result
        else:
            return super()._on_step()


def load_model(
    model_path: str,
    env: VecNormalize,
    log_dir: str,
) -> tuple[PPO, int]:
    """Load a saved PPO model, its curriculum phase, and VecNormalize stats."""
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path, env=env, tensorboard_log=log_dir)

    phase_path = model_path.replace(".zip", "_phase.pkl")
    initial_phase = 1
    if os.path.exists(phase_path):
        with open(phase_path, "rb") as f:
            initial_phase = pickle.load(f)  # noqa: S301
        print(f"Resuming curriculum from phase {initial_phase}")

    vecnorm_path = model_path.replace(".zip", "_vecnorm.pkl")
    if os.path.exists(vecnorm_path):
        env.load(vecnorm_path)
        print("Loaded VecNormalize stats")

    return model, initial_phase


def save_model(model: PPO, path: str, phase: int) -> None:
    """Save a PPO model, its current curriculum phase, and VecNormalize stats."""
    model.save(path)
    phase_path = path.replace(".zip", "_phase.pkl")
    with open(phase_path, "wb") as f:
        pickle.dump(phase, f)
    vec_normalize = model.get_vec_normalize_env()
    if vec_normalize is not None:
        vec_normalize.save(path.replace(".zip", "_vecnorm.pkl"))


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
        phase_path = args.resume.replace(".zip", "_phase.pkl")
        if os.path.exists(phase_path):
            with open(phase_path, "rb") as f:
                initial_phase = pickle.load(f)  # noqa: S301
            print(f"Resuming curriculum from phase {initial_phase}")

    print("Initializing training environments...")
    env = SubprocVecEnv([make_env(phase=initial_phase) for _ in range(NUM_ENVS)])
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)
    check_env(RobotCoverageEnv(phase=initial_phase), warn=True)

    if args.resume:
        model, _ = load_model(args.resume, env, log_dir)
    else:
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

    eval_env = SubprocVecEnv(
        [
            make_env(phase=initial_phase, render_mode="rgb_array")
            for _ in range(NUM_EVAL_ENVS)
        ]
    )
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False)
    eval_env.env_method("set_phase", initial_phase)
    eval_callback = CustomEvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=EVAL_FREQ,
        deterministic=True,
        n_eval_episodes=N_EVAL_EPISODES,
    )

    curriculum_callback = CurriculumCallback(eval_env, eval_callback, verbose=1)
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
