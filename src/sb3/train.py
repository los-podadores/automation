import os
import typing
from collections import deque

from robot_env import RobotCoverageEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor

LEARNING_RATE_INITIAL = 3e-4
N_STEPS = 2048
BATCH_SIZE = 64
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.01
LOG_STD_INIT = 0.0
TOTAL_TIMESTEPS = 1_000_000
N_ENVS = 16
SAVE_FREQ = 200_000
EVAL_FREQ = 200_000
SUCCESS_THRESHOLD = 0.8
WINDOW_SIZE = 50
N_STACK = 4


class CleanEvalCallback(EvalCallback):
    """Overrides EvalCallback to close the Pygame window after evaluation."""

    def _on_step(self) -> bool:
        result = super()._on_step()

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if hasattr(self.eval_env, "env_method"):
                self.eval_env.env_method("close_display")

        return result


class CurriculumCallback(BaseCallback):
    """Tracks 95% completion successes and increments phase when threshold met."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.success_window = deque(maxlen=WINDOW_SIZE)
        self.current_phase = 1

    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                info = self.locals["infos"][i]
                coverage = info.get("coverage_cells", 0)
                total = info.get("total_cells", 1)
                success = coverage > 0.95 * total if total > 1 else False
                self.success_window.append(success)

        if len(self.success_window) >= WINDOW_SIZE:
            success_rate = sum(self.success_window) / len(self.success_window)
            if success_rate >= SUCCESS_THRESHOLD and self.current_phase < 3:
                self.current_phase += 1
                self.training_env.env_method("set_phase", self.current_phase)
                if self.verbose > 0:
                    print(f"Curriculum: Advancing to phase {self.current_phase}")
                self.success_window.clear()

        return True


def linear_schedule(initial_value: float) -> typing.Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


def make_env(phase=1, render_mode=None):
    def _init():
        return RobotCoverageEnv(a=2.0, b=1.0, render_mode=render_mode, phase=phase)

    return _init


def main():
    print("Initializing training environments...")
    env = DummyVecEnv([make_env(phase=1) for _ in range(N_ENVS)])
    env = VecFrameStack(env, n_stack=N_STACK)
    env = VecMonitor(env)

    check_env(RobotCoverageEnv(a=2.0, b=1.0, render_mode=None, phase=1), warn=True)

    log_dir = "./logs/robot_coverage/"
    model_dir = "./models/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=linear_schedule(LEARNING_RATE_INITIAL),
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        policy_kwargs=dict(log_std_init=LOG_STD_INIT),
        verbose=1,
        tensorboard_log=log_dir,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ, save_path=model_dir, name_prefix="ppo_robot"
    )

    eval_env = DummyVecEnv([make_env(phase=1, render_mode="rgb_array")])
    eval_env = VecFrameStack(eval_env, n_stack=N_STACK)
    eval_env = VecMonitor(eval_env)

    eval_callback = CleanEvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=EVAL_FREQ,
        deterministic=True,
        render=True,
    )

    curriculum_callback = CurriculumCallback(verbose=1)

    callback_list = CallbackList(
        [checkpoint_callback, eval_callback, curriculum_callback]
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback_list)

    final_model_path = os.path.join(model_dir, "ppo_robot_final")
    model.save(final_model_path)

    eval_env.close()


if __name__ == "__main__":
    main()
