import os
import typing

from robot_env import RobotCoverageEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

LEARNING_RATE_INITIAL = 1e-3
N_STEPS = 2048
BATCH_SIZE = 64
N_EPOCHS = 10
GAMMA = 0.98
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.01
LOG_STD_INIT = 0.5
TOTAL_TIMESTEPS = 1_000_000
N_ENVS = 8
SAVE_FREQ = 50000
EVAL_FREQ = 50000


class CleanEvalCallback(EvalCallback):
    """Overrides EvalCallback to close the Pygame window after evaluation."""

    def _on_step(self) -> bool:
        result = super()._on_step()

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if hasattr(self.eval_env, "env_method"):
                self.eval_env.env_method("close_display")

        return result


def linear_schedule(initial_value: float) -> typing.Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


def main():
    print("Initializing training environments...")
    env = DummyVecEnv(
        [
            lambda: RobotCoverageEnv(a=2.0, b=1.0, render_mode=None)
            for _ in range(N_ENVS)
        ]
    )
    env = VecMonitor(env)

    check_env(RobotCoverageEnv(a=2.0, b=1.0, render_mode=None), warn=True)

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

    eval_env = DummyVecEnv(
        [lambda: RobotCoverageEnv(a=2.0, b=1.0, render_mode="rgb_array")]
    )
    eval_env = VecMonitor(eval_env)

    eval_callback = CleanEvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=EVAL_FREQ,
        deterministic=True,
        render=True,
    )

    callback_list = CallbackList([checkpoint_callback, eval_callback])

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback_list)

    final_model_path = os.path.join(model_dir, "ppo_robot_final")
    model.save(final_model_path)

    eval_env.close()


if __name__ == "__main__":
    main()
