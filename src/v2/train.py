import os
from collections import deque

from architectures import StackedMapFeaturesExtractor
from robot_env import (
    MAP_SIZE,
    NUM_MAPS,
    PHASES,
    SENSOR_DIM,
    RobotCoverageEnv,
)
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

LEARNING_RATE = 2e-5
BUFFER_SIZE = 100_000
TRAIN_FREQ = 1
GRADIENT_STEPS = 1
BATCH_SIZE = 256
GAMMA = 0.99
TOTAL_TIMESTEPS = 1_000_000
SAVE_FREQ = 100_000
EVAL_FREQ = 100_000
CNN_DIMS = 256
SUCCESS_WINDOW = 50
SUCCESS_THRESHOLD = 0.8


class CurriculumCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.success_window = deque(maxlen=SUCCESS_WINDOW)
        self.current_phase = 1

    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                info = self.locals["infos"][i]
                cov = info.get("coverage_percent", 0.0)
                goal = PHASES[self.current_phase]["goal"]
                self.success_window.append(cov >= goal)

        if len(self.success_window) >= SUCCESS_WINDOW:
            rate = sum(self.success_window) / len(self.success_window)
            if rate >= SUCCESS_THRESHOLD and self.current_phase < 8:
                self.current_phase += 1
                self.training_env.env_method("set_phase", self.current_phase)
                self.success_window.clear()
                if self.verbose > 0:
                    print(f"Curriculum: advancing to phase {self.current_phase}")

        return True


def make_env(phase=1, render_mode=None):
    def _init():
        env = RobotCoverageEnv(render_mode=render_mode, phase=phase)
        return env

    return _init


def main():
    print("Initializing training environments...")
    env = DummyVecEnv([make_env(phase=1)])
    check_env(RobotCoverageEnv(phase=1), warn=True)

    log_dir = "./logs/v2/"
    model_dir = "./models/v2/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    policy_kwargs = dict(
        features_extractor_class=StackedMapFeaturesExtractor,
        features_extractor_kwargs=dict(
            features_dim=CNN_DIMS,
            map_size=MAP_SIZE,
            num_maps=NUM_MAPS,
            sensor_dim=SENSOR_DIM,
            num_map_types=3,
        ),
        net_arch=dict(pi=[CNN_DIMS, CNN_DIMS], qf=[CNN_DIMS, CNN_DIMS]),
    )

    model = SAC(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        train_freq=TRAIN_FREQ,
        gradient_steps=GRADIENT_STEPS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=log_dir,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ, save_path=model_dir, name_prefix="sac_v2"
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

    callback_list = CallbackList(
        [checkpoint_callback, eval_callback, curriculum_callback]
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback_list)
    model.save(os.path.join(model_dir, "sac_v2_final"))
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
