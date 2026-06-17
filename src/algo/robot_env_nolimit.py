"""
Environment variant for classical (non-RL) controllers.

Removes the RL-oriented step limit truncation and replaces it with a much
higher safety cap.  The `non_new_steps` stall detector is relaxed to
5000 steps so the controller has time to navigate around obstacles.

Usage:
    from robot_env_nolimit import RobotCoverageEnvNoLimit
    env = RobotCoverageEnvNoLimit(render_mode="human", phase=1)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from robot_env import RobotCoverageEnv, PHASES, MAX_STEPS


class RobotCoverageEnvNoLimit(RobotCoverageEnv):
    """RobotCoverageEnv without RL-oriented truncation."""

    def __init__(self, render_mode=None, phase=1, max_steps=None):
        super().__init__(render_mode=render_mode, phase=phase)
        self._classical_max_steps = max_steps

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if truncated:
            if self._classical_max_steps is not None:
                truncated = self.current_step >= self._classical_max_steps
            else:
                truncated = False

        return obs, reward, terminated, truncated, info
