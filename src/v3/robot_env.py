"""RobotCoverageEnv — Gymnasium environment for 2D robot coverage.

This module composes the decomposed sub-modules in ``env/`` into a single
Gymnasium-compatible class with the same public API as the original
monolithic implementation.
"""

from __future__ import annotations

import logging
import math
import warnings

import cv2
import gymnasium as gym
import numpy as np
import pygame
from env import collision as _col
from env import maps as _maps
from env import sensors as _sensors
from env.config import (
    CELLS_MISSED_THRESHOLD,  # noqa: F401 — re-exported for external consumers
    DT,
    HEADING_NOISE,
    MAP_SIZE,
    MAX_FIELD_ATTEMPTS,
    MAX_NON_NEW_STEPS,
    METERS_PER_PIXEL,
    NUM_MAPS,
    PHASE_WEIGHT_DECAY,
    PHASES,
    POSITION_NOISE,
    REWARD_AREA_MAX,
    REWARD_AREA_SCALE,
    REWARD_BASE_PENALTY,
    REWARD_COLLISION,
    REWARD_TV_MAX,
    REWARD_TV_SCALE,
    ROBOT_RADIUS,
    ROBOT_RADIUS_PX,  # noqa: F401 — re-exported for external consumers
    ROBOT_SPEED_V,
    ROBOT_SPEED_W,
    SENSOR_DIM,  # noqa: F401 — re-exported for external consumers
)
from env.field_generator import generate_random_field, get_safe_spawn
from env.renderer import draw_robot_footprint_local, render_frame
from env.transforms import get_multi_scale_map, m_to_grid_px
from env.utils import total_variation
from gymnasium import spaces

logger = logging.getLogger(__name__)


class RobotCoverageEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode: str | None = None, phase: int = 1) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.phase = phase
        self._active_phase = phase

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        obs_shape = (NUM_MAPS, MAP_SIZE, MAP_SIZE)
        self.observation_space = spaces.Dict(
            {
                "coverage": spaces.Box(
                    low=0, high=1, shape=obs_shape, dtype=np.float32
                ),
                "obstacles": spaces.Box(
                    low=0, high=1, shape=obs_shape, dtype=np.float32
                ),
                "frontier": spaces.Box(
                    low=0, high=1, shape=obs_shape, dtype=np.float32
                ),
                "sensors": spaces.Box(
                    low=np.array(
                        [0.0] * 6 + [-1.0, -1.0, 0.0, -1.0, -1.0], dtype=np.float32
                    ),
                    high=np.array(
                        [1.0] * 6 + [1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32
                    ),
                    dtype=np.float32,
                ),
            }
        )

        self.field = None
        self.field_grid = None
        self.agent_pos_m = None
        self.agent_heading = 0.0
        self.last_v = 0.0
        self.last_w = 0.0

        self.pixels_per_meter = 1.0 / METERS_PER_PIXEL
        self.grid_size_p = 1
        self.grid_size_m = 1.0
        self.coverage_map = None
        self.overlap_map = None
        self.obstacle_map = None
        self.frontier_map = None

        self.current_step = 0
        self.non_new_steps = 0
        self.num_collisions = 0
        self.total_cells = 1
        self.coverage_in_pixels = 0
        self.coverage_in_percent = 0.0
        self.global_tv = 0.0
        self.local_coverage_old = None
        self.local_known_obstacles_old = None
        self._last_stamp_bbox = None

        self.window_size = 800
        self.window = None
        self.clock = None
        self.render_offset = np.array([0.0, 0.0])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _m_to_grid_px(self, pos_m: np.ndarray) -> tuple[int, int]:
        return m_to_grid_px(pos_m, self.render_offset, self.pixels_per_meter)

    def _check_collision(self, pos_m: np.ndarray) -> bool:
        return _col.check_collision(
            pos_m,
            self.render_offset,
            self.pixels_per_meter,
            self.grid_size_p,
            self.field_grid,
            self.collision_map,
        )

    def _draw_robot_footprint_local(self, pos_m, heading, local_size=32):
        return draw_robot_footprint_local(
            pos_m,
            heading,
            local_size,
            self.render_offset,
            self.pixels_per_meter,
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.non_new_steps = 0
        self.num_collisions = 0
        self.last_v = 0.0
        self.last_w = 0.0

        self.phase = self._sample_phase()
        rule = PHASES[self.phase]

        for _attempt in range(MAX_FIELD_ATTEMPTS):
            self.field = generate_random_field(self.np_random, rule)

            # Rasterize field
            minx, miny, maxx, maxy = self.field.bounds
            pad = 5.0
            self.grid_size_m = max(maxx - minx, maxy - miny) + 2 * pad
            self.grid_size_p = max(1, int(self.grid_size_m * self.pixels_per_meter))
            self.render_offset = np.array([minx - pad, miny - pad])

            self.field_grid = np.zeros(
                (self.grid_size_p, self.grid_size_p), dtype=np.uint8
            )
            exterior = np.array(self.field.exterior.coords, dtype=np.float32)
            exterior_px = (
                (exterior - self.render_offset) * self.pixels_per_meter
            ).astype(np.int32)
            cv2.fillPoly(self.field_grid, [exterior_px], 1)
            for interior in self.field.interiors:
                hole = np.array(interior.coords, dtype=np.float32)
                hole_px = ((hole - self.render_offset) * self.pixels_per_meter).astype(
                    np.int32
                )
                cv2.fillPoly(self.field_grid, [hole_px], 0)

            self.true_obstacle_map = (1 - self.field_grid).astype(np.float32)
            self.collision_map, self.virtual_wall_map = _maps.compute_static_maps(
                self.true_obstacle_map
            )
            self.spawn_safety_map = _maps.compute_spawn_safety_map(
                self.true_obstacle_map
            )

            try:
                self.agent_pos_m, self.agent_heading = get_safe_spawn(
                    self.field, self.np_random
                )
            except RuntimeError:
                continue

            self.coverable_area = _maps.compute_coverable_area(
                self.field_grid,
                self.collision_map,
                self.agent_pos_m,
                self.render_offset,
                self.pixels_per_meter,
            )

            valid_positions = (
                (self.field_grid > 0) & (self.virtual_wall_map == 0)
            ).astype(np.uint8)
            total_valid = valid_positions.sum()
            reachable_valid = self.coverable_area.sum()

            if total_valid > 0 and (reachable_valid / total_valid) > 0.50:
                break
        else:
            warnings.warn(
                f"Failed to generate a solvable field after "
                f"{MAX_FIELD_ATTEMPTS} attempts.",
                stacklevel=2,
            )

        # Init maps
        self.coverage_map = np.zeros(
            (self.grid_size_p, self.grid_size_p), dtype=np.float32
        )
        self.overlap_map = np.zeros(
            (self.grid_size_p, self.grid_size_p), dtype=np.float32
        )
        self.obstacle_map = np.zeros(
            (self.grid_size_p, self.grid_size_p), dtype=np.float32
        )

        _maps.stamp_initial_coverage(
            self.coverage_map,
            self.overlap_map,
            self.collision_map,
            self.agent_pos_m,
            self.render_offset,
            self.pixels_per_meter,
            self.grid_size_p,
        )
        self.frontier_map = _maps.compute_frontier_map(
            self.coverage_map, self.collision_map
        )

        self.total_cells = max(int(self.coverable_area.sum()), 1)
        cov = self.coverage_map.copy()
        cov[self.coverable_area == 0] = 0
        self.coverage_in_pixels = int(cov.sum())
        self.coverage_in_percent = self.coverage_in_pixels / self.total_cells
        self.global_tv = total_variation(self.coverage_map, self.virtual_wall_map)

        self._noisy_pos_m = self.agent_pos_m.copy() + self.np_random.normal(
            0, POSITION_NOISE, 2
        )
        self._noisy_heading = self.agent_heading + self.np_random.normal(
            0, HEADING_NOISE
        )

        sensors, hit_points = _sensors.compute_sensors(
            self.agent_pos_m,
            self.agent_heading,
            self.last_v,
            self.last_w,
            self.frontier_map,
            self.render_offset,
            self.pixels_per_meter,
            self.grid_size_p,
            self.true_obstacle_map,
        )
        self._last_sensors = sensors
        _maps.update_obstacle_map_from_sensors(
            self.obstacle_map,
            self.virtual_wall_map,
            hit_points,
            self.grid_size_p,
        )

        self.old_frontier_distance = _sensors.get_distance_to_closest_frontier(
            self.frontier_map,
            self.agent_pos_m,
            self.render_offset,
            self.pixels_per_meter,
            self.grid_size_p,
        )

        obs = self._get_obs()
        info = {
            "coverage_cells": self.coverage_in_pixels,
            "total_cells": self.total_cells,
            "coverage_percent": self.coverage_in_percent,
            "num_collisions": self.num_collisions,
            "phase": self.phase,
        }
        return obs, info

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action):
        self.current_step += 1
        raw_throttle = float(action[0])
        throttle = raw_throttle if raw_throttle >= 0 else raw_throttle * 0.5
        steering = float(np.clip(action[1], -1.0, 1.0))

        lin_vel = throttle * ROBOT_SPEED_V
        lin_vel *= 1 - abs(steering) * 0.5
        ang_vel = steering * ROBOT_SPEED_W

        old_pos = self.agent_pos_m.copy()

        new_heading = (self.agent_heading + ang_vel * DT) % (2 * math.pi)
        inter_heading = self.agent_heading + ang_vel * DT / 2
        dx = lin_vel * DT * math.cos(inter_heading)
        dy = lin_vel * DT * math.sin(inter_heading)

        test_pos = self.agent_pos_m.copy()
        test_pos[0] += dx
        col_x = self._check_collision(test_pos)
        if col_x:
            test_pos[0] -= dx

        test_pos[1] += dy
        col_y = self._check_collision(test_pos)
        if col_y:
            test_pos[1] -= dy

        collided = col_x or col_y
        reward_coll = 0.0

        if collided:
            self.num_collisions += 1
            self.last_v = 0.0
            self.last_w = steering
            reward_coll = REWARD_COLLISION
        else:
            self.last_v = lin_vel / ROBOT_SPEED_V
            self.last_w = steering

        self.agent_pos_m = test_pos
        self.agent_heading = new_heading

        self._noisy_pos_m = self.agent_pos_m.copy() + self.np_random.normal(
            0, POSITION_NOISE, 2
        )
        self._noisy_heading = self.agent_heading + self.np_random.normal(
            0, HEADING_NOISE
        )

        sensors, hit_points = _sensors.compute_sensors(
            self.agent_pos_m,
            self.agent_heading,
            self.last_v,
            self.last_w,
            self.frontier_map,
            self.render_offset,
            self.pixels_per_meter,
            self.grid_size_p,
            self.true_obstacle_map,
        )
        self._last_sensors = sensors
        _maps.update_obstacle_map_from_sensors(
            self.obstacle_map,
            self.virtual_wall_map,
            hit_points,
            self.grid_size_p,
        )

        new_cells = 0
        if not collided:
            new_cells, self._last_stamp_bbox = _maps.stamp_coverage(
                self.coverage_map,
                self.overlap_map,
                self.collision_map,
                old_pos,
                self.agent_pos_m,
                self.render_offset,
                self.pixels_per_meter,
                self.grid_size_p,
            )

        self.frontier_map = _maps.compute_frontier_map(
            self.coverage_map, self.collision_map
        )

        new_frontier_distance = _sensors.get_distance_to_closest_frontier(
            self.frontier_map,
            self.agent_pos_m,
            self.render_offset,
            self.pixels_per_meter,
            self.grid_size_p,
        )
        reward_frontier = 0.0
        if new_cells == 0 and not collided:
            dist_change = self.old_frontier_distance - new_frontier_distance
            max_dist_change = ROBOT_SPEED_V * DT
            progress_ratio = float(np.clip(dist_change / max_dist_change, -1.0, 1.0))
            target_penalty = abs(2 * REWARD_BASE_PENALTY)
            reward_frontier = progress_ratio * target_penalty
        self.old_frontier_distance = new_frontier_distance

        cov = self.coverage_map.copy()
        cov[self.coverable_area == 0] = 0
        self.coverage_in_pixels = int(cov.sum())
        self.coverage_in_percent = self.coverage_in_pixels / self.total_cells

        if new_cells > 0:
            self.non_new_steps = 0
        else:
            self.non_new_steps += 1

        reward_area = 0.0
        reward_tv = 0.0

        if not collided:
            max_new = 2 * ROBOT_RADIUS * ROBOT_SPEED_V * DT * self.pixels_per_meter**2
            if max_new > 0:
                reward_area = REWARD_AREA_SCALE * min(
                    new_cells / max_new, REWARD_AREA_MAX
                )

            local_cov_old = self.local_coverage_old
            local_obs_old = self.local_known_obstacles_old
            radius_m = ROBOT_RADIUS

            local_cov_new_aligned = _maps.get_local_crop(
                self.coverage_map,
                old_pos,
                radius_m,
                self.render_offset,
                self.pixels_per_meter,
                self.grid_size_p,
            )
            local_obs_new_aligned = _maps.get_local_crop(
                self.collision_map,
                old_pos,
                radius_m,
                self.render_offset,
                self.pixels_per_meter,
                self.grid_size_p,
            )

            if (
                local_cov_old is not None
                and local_obs_old is not None
                and local_cov_new_aligned.shape == local_cov_old.shape
            ):
                tv_new = total_variation(local_cov_new_aligned, local_obs_new_aligned)
                tv_old = total_variation(local_cov_old, local_obs_old)
                tv_diff = tv_new - tv_old
                self.global_tv += tv_diff
                reward_tv = -tv_diff
                reward_tv *= METERS_PER_PIXEL / DT / ROBOT_SPEED_V / 2.5
                reward_tv *= REWARD_TV_SCALE
                reward_tv = np.sign(reward_tv) * min(abs(reward_tv), REWARD_TV_MAX)

            self.local_coverage_old = _maps.get_local_crop(
                self.coverage_map,
                self.agent_pos_m,
                radius_m,
                self.render_offset,
                self.pixels_per_meter,
                self.grid_size_p,
            )
            self.local_known_obstacles_old = _maps.get_local_crop(
                self.collision_map,
                self.agent_pos_m,
                radius_m,
                self.render_offset,
                self.pixels_per_meter,
                self.grid_size_p,
            )

        reward_const = REWARD_BASE_PENALTY if new_cells > 0 else 2 * REWARD_BASE_PENALTY
        reward = reward_area + reward_tv + reward_coll + reward_const + reward_frontier

        # Termination
        cells_missed = self.total_cells - self.coverage_in_pixels
        terminated = cells_missed == 0

        truncated = False
        max_steps = PHASES[self.phase]["max_steps"]
        if self.current_step >= max_steps:
            truncated = True
        if self.non_new_steps >= MAX_NON_NEW_STEPS:
            truncated = True

        obs = self._get_obs()
        info = {
            "coverage_cells": self.coverage_in_pixels,
            "total_cells": self.total_cells,
            "phase": self.phase,
            "num_collisions": self.num_collisions,
            "coverage_percent": self.coverage_in_percent,
            "cells_missed": cells_missed,
            "is_success": bool(cells_missed < CELLS_MISSED_THRESHOLD),
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _get_obs(self):
        cov = np.tanh(0.2 * self.overlap_map)
        obs = {
            "coverage": get_multi_scale_map(
                cov,
                0,
                False,
                self.grid_size_p,
                self._noisy_heading,
                self._noisy_pos_m,
                self.render_offset,
                self.pixels_per_meter,
            ),
            "obstacles": get_multi_scale_map(
                self.obstacle_map,
                0,
                False,
                self.grid_size_p,
                self._noisy_heading,
                self._noisy_pos_m,
                self.render_offset,
                self.pixels_per_meter,
            ),
            "frontier": get_multi_scale_map(
                self.frontier_map,
                0,
                True,
                self.grid_size_p,
                self._noisy_heading,
                self._noisy_pos_m,
                self.render_offset,
                self.pixels_per_meter,
            ),
            "sensors": self._last_sensors,
        }
        for key in ("coverage", "obstacles", "frontier"):
            np.clip(obs[key], 0.0, 1.0, out=obs[key])
        return obs

    # ------------------------------------------------------------------
    # Curriculum
    # ------------------------------------------------------------------

    def set_phase(self, phase: int) -> None:
        self.phase = max(1, min(phase, 8))
        self._active_phase = self.phase

    def _sample_phase(self) -> int:
        current = self._active_phase
        phases = list(range(1, current + 1))
        weights = [PHASE_WEIGHT_DECAY ** (current - p) for p in phases]
        total = sum(weights)
        probs = [w / total for w in weights]
        return int(self.np_random.choice(phases, p=probs))

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def close_display(self) -> None:
        if self.window is not None:
            if self.render_mode == "human":
                pygame.display.quit()
            self.window = None

    def close(self) -> None:
        self.close_display()
        if pygame.get_init():
            pygame.quit()

    def render(self, toggles=None):
        if self.render_mode is None:
            return

        if toggles is None:
            toggles = getattr(
                self,
                "render_toggles",
                {"dilated": False, "stamped": False, "rays": False, "coverable": False},
            )

        if self.window is None:
            if not pygame.get_init():
                pygame.init()
            if self.render_mode == "human":
                self.window = pygame.display.set_mode(
                    (self.window_size, self.window_size)
                )
                pygame.display.set_caption("Robot Coverage v3")
            else:
                self.window = pygame.Surface((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

        frame = render_frame(self, toggles, self.window_size)

        # Blit frame onto the window surface
        frame_surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        self.window.blit(frame_surf, (0, 0))

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return frame
