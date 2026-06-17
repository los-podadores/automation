import math
from collections import deque

import cv2
import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from shapely.geometry import MultiPolygon, Point, Polygon
from utils import total_variation

ROBOT_SIDE = 1.0
ROBOT_RADIUS = ROBOT_SIDE / 2.0
MAX_STEPS = 15000
REWARD_BASE_PENALTY = -0.1
REWARD_COLLISION = -10.0
REWARD_TV_SCALE = 1.0
REWARD_TV_MAX = 5.0
REWARD_AREA_SCALE = 1.0
REWARD_AREA_MAX = 2.0
ROBOT_SPEED_V = 0.26
ROBOT_SPEED_W = 1.0
DT = 0.5
METERS_PER_PIXEL = 0.0375
NUM_MAPS = 4
MAP_SIZE = 32
SCALE_FACTOR = 4
SENSOR_DIM = 11
NUM_RAYS = 6
RAY_MAX_DIST = 1
POSITION_NOISE = 0.01
HEADING_NOISE = 0.05
OBSTACLE_DILATION = 9
REWIND_STEPS = 5
MAX_FIELD_ATTEMPTS = 100
SUCCESS_WINDOW = 50
SUCCESS_THRESHOLD = 0.8

PHASES = {
    1: {"radii": (5, 8), "obst": (0, 2), "max_steps": 2000, "goal": 0.90},
    2: {"radii": (8, 11), "obst": (2, 5), "max_steps": 3000, "goal": 0.90},
    3: {"radii": (11, 14), "obst": (5, 8), "max_steps": 4000, "goal": 0.95},
    4: {"radii": (14, 17), "obst": (8, 12), "max_steps": 5000, "goal": 0.95},
    5: {"radii": (17, 20), "obst": (12, 16), "max_steps": 7000, "goal": 0.97},
    6: {"radii": (20, 24), "obst": (16, 20), "max_steps": 9000, "goal": 0.97},
    7: {"radii": (24, 28), "obst": (20, 25), "max_steps": 12000, "goal": 0.99},
    8: {"radii": (28, 32), "obst": (25, 30), "max_steps": 15000, "goal": 0.99},
}


class RobotCoverageEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, phase=1):
        super().__init__()
        self.render_mode = render_mode
        self.phase = phase

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
                        [0.0] * 6 + [0.0, -1.0, -1.0, -1.0, -1.0],
                        dtype=np.float32,
                    ),
                    high=np.array(
                        [RAY_MAX_DIST] * 6 + [1.0, 1.0, 1.0, 1.0, 1.0],
                        dtype=np.float32,
                    ),
                    dtype=np.float32,
                ),
            }
        )

        self.field = None
        self.field_grid = None
        self.agent_pos = None
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
        self.head_vec_old = None
        self.orth_vec_old = None

        self.rewind_history = deque(maxlen=REWIND_STEPS)

        self.window_size = 800
        self.window = None
        self.clock = None
        self.render_scale = 1.0
        self.render_offset = np.array([0.0, 0.0])

    # ------------------------------------------------------------------ #
    #  Field generation (Shapely random polygons, same as v1)
    # ------------------------------------------------------------------ #

    def _generate_random_field(self):
        rule = PHASES[self.phase]
        radii_low, radii_high = rule["radii"]
        obst_min, obst_max = rule["obst"]

        while True:
            angles = np.sort(self.np_random.uniform(0, 2 * np.pi, 12))
            radii = self.np_random.uniform(radii_low, radii_high, 12)
            points = [(r * math.cos(a), r * math.sin(a)) for r, a in zip(radii, angles)]
            outer = Polygon(points).buffer(2.0).simplify(1.0)

            num_obstacles = (
                self.np_random.integers(obst_min, obst_max + 1) if obst_max > 0 else 0
            )
            obstacles = []
            for _ in range(num_obstacles):
                ox = self.np_random.uniform(outer.bounds[0] + 5, outer.bounds[2] - 5)
                oy = self.np_random.uniform(outer.bounds[1] + 5, outer.bounds[3] - 5)
                obs_poly = (
                    Point(ox, oy).buffer(self.np_random.uniform(1.0, 3.0)).simplify(0.5)
                )
                if outer.contains(obs_poly):
                    obstacles.append(obs_poly)

            field = outer
            for obs in obstacles:
                field = field.difference(obs)

            if not isinstance(field, MultiPolygon):
                return field

    def _validate_field(self, field, spawn_pos):
        erosion = ROBOT_RADIUS
        nav = field.buffer(-erosion)
        if nav.is_empty or isinstance(nav, MultiPolygon):
            return False
        return True

    def _get_safe_spawn(self):
        for _ in range(100):
            boundary = list(self.field.exterior.coords)
            num_edges = len(boundary) - 1
            edge_idx = self.np_random.integers(0, num_edges)
            x1, y1 = boundary[edge_idx]
            x2, y2 = boundary[(edge_idx + 1) % num_edges]
            t = self.np_random.uniform(0.25, 0.75)
            px = x1 + t * (x2 - x1)
            py = y1 + t * (y2 - y1)

            dx, dy = x2 - x1, y2 - y1
            edge_len = math.hypot(dx, dy)
            nx, ny = -dy / edge_len, dx / edge_len
            inward = self.field.representative_point()
            if nx * (inward.x - px) + ny * (inward.y - py) < 0:
                nx, ny = -nx, -ny

            spawn_dist = ROBOT_SIDE
            x = px + nx * spawn_dist
            y = py + ny * spawn_dist
            theta = math.atan2(dy, dx)

            self.agent_pos_m = np.array([x, y], dtype=np.float64)
            self.agent_heading = theta

            if self._is_valid_pose():
                return

        raise RuntimeError("Failed to find valid spawn position")

    def _is_valid_pose(self):
        if self._is_obstacle_collision(self.agent_pos_m):
            return False
        if self._is_out_of_bounds(self.agent_pos_m):
            return False
        return True

    # ------------------------------------------------------------------ #
    #  Grid rasterization
    # ------------------------------------------------------------------ #

    def _rasterize_field(self):
        minx, miny, maxx, maxy = self.field.bounds
        pad = 5.0
        self.grid_size_m = max(maxx - minx, maxy - miny) + 2 * pad
        self.grid_size_p = max(1, int(self.grid_size_m * self.pixels_per_meter))
        self.render_offset = np.array([minx - pad, miny - pad])

        self.field_grid = np.zeros((self.grid_size_p, self.grid_size_p), dtype=np.uint8)
        exterior = np.array(self.field.exterior.coords, dtype=np.float32)
        exterior_px = ((exterior - self.render_offset) * self.pixels_per_meter).astype(
            np.int32
        )
        cv2.fillPoly(self.field_grid, [exterior_px], 1)

        for interior in self.field.interiors:
            hole = np.array(interior.coords, dtype=np.float32)
            hole_px = ((hole - self.render_offset) * self.pixels_per_meter).astype(
                np.int32
            )
            cv2.fillPoly(self.field_grid, [hole_px], 0)

    def _init_maps(self):
        self.obstacle_map = np.zeros(
            (self.grid_size_p, self.grid_size_p), dtype=np.float32
        )
        # Mark field boundary as known from the start
        boundary = cv2.Canny((self.field_grid * 255).astype(np.uint8), 50, 150)
        self.obstacle_map[boundary > 0] = 1.0
        self.coverage_map = np.zeros(
            (self.grid_size_p, self.grid_size_p), dtype=np.float32
        )
        self.overlap_map = np.zeros(
            (self.grid_size_p, self.grid_size_p), dtype=np.float32
        )
        self._update_coverage_at(self.agent_pos_m)
        self.frontier_map = self._compute_frontier_map()
        self._init_metrics()

    def _init_metrics(self):
        all_obs = self._get_dilated_obstacles()
        free = self.field_grid.astype(np.float32)
        free[all_obs > 0] = 0
        self.total_cells = max(int(free.sum()), 1)

        cov = self.coverage_map.copy()
        cov[all_obs > 0] = 0
        self.coverage_in_pixels = int(cov.sum())
        self.coverage_in_percent = self.coverage_in_pixels / self.total_cells

        local_cov = self._get_local_crop(
            self.coverage_map, self.agent_pos_m, ROBOT_RADIUS
        )
        local_obs = self._get_local_crop(all_obs, self.agent_pos_m, ROBOT_RADIUS)
        self.global_tv = total_variation(local_cov, local_obs)

    def _get_dilated_obstacles(self):
        obs = self.true_obstacle_map.copy()
        if OBSTACLE_DILATION > 1:
            k = np.ones((OBSTACLE_DILATION,) * 2, dtype=np.float32)
            obs[0, :] = 1
            obs[-1, :] = 1
            obs[:, 0] = 1
            obs[:, -1] = 1
            obs = cv2.dilate(obs, k, iterations=1)
        return obs

    # ------------------------------------------------------------------ #
    #  Coordinate helpers
    # ------------------------------------------------------------------ #

    def _m_to_p(self, pos_m):
        return (pos_m - self.render_offset) * self.pixels_per_meter

    def _p_to_m(self, pos_p):
        return pos_p / self.pixels_per_meter + self.render_offset

    def _is_out_of_bounds(self, pos_m):
        pos_p = self._m_to_p(pos_m)
        r = int(ROBOT_RADIUS * self.pixels_per_meter)
        x, y = int(pos_p[0]), int(pos_p[1])
        if x - r < 0 or x + r >= self.grid_size_p:
            return True
        if y - r < 0 or y + r >= self.grid_size_p:
            return True
        return self.field_grid[y, x] == 0

    def _is_obstacle_collision(self, pos_m):
        pos_p = self._m_to_p(pos_m)
        r = int(ROBOT_RADIUS * self.pixels_per_meter)
        cx, cy = int(pos_p[0]), int(pos_p[1])
        y1 = max(0, cy - r)
        y2 = min(self.grid_size_p, cy + r + 1)
        x1 = max(0, cx - r)
        x2 = min(self.grid_size_p, cx + r + 1)
        if y1 >= y2 or x1 >= x2:
            return True
        local_obs = self.true_obstacle_map[y1:y2, x1:x2]
        local_circle = np.zeros_like(local_obs, dtype=np.uint8)
        local_cx = cx - x1
        local_cy = cy - y1
        cv2.circle(local_circle, (local_cx, local_cy), r, 1, cv2.FILLED)
        return bool(np.logical_and(local_circle, local_obs).any())

    def _check_collision(self, pos_m):
        if self._is_out_of_bounds(pos_m):
            return True
        if self._is_obstacle_collision(pos_m):
            return True
        return False

    # ------------------------------------------------------------------ #
    #  Coverage computation (swept area for square robot)
    # ------------------------------------------------------------------ #

    def _get_square_corners(self, pos_m, heading):
        half = ROBOT_RADIUS
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        corners_local = [
            (-half, -half),
            (half, -half),
            (half, half),
            (-half, half),
        ]
        corners = []
        for lx, ly in corners_local:
            gx = pos_m[0] + lx * cos_h - ly * sin_h
            gy = pos_m[1] + lx * sin_h + ly * cos_h
            corners.append([gx, gy])
        return np.array(corners, dtype=np.float64)

    def _update_coverage_at(self, pos_m):
        corners = self._get_square_corners(pos_m, self.agent_heading)
        corners_p = self._m_to_p(corners).astype(np.int32)
        mask = np.zeros((self.grid_size_p, self.grid_size_p), dtype=np.uint8)
        cv2.fillConvexPoly(mask, corners_p, 1)
        new_cells = int(np.logical_and(mask, (self.coverage_map == 0)).sum())
        self.coverage_map = np.maximum(self.coverage_map, mask.astype(np.float32))
        self.overlap_map += mask.astype(np.float32)
        return new_cells

    def _compute_swept_coverage(self, old_pos_m, new_pos_m, old_heading, new_heading):
        c1 = self._get_square_corners(old_pos_m, old_heading)
        c2 = self._get_square_corners(new_pos_m, new_heading)
        all_pts = np.vstack([c1, c2])
        hull = cv2.convexHull(all_pts.reshape(-1, 1, 2).astype(np.float32))
        hull_pts = hull.reshape(-1, 2).astype(np.int32)
        hull_p = self._m_to_p(hull_pts)
        mask = np.zeros((self.grid_size_p, self.grid_size_p), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull_p.astype(np.int32), 1)
        new_cells = int(np.logical_and(mask, (self.coverage_map == 0)).sum())
        self.coverage_map = np.maximum(self.coverage_map, mask.astype(np.float32))
        self.overlap_map += mask.astype(np.float32)
        return new_cells

    # ------------------------------------------------------------------ #
    #  Frontier map
    # ------------------------------------------------------------------ #

    def _compute_frontier_map(self):
        cov = self.coverage_map.copy()
        obs = self.obstacle_map.copy()
        if OBSTACLE_DILATION > 1:
            k = np.ones((OBSTACLE_DILATION,) * 2, dtype=np.float32)
            obs[0, :] = 1
            obs[-1, :] = 1
            obs[:, 0] = 1
            obs[:, -1] = 1
            obs = cv2.dilate(obs, k, iterations=1)
        cov[obs > 0] = 0
        free = (cov + obs) == 0
        k3 = np.ones((3, 3), dtype=np.float32)
        cov_dilated = cv2.dilate(cov, k3, iterations=1)
        return (np.logical_and(cov_dilated, free)).astype(np.float32)

    # ------------------------------------------------------------------ #
    #  Multi-scale egocentric maps
    # ------------------------------------------------------------------ #

    def _get_transform_matrix(self, scale):
        heading_deg = self._noisy_heading * 180 / math.pi
        noisy_p = self._m_to_p(self._noisy_pos_m)

        t1 = np.eye(3)
        t1[0, 2] = -noisy_p[0] / scale
        t1[1, 2] = -noisy_p[1] / scale

        rot = np.eye(3)
        rot[:2] = cv2.getRotationMatrix2D(
            center=(0, 0), angle=90 - heading_deg, scale=1
        )

        t2 = np.eye(3)
        t2[0, 2] = MAP_SIZE / 2
        t2[1, 2] = MAP_SIZE / 2

        return t2 @ rot @ t1

    def _get_relative_map(self, world_map, pad_value, scale=1):
        sc = min(scale, self.grid_size_p)
        matrix = self._get_transform_matrix(sc)
        downsampled = cv2.resize(
            world_map,
            (int(0.5 + self.grid_size_p / sc),) * 2,
            interpolation=cv2.INTER_AREA,
        )
        warped = cv2.warpAffine(
            downsampled,
            M=matrix[:2],
            dsize=(MAP_SIZE,) * 2,
            borderValue=pad_value,
            flags=cv2.INTER_AREA,
        )
        return warped

    def _get_multi_scale_map(self, world_map, pad_value):
        ms = np.zeros((NUM_MAPS, MAP_SIZE, MAP_SIZE), dtype=np.float32)
        for i in range(NUM_MAPS):
            ms[i] = self._get_relative_map(world_map, pad_value, SCALE_FACTOR**i)
        return ms

    # ------------------------------------------------------------------ #
    #  Sensor (6 rays, same as v1)
    # ------------------------------------------------------------------ #

    def _local_to_global(self, lx, ly):
        x, y = self.agent_pos_m
        theta = self.agent_heading
        gx = x + lx * math.cos(theta) - ly * math.sin(theta)
        gy = y + lx * math.sin(theta) + ly * math.cos(theta)
        return gx, gy

    def _cast_ray_pixel(self, origin_m, angle):
        step = METERS_PER_PIXEL * 0.5
        dx = math.cos(angle)
        dy = math.sin(angle)
        for i in range(1, int(RAY_MAX_DIST / step) + 1):
            px = origin_m[0] + dx * step * i
            py = origin_m[1] + dy * step * i
            pp = self._m_to_p(np.array([px, py]))
            ix, iy = int(pp[0]), int(pp[1])
            if ix < 0 or ix >= self.grid_size_p or iy < 0 or iy >= self.grid_size_p:
                return RAY_MAX_DIST, None
            if self.true_obstacle_map[iy, ix] > 0:
                dist = math.hypot(px - origin_m[0], py - origin_m[1])
                return min(dist, RAY_MAX_DIST), (ix, iy)
        return RAY_MAX_DIST, None

    def _compute_sensors(self):
        a = ROBOT_SIDE
        b = ROBOT_SIDE
        origins_local = [
            (a / 2, b / 2),
            (a / 2, 0),
            (a / 2, -b / 2),
            (0, b / 2),
            (0, -b / 2),
            (-a / 2, 0),
        ]
        angles_offset = [
            math.pi / 4,
            0.0,
            -math.pi / 4,
            math.pi / 2,
            -math.pi / 2,
            math.pi,
        ]

        dists = []
        hit_points = []
        for (lx, ly), ang_off in zip(origins_local, angles_offset):
            ox, oy = self._local_to_global(lx, ly)
            dist, hit = self._cast_ray_pixel(
                np.array([ox, oy]), self.agent_heading + ang_off
            )
            dists.append(dist)
            hit_points.append(hit)

        min_front = min(dists[0], dists[1], dists[2]) / RAY_MAX_DIST
        asymmetry = (dists[0] - dists[2]) / RAY_MAX_DIST
        center_d = dists[1]
        min_front_raw = min(dists[0], dists[1], dists[2])
        denom = center_d - min_front_raw
        if abs(denom) < 1e-6:
            wall_angle = 0.0
        else:
            wall_angle = math.atan2(dists[0] - dists[2], denom) / math.pi

        sensors = np.array(
            dists + [min_front, asymmetry, wall_angle, self.last_v, self.last_w],
            dtype=np.float32,
        )
        return sensors, hit_points

    # ------------------------------------------------------------------ #
    #  Obstacle map update from 6-ray hit points
    # ------------------------------------------------------------------ #

    def _update_obstacle_map_from_sensors(self, hit_points):
        for hp in hit_points:
            if hp is not None:
                ix, iy = hp
                if 0 <= ix < self.grid_size_p and 0 <= iy < self.grid_size_p:
                    r = 3
                    y1 = max(0, iy - r)
                    y2 = min(self.grid_size_p, iy + r + 1)
                    x1 = max(0, ix - r)
                    x2 = min(self.grid_size_p, ix + r + 1)
                    self.obstacle_map[y1:y2, x1:x2] = 1

    # ------------------------------------------------------------------ #
    #  Local crop helpers
    # ------------------------------------------------------------------ #

    def _get_local_crop(self, world_map, pos_m, radius_m):
        pos_p = self._m_to_p(pos_m)
        r = int(radius_m * self.pixels_per_meter) + 10
        y1 = max(0, int(pos_p[1]) - r)
        y2 = min(self.grid_size_p, int(pos_p[1]) + r + 1)
        x1 = max(0, int(pos_p[0]) - r)
        x2 = min(self.grid_size_p, int(pos_p[0]) + r + 1)
        return world_map[y1:y2, x1:x2].copy()

    # ------------------------------------------------------------------ #
    #  Observation
    # ------------------------------------------------------------------ #

    def _get_obs(self):
        cov = np.tanh(0.2 * self.overlap_map)
        obs = {
            "coverage": self._get_multi_scale_map(cov, 0),
            "obstacles": self._get_multi_scale_map(self.obstacle_map, 0),
            "frontier": self._get_multi_scale_map(self.frontier_map, 0),
            "sensors": self._last_sensors,
        }
        return obs

    # ------------------------------------------------------------------ #
    #  Pose history for rewind
    # ------------------------------------------------------------------ #

    def _save_pose(self):
        self.rewind_history.append(
            (
                self.agent_pos_m.copy(),
                self.agent_heading,
                self.last_v,
                self.last_w,
            )
        )

    def _rewind_pose(self):
        if len(self.rewind_history) > 0:
            pos, heading, v, w = self.rewind_history.popleft()
            self.agent_pos_m = pos.copy()
            self.agent_heading = heading
            self.last_v = 0.0
            self.last_w = 0.0

    # ------------------------------------------------------------------ #
    #  gym interface
    # ------------------------------------------------------------------ #

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.non_new_steps = 0
        self.num_collisions = 0
        self.last_v = 0.0
        self.last_w = 0.0
        self.rewind_history.clear()

        self.field = self._generate_random_field()
        self._rasterize_field()
        self.true_obstacle_map = (1 - self.field_grid).astype(np.float32)
        self._get_safe_spawn()
        self._init_maps()

        self._noisy_pos_m = self.agent_pos_m.copy() + np.random.normal(
            0, POSITION_NOISE, 2
        )
        self._noisy_heading = self.agent_heading + np.random.normal(0, HEADING_NOISE)

        sensors, hit_points = self._compute_sensors()
        self._last_sensors = sensors
        self._update_obstacle_map_from_sensors(hit_points)

        obs = self._get_obs()
        info = {
            "coverage_cells": self.coverage_in_pixels,
            "total_cells": self.total_cells,
            "phase": self.phase,
        }
        return obs, info

    def step(self, action):
        self.current_step += 1
        throttle = float(np.clip(action[0], -1, 1))
        steering = float(np.clip(action[1], -1, 1))

        throttle = (throttle + 1.0) / 2.0
        lin_vel = throttle * ROBOT_SPEED_V
        lin_vel *= 1 - abs(steering) * 0.5
        ang_vel = steering * ROBOT_SPEED_W

        self.last_v = lin_vel / ROBOT_SPEED_V
        self.last_w = steering

        old_pos = self.agent_pos_m.copy()
        old_heading = self.agent_heading

        new_heading = (self.agent_heading + ang_vel * DT) % (2 * math.pi)
        inter_heading = self.agent_heading + ang_vel * DT / 2
        dx = lin_vel * DT * math.cos(inter_heading)
        dy = lin_vel * DT * math.sin(inter_heading)
        new_pos = self.agent_pos_m + np.array([dx, dy])

        collided = self._check_collision(new_pos)
        reward_coll = 0.0
        self._save_pose()

        if collided:
            self.num_collisions += 1
            self._rewind_pose()
            reward_coll = REWARD_COLLISION
        else:
            self.agent_pos_m = new_pos
            self.agent_heading = new_heading

        self._noisy_pos_m = self.agent_pos_m.copy() + np.random.normal(
            0, POSITION_NOISE, 2
        )
        self._noisy_heading = self.agent_heading + np.random.normal(0, HEADING_NOISE)

        sensors, hit_points = self._compute_sensors()
        self._last_sensors = sensors
        self._update_obstacle_map_from_sensors(hit_points)

        new_cells = 0
        if not collided:
            new_cells = self._compute_swept_coverage(
                old_pos, self.agent_pos_m, old_heading, self.agent_heading
            )

        self.frontier_map = self._compute_frontier_map()

        all_obs = self._get_dilated_obstacles()
        cov = self.coverage_map.copy()
        cov[all_obs > 0] = 0
        self.coverage_in_pixels = int(cov.sum())
        self.coverage_in_percent = self.coverage_in_pixels / self.total_cells

        if new_cells > 0:
            self.non_new_steps = 0
        else:
            self.non_new_steps += 1

        # --- Rewards --- #
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
            local_cov_new = self._get_local_crop(
                self.coverage_map, self.agent_pos_m, radius_m
            )
            local_obs_new = self._get_local_crop(all_obs, self.agent_pos_m, radius_m)

            if local_cov_old is not None and local_obs_old is not None:
                if local_cov_new.shape == local_cov_old.shape:
                    tv_new = total_variation(local_cov_new, local_obs_new)
                    tv_old = total_variation(local_cov_old, local_obs_old)
                    tv_diff = tv_new - tv_old
                    self.global_tv += tv_diff
                    reward_tv = -tv_diff
                    reward_tv *= METERS_PER_PIXEL / DT / ROBOT_SPEED_V / 2.5
                    reward_tv = np.sign(reward_tv) * min(abs(reward_tv), REWARD_TV_MAX)
                    reward_tv *= REWARD_TV_SCALE

            self.local_coverage_old = local_cov_new
            self.local_known_obstacles_old = local_obs_new

        reward = reward_area + reward_tv + reward_coll + REWARD_BASE_PENALTY

        terminated = False
        goal = PHASES[self.phase]["goal"]
        if self.coverage_in_percent >= goal:
            terminated = True

        truncated = False
        max_steps = PHASES[self.phase]["max_steps"]
        if self.current_step >= max_steps:
            truncated = True
        if self.non_new_steps >= 1000:
            truncated = True

        obs = self._get_obs()
        info = {
            "coverage_cells": self.coverage_in_pixels,
            "total_cells": self.total_cells,
            "phase": self.phase,
            "num_collisions": self.num_collisions,
            "coverage_percent": self.coverage_in_percent,
        }
        return obs, reward, terminated, truncated, info

    def set_phase(self, phase):
        self.phase = max(1, min(phase, 8))

    def close_display(self):
        if self.window is not None:
            if self.render_mode == "human":
                pygame.display.quit()
            self.window = None

    def close(self):
        self.close_display()
        if pygame.get_init():
            pygame.quit()

    # ------------------------------------------------------------------ #
    #  Rendering (Pygame)
    # ------------------------------------------------------------------ #

    def _to_pygame(self, x, y):
        px = int((x - self.render_offset[0]) * self.render_scale)
        py = int(self.window_size - (y - self.render_offset[1]) * self.render_scale)
        return px, py

    def render(self):
        if self.render_mode is None:
            return

        if self.window is None:
            if not pygame.get_init():
                pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.window = pygame.display.set_mode(
                    (self.window_size, self.window_size)
                )
                pygame.display.set_caption("Robot Coverage v2")
            else:
                self.window = pygame.Surface((self.window_size, self.window_size))

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        pad = 5.0
        minx, miny, maxx, maxy = self.field.bounds
        width = (maxx - minx) + 2 * pad
        height = (maxy - miny) + 2 * pad
        self.render_scale = self.window_size / max(width, height)

        # Draw field fill
        ext_pts = [self._to_pygame(x, y) for x, y in self.field.exterior.coords]
        pygame.draw.polygon(canvas, (220, 220, 220), ext_pts)

        # Draw covered cells
        res = 1.0 / self.pixels_per_meter
        vis_cells = np.argwhere(self.coverage_map > 0)
        rect_size = max(1, int(res * self.render_scale))
        for idx in vis_cells:
            gy, gx = int(idx[0]), int(idx[1])
            px = gx * res + self.render_offset[0]
            py = (gy + 1) * res + self.render_offset[1]
            pg = self._to_pygame(px, py)
            pygame.draw.rect(
                canvas, (100, 220, 100), (pg[0], pg[1], rect_size, rect_size)
            )

        # Draw field boundary
        pygame.draw.polygon(canvas, (0, 0, 0), ext_pts, 3)
        for interior in self.field.interiors:
            in_pts = [self._to_pygame(x, y) for x, y in interior.coords]
            pygame.draw.polygon(canvas, (255, 255, 255), in_pts)
            pygame.draw.polygon(canvas, (255, 0, 0), in_pts, 2)

        # Draw obstacle cells
        obs_cells = np.argwhere(self.obstacle_map > 0)
        for idx in obs_cells:
            gy, gx = int(idx[0]), int(idx[1])
            px = gx * res + self.render_offset[0]
            py = (gy + 1) * res + self.render_offset[1]
            pg = self._to_pygame(px, py)
            pygame.draw.rect(canvas, (80, 80, 80), (pg[0], pg[1], rect_size, rect_size))

        # Draw robot
        corners = self._get_square_corners(self.agent_pos_m, self.agent_heading)
        corner_pg = [self._to_pygame(c[0], c[1]) for c in corners]
        pygame.draw.polygon(canvas, (50, 50, 200), corner_pg)

        # Heading arrow
        hx = self.agent_pos_m[0] + ROBOT_RADIUS * math.cos(self.agent_heading)
        hy = self.agent_pos_m[1] + ROBOT_RADIUS * math.sin(self.agent_heading)
        pygame.draw.line(
            canvas,
            (0, 255, 0),
            self._to_pygame(*self.agent_pos_m),
            self._to_pygame(hx, hy),
            3,
        )

        # Draw 6 sensor rays
        a = ROBOT_SIDE
        b = ROBOT_SIDE
        theta = self.agent_heading
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        def local_pt(lx, ly):
            return (
                self.agent_pos_m[0] + lx * cos_t - ly * sin_t,
                self.agent_pos_m[1] + lx * sin_t + ly * cos_t,
            )

        origins = [
            local_pt(a / 2, b / 2),
            local_pt(a / 2, 0),
            local_pt(a / 2, -b / 2),
            local_pt(0, b / 2),
            local_pt(0, -b / 2),
            local_pt(-a / 2, 0),
        ]
        angles_off = [
            math.pi / 4,
            0.0,
            -math.pi / 4,
            math.pi / 2,
            -math.pi / 2,
            math.pi,
        ]
        colors = [
            (255, 165, 0),
            (0, 255, 200),
            (255, 165, 0),
            (100, 100, 255),
            (100, 100, 255),
            (200, 200, 200),
        ]

        for origin, ang_off, color in zip(origins, angles_off, colors):
            ang = theta + ang_off
            end = (
                origin[0] + math.cos(ang) * RAY_MAX_DIST,
                origin[1] + math.sin(ang) * RAY_MAX_DIST,
            )
            pygame.draw.line(
                canvas, color, self._to_pygame(*origin), self._to_pygame(*end), 1
            )
            pygame.draw.circle(canvas, color, self._to_pygame(*origin), 3)

        # HUD
        font = pygame.font.SysFont(None, 28)
        hud_lines = [
            f"Step: {self.current_step}",
            f"Coverage: {self.coverage_in_percent:.1%}",
            f"Phase: {self.phase}",
            f"Collisions: {self.num_collisions}",
        ]
        for i, line in enumerate(hud_lines):
            text = font.render(line, True, (0, 0, 0))
            canvas.blit(text, (10, 10 + i * 24))

        if self.render_mode == "human":
            pygame.event.pump()
            if self.current_step % 4 == 0:
                self.window.blit(canvas, canvas.get_rect())
                pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
