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
REWARD_BASE_PENALTY = -0.05
REWARD_COLLISION = -5.0
REWARD_TV_SCALE = 1.0
REWARD_TV_MAX = 1.5
REWARD_AREA_SCALE = 1.5
REWARD_AREA_MAX = 2.0
ROBOT_SPEED_V = 0.26
ROBOT_SPEED_W = 1.0
DT = 0.5
METERS_PER_PIXEL = 0.1
NUM_MAPS = 3
MAP_SIZE = 32
SCALE_FACTOR = 4
SENSOR_DIM = 8
NUM_RAYS = 6
RAY_MAX_DIST = 1
POSITION_NOISE = 0.01
HEADING_NOISE = 0.05
OBSTACLE_DILATION = 1
ROBOT_RADIUS_PX = 5
SPAWN_SAFETY_RADIUS_PX = 1
MAX_FIELD_ATTEMPTS = 100
SUCCESS_WINDOW = 50
SUCCESS_THRESHOLD = 0.8
MAX_NON_NEW_STEPS = 1000

PHASES = {
    1: {"radii": (2.5, 7.5), "obst": (0, 4), "max_steps": 4000, "goal": 0.90},
    2: {"radii": (7.5, 10.0), "obst": (4, 10), "max_steps": 6000, "goal": 0.90},
    3: {"radii": (10.0, 12.0), "obst": (10, 16), "max_steps": 8000, "goal": 0.95},
    4: {"radii": (12.0, 14.0), "obst": (16, 24), "max_steps": 10000, "goal": 0.95},
    5: {"radii": (14.0, 16.0), "obst": (24, 32), "max_steps": 14000, "goal": 0.97},
    6: {"radii": (16.0, 18.0), "obst": (32, 40), "max_steps": 18000, "goal": 0.97},
    7: {"radii": (18.0, 20.0), "obst": (40, 50), "max_steps": 24000, "goal": 0.99},
    8: {"radii": (20.0, 24.0), "obst": (50, 60), "max_steps": 30000, "goal": 0.99},
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
                        [0.0] * 6 + [-1.0, -1.0],
                        dtype=np.float32,
                    ),
                    high=np.array(
                        [1.0] * 6 + [1.0, 1.0],
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
        self.position_history = deque(maxlen=11)
        self._last_swept_bbox = None  # legacy attribute, no longer populated
        self._last_stamp_bbox = (
            None  # (min_x, max_x, min_y, max_y) of last _stamp_coverage window
        )

        self.window_size = 800
        self.window = None
        self.clock = None
        self.render_offset = np.array([0.0, 0.0])

    def _generate_random_field(self):
        rule = PHASES[self.phase]
        radii_low, radii_high = rule["radii"]
        obst_min, obst_max = rule["obst"]

        while True:
            angles = np.sort(self.np_random.uniform(0, 2 * np.pi, 12))
            radii = self.np_random.uniform(radii_low, radii_high, 12)
            points = [(r * math.cos(a), r * math.sin(a)) for r, a in zip(radii, angles)]
            outer = Polygon(points).buffer(0.5).simplify(0.3)

            num_obstacles = (
                self.np_random.integers(obst_min, obst_max + 1) if obst_max > 0 else 0
            )
            obstacles = []
            margin = 1.0
            for _ in range(num_obstacles):
                lo, la, hi, ha = outer.bounds
                if hi - lo < 2 * margin or ha - la < 2 * margin:
                    break
                ox = self.np_random.uniform(lo + margin, hi - margin)
                oy = self.np_random.uniform(la + margin, ha - margin)
                obs_poly = (
                    Point(ox, oy).buffer(self.np_random.uniform(0.3, 1.0)).simplify(0.2)
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

            spawn_dist = ROBOT_RADIUS + 0.5 * ROBOT_SIDE
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

    def _compute_pre_dilated_map(self):
        robot_r_px = ROBOT_RADIUS_PX
        obs_for_dilation = self.true_obstacle_map.copy()
        obs_for_dilation[0, :] = 1
        obs_for_dilation[-1, :] = 1
        obs_for_dilation[:, 0] = 1
        obs_for_dilation[:, -1] = 1
        kernel = np.ones((robot_r_px * 2 + 1,) * 2, dtype=np.float32)
        self.pre_dilated_map = cv2.dilate(obs_for_dilation, kernel, iterations=1)

    def _compute_spawn_safety_map(self):
        safety_r_px = SPAWN_SAFETY_RADIUS_PX
        obs_for_dilation = self.true_obstacle_map.copy()
        obs_for_dilation[0, :] = 1
        obs_for_dilation[-1, :] = 1
        obs_for_dilation[:, 0] = 1
        obs_for_dilation[:, -1] = 1
        if safety_r_px > 0:
            kernel = np.ones((safety_r_px * 2 + 1,) * 2, dtype=np.float32)
            self.spawn_safety_map = cv2.dilate(obs_for_dilation, kernel, iterations=1)
        else:
            self.spawn_safety_map = obs_for_dilation

    def _compute_coverable_area(self):
        valid_positions = ((self.field_grid > 0) & (self.pre_dilated_map == 0)).astype(
            np.uint8
        )
        k = ROBOT_RADIUS_PX
        y, x = np.ogrid[-k : k + 1, -k : k + 1]
        kernel = ((x * x + y * y) <= k * k).astype(np.uint8)
        swept = cv2.dilate(valid_positions, kernel, iterations=1)
        self.coverable_area = (swept & (self.field_grid > 0)).astype(np.uint8)

    def _init_maps(self):
        self.obstacle_map = np.zeros(
            (self.grid_size_p, self.grid_size_p), dtype=np.float32
        )
        self.coverage_map = np.zeros(
            (self.grid_size_p, self.grid_size_p), dtype=np.float32
        )
        self.overlap_map = np.zeros(
            (self.grid_size_p, self.grid_size_p), dtype=np.float32
        )

        self._stamp_initial_coverage(self.agent_pos_m)
        self.frontier_map = self._compute_frontier_map()
        self._init_metrics()

    def _init_metrics(self):
        self.total_cells = max(int(self.coverable_area.sum()), 1)

        cov = self.coverage_map.copy()
        cov[self.coverable_area == 0] = 0
        self.coverage_in_pixels = int(cov.sum())
        self.coverage_in_percent = self.coverage_in_pixels / self.total_cells

        local_cov = self._get_local_crop(
            self.coverage_map, self.agent_pos_m, ROBOT_RADIUS
        )
        local_obs = self._get_local_crop(
            self.pre_dilated_map, self.agent_pos_m, ROBOT_RADIUS
        )
        self.global_tv = total_variation(local_cov, local_obs)

    def _m_to_p(self, pos_m):
        return (pos_m - self.render_offset) * self.pixels_per_meter

    def _p_to_m(self, pos_p):
        return pos_p / self.pixels_per_meter + self.render_offset

    def _m_to_grid_px(self, pos_m):
        pos_p = self._m_to_p(np.asarray(pos_m))
        return int(round(pos_p[0])), int(round(pos_p[1]))

    def _draw_robot_footprint_local(self, pos_m, heading, local_size=32):
        half = ROBOT_RADIUS
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        corners = []
        for lx, ly in [(-half, -half), (half, -half), (half, half), (-half, half)]:
            gx = pos_m[0] + lx * cos_h - ly * sin_h
            gy = pos_m[1] + lx * sin_h + ly * cos_h
            corners.append([gx, gy])
        corners = np.array(corners, dtype=np.float64)
        center_p = self._m_to_p(np.array(pos_m))
        corners_p = self._m_to_p(corners)
        local_offset = corners_p - center_p + local_size // 2
        footprint = np.zeros((local_size, local_size), dtype=np.uint8)
        cv2.fillConvexPoly(footprint, local_offset.astype(np.int32), 1)
        return footprint

    def _is_out_of_bounds(self, pos_m):
        pos_p = self._m_to_p(pos_m)
        r = ROBOT_RADIUS_PX
        x, y = int(pos_p[0]), int(pos_p[1])
        if x - r < 0 or x + r >= self.grid_size_p:
            return True
        if y - r < 0 or y + r >= self.grid_size_p:
            return True
        return self.field_grid[y, x] == 0

    def _is_obstacle_collision(self, pos_m):
        pos_p = self._m_to_p(pos_m)
        ix, iy = int(round(pos_p[0])), int(round(pos_p[1]))
        if ix < 0 or ix >= self.grid_size_p or iy < 0 or iy >= self.grid_size_p:
            return True
        return self.pre_dilated_map[iy, ix] > 0

    def _check_collision(self, pos_m):
        if self._is_out_of_bounds(pos_m):
            return True
        if self._is_obstacle_collision(pos_m):
            return True
        return False

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

    def _stamp_coverage(self, old_pos_m, new_pos_m, old_heading, new_heading):
        old_px = self._m_to_grid_px(old_pos_m)
        new_px = self._m_to_grid_px(new_pos_m)
        radius = ROBOT_RADIUS_PX

        min_x = max(0, min(old_px[0], new_px[0]) - radius - 1)
        max_x = min(self.grid_size_p, max(old_px[0], new_px[0]) + radius + 2)
        min_y = max(0, min(old_px[1], new_px[1]) - radius - 1)
        max_y = min(self.grid_size_p, max(old_px[1], new_px[1]) + radius + 2)
        self._last_stamp_bbox = (min_x, max_x, min_y, max_y)
        if min_x >= max_x or min_y >= max_y:
            return 0

        local_cov = self.coverage_map[min_y:max_y, min_x:max_x]
        local_overlap = self.overlap_map[min_y:max_y, min_x:max_x]
        local_obs = self.pre_dilated_map[min_y:max_y, min_x:max_x]

        local_h = max_y - min_y
        local_w = max_x - min_x
        local_mask = np.zeros((local_h, local_w), dtype=np.uint8)

        ox_local = old_px[0] - min_x
        oy_local = old_px[1] - min_y
        nx_local = new_px[0] - min_x
        ny_local = new_px[1] - min_y

        dist_sq = (nx_local - ox_local) ** 2 + (ny_local - oy_local) ** 2
        if dist_sq <= 2:
            cv2.circle(local_mask, (ox_local, oy_local), radius, 1, thickness=-1)
        else:
            cv2.line(
                local_mask,
                (ox_local, oy_local),
                (nx_local, ny_local),
                1,
                thickness=2 * radius,
            )

        local_mask[local_obs > 0] = 0

        new_pixels = int(np.logical_and(local_mask, (local_cov == 0)).sum())
        local_cov[:] = np.maximum(local_cov, local_mask.astype(np.float32))
        local_overlap[:] = local_overlap + local_mask.astype(np.float32)

        return new_pixels

    def _stamp_initial_coverage(self, pos_m):
        px = self._m_to_grid_px(pos_m)
        radius = ROBOT_RADIUS_PX

        min_x = max(0, px[0] - radius - 1)
        max_x = min(self.grid_size_p, px[0] + radius + 2)
        min_y = max(0, px[1] - radius - 1)
        max_y = min(self.grid_size_p, px[1] + radius + 2)
        self._last_stamp_bbox = (min_x, max_x, min_y, max_y)
        if min_x >= max_x or min_y >= max_y:
            return

        local_cov = self.coverage_map[min_y:max_y, min_x:max_x]
        local_overlap = self.overlap_map[min_y:max_y, min_x:max_x]
        local_obs = self.pre_dilated_map[min_y:max_y, min_x:max_x]
        local_h = max_y - min_y
        local_w = max_x - min_x
        local_mask = np.zeros((local_h, local_w), dtype=np.uint8)
        cv2.circle(
            local_mask,
            (px[0] - min_x, px[1] - min_y),
            radius,
            1,
            thickness=-1,
        )
        local_mask[local_obs > 0] = 0
        local_cov[:] = np.maximum(local_cov, local_mask.astype(np.float32))
        local_overlap[:] = local_overlap + local_mask.astype(np.float32)

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

    def _local_to_global(self, lx, ly):
        x, y = self.agent_pos_m
        theta = self.agent_heading
        gx = x + lx * math.cos(theta) - ly * math.sin(theta)
        gy = y + lx * math.sin(theta) + ly * math.cos(theta)
        return gx, gy

    def _cast_ray_pixel(self, origin_m, angle):
        origin_p = self._m_to_p(np.array(origin_m))
        sx, sy = int(origin_p[0]), int(origin_p[1])
        max_steps = int(RAY_MAX_DIST / METERS_PER_PIXEL)
        dx = math.cos(angle)
        dy = math.sin(angle)
        ex = sx + int(round(dx * max_steps))
        ey = sy + int(round(dy * max_steps))

        abs_dx = abs(ex - sx)
        abs_dy = abs(ey - sy)
        if abs_dx == 0 and abs_dy == 0:
            return RAY_MAX_DIST, None
        step_x = 1 if ex > sx else -1
        step_y = 1 if ey > sy else -1
        err = abs_dx - abs_dy

        cx, cy = sx, sy
        for _ in range(max_steps + 1):
            if 0 <= cx < self.grid_size_p and 0 <= cy < self.grid_size_p:
                if self.true_obstacle_map[cy, cx] > 0:
                    hit_m = cx * METERS_PER_PIXEL + self.render_offset[0]
                    hit_my = cy * METERS_PER_PIXEL + self.render_offset[1]
                    dist = math.hypot(hit_m - origin_m[0], hit_my - origin_m[1])
                    return min(dist, RAY_MAX_DIST), (cx, cy)
            else:
                return RAY_MAX_DIST, None
            e2 = 2 * err
            if e2 > -abs_dy:
                err -= abs_dy
                cx += step_x
            if e2 < abs_dx:
                err += abs_dx
                cy += step_y
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

        normalized_dists = [d / RAY_MAX_DIST for d in dists]
        sensors = np.array(
            normalized_dists + [self.last_v, self.last_w],
            dtype=np.float32,
        )
        return sensors, hit_points

    def _update_obstacle_map_from_sensors(self, hit_points):
        for hp in hit_points:
            if hp is not None:
                ix, iy = hp
                if 0 <= ix < self.grid_size_p and 0 <= iy < self.grid_size_p:
                    r = ROBOT_RADIUS_PX
                    y1 = max(0, iy - r)
                    y2 = min(self.grid_size_p, iy + r + 1)
                    x1 = max(0, ix - r)
                    x2 = min(self.grid_size_p, ix + r + 1)
                    self.obstacle_map[y1:y2, x1:x2] = 1

    def _get_local_crop(self, world_map, pos_m, radius_m):
        pos_p = self._m_to_p(pos_m)
        r = int(radius_m * self.pixels_per_meter) + 10
        y1 = max(0, int(pos_p[1]) - r)
        y2 = min(self.grid_size_p, int(pos_p[1]) + r + 1)
        x1 = max(0, int(pos_p[0]) - r)
        x2 = min(self.grid_size_p, int(pos_p[0]) + r + 1)
        return world_map[y1:y2, x1:x2].copy()

    def _get_obs(self):
        cov = np.tanh(0.2 * self.overlap_map)
        obs = {
            "coverage": self._get_multi_scale_map(cov, 0),
            "obstacles": self._get_multi_scale_map(self.obstacle_map, 0),
            "frontier": self._get_multi_scale_map(self.frontier_map, 0),
            "sensors": self._last_sensors,
        }
        for key in ("coverage", "obstacles", "frontier"):
            np.clip(obs[key], 0.0, 1.0, out=obs[key])
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.non_new_steps = 0
        self.num_collisions = 0
        self.last_v = 0.0
        self.last_w = 0.0

        self.field = self._generate_random_field()
        self._rasterize_field()
        self.true_obstacle_map = (1 - self.field_grid).astype(np.float32)
        self._compute_pre_dilated_map()
        self._compute_spawn_safety_map()
        self._compute_coverable_area()
        self._get_safe_spawn()
        for _ in range(11):
            self.position_history.append((self.agent_pos_m.copy(), self.agent_heading))
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
            "coverage_percent": self.coverage_in_percent,
            "num_collisions": self.num_collisions,
            "phase": self.phase,
        }
        return obs, info

    def step(self, action):
        self.current_step += 1
        throttle = float(np.clip(action[0], -1, 1))
        steering = float(np.clip(action[1], -1, 1))

        lin_vel = throttle * ROBOT_SPEED_V
        lin_vel *= 1 - abs(steering) * 0.5
        ang_vel = steering * ROBOT_SPEED_W

        old_pos = self.agent_pos_m.copy()
        old_heading = self.agent_heading

        self.position_history.append((self.agent_pos_m.copy(), self.agent_heading))

        new_heading = (self.agent_heading + ang_vel * DT) % (2 * math.pi)
        inter_heading = self.agent_heading + ang_vel * DT / 2
        dx = lin_vel * DT * math.cos(inter_heading)
        dy = lin_vel * DT * math.sin(inter_heading)
        new_pos = self.agent_pos_m + np.array([dx, dy])

        collided = self._check_collision(new_pos)
        reward_coll = 0.0

        if collided:
            self.num_collisions += 1
            self.last_v = 0.0
            self.last_w = 0.0
            reward_coll = REWARD_COLLISION
            self.agent_pos_m, self.agent_heading = self.position_history[0]
            self.position_history.popleft()
        else:
            self.agent_pos_m = new_pos
            self.agent_heading = new_heading
            self.last_v = lin_vel / ROBOT_SPEED_V
            self.last_w = steering

        self._noisy_pos_m = self.agent_pos_m.copy() + np.random.normal(
            0, POSITION_NOISE, 2
        )
        self._noisy_heading = self.agent_heading + np.random.normal(0, HEADING_NOISE)

        sensors, hit_points = self._compute_sensors()
        self._last_sensors = sensors
        self._update_obstacle_map_from_sensors(hit_points)

        new_cells = 0
        if not collided:
            new_cells = self._stamp_coverage(
                old_pos, self.agent_pos_m, old_heading, self.agent_heading
            )

        self.frontier_map = self._compute_frontier_map()

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
            local_cov_new = self._get_local_crop(
                self.coverage_map, self.agent_pos_m, radius_m
            )
            local_obs_new = self._get_local_crop(
                self.pre_dilated_map, self.agent_pos_m, radius_m
            )

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

        if new_cells > 0:
            reward_const = 0.0
        else:
            reward_const = REWARD_BASE_PENALTY
        reward = reward_area + reward_tv + reward_coll + reward_const

        terminated = False
        goal = PHASES[self.phase]["goal"]
        if self.coverage_in_percent >= goal:
            terminated = True

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

    def render(self):
        if self.render_mode is None:
            return

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

        ws = self.window_size
        canvas = pygame.Surface((ws, ws))
        canvas.fill((30, 30, 30))

        if self.grid_size_p <= 0 or self.field is None:
            self.window.blit(canvas, (0, 0))
            if self.render_mode == "human":
                pygame.display.flip()
                self.clock.tick(self.metadata["render_fps"])
            return

        pad = 5.0
        minx, miny, maxx, maxy = self.field.bounds
        width = (maxx - minx) + 2 * pad
        height = (maxy - miny) + 2 * pad
        scl = ws / max(width, height)
        off = np.array([minx - pad, miny - pad])

        def to_screen(wx, wy):
            px = int((wx - off[0]) * scl)
            py = int(ws - (wy - off[1]) * scl)
            return px, py

        img = np.full((self.grid_size_p, self.grid_size_p, 3), 30, dtype=np.uint8)
        img[self.field_grid > 0] = [220, 220, 220]
        img[self.coverage_map > 0] = [80, 160, 80]
        img[self.obstacle_map > 0] = [200, 80, 80]

        img = cv2.resize(img, (ws, ws), interpolation=cv2.INTER_NEAREST)
        img = cv2.cvtColor(img[::-1], cv2.COLOR_BGR2RGB)
        screen_arr = np.transpose(img, (1, 0, 2))
        surf = pygame.surfarray.make_surface(screen_arr)
        canvas.blit(surf, (0, 0))

        ext_points = [to_screen(x, y) for x, y in self.field.exterior.coords]
        pygame.draw.polygon(canvas, (0, 0, 0), ext_points, 2)
        for interior in self.field.interiors:
            in_points = [to_screen(x, y) for x, y in interior.coords]
            pygame.draw.polygon(canvas, (255, 255, 255), in_points)
            pygame.draw.polygon(canvas, (255, 0, 0), in_points, 1)

        corners = self._get_square_corners(self.agent_pos_m, self.agent_heading)
        corner_pg = [to_screen(c[0], c[1]) for c in corners]
        pygame.draw.polygon(canvas, (50, 50, 200), corner_pg)

        hx = self.agent_pos_m[0] + ROBOT_RADIUS * math.cos(self.agent_heading)
        hy = self.agent_pos_m[1] + ROBOT_RADIUS * math.sin(self.agent_heading)
        pygame.draw.line(
            canvas, (0, 255, 0), to_screen(*self.agent_pos_m), to_screen(hx, hy), 2
        )

        self.window.blit(canvas, (0, 0))

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(pygame.surfarray.array3d(self.window), (1, 0, 2))
