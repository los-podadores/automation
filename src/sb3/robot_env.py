import math
from collections import deque

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from shapely.affinity import rotate, translate
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

MAX_STEPS = 15000
REWARD_BASE_PENALTY = -0.02
REWARD_CRASH_PENALTY = -40.0
REWARD_NEW_COVERAGE = 0.055
REWARD_FORWARD = 0.03
ROBOT_SPEED_V = 1.5
ROBOT_SPEED_W = 1.0
DT = 0.1
MAX_FIELD_ATTEMPTS = 100

PHASE_RULES = {
    1: {"radii_bounds": (10.0, 15.0), "obst_range": (0, 0)},
    2: {"radii_bounds": (15.0, 20.0), "obst_range": (5, 10)},
    3: {"radii_bounds": (20.0, 30.0), "obst_range": (15, 25)},
}


class RobotCoverageEnv(gym.Env):
    """
    Custom Environment that follows gymnasium interface.
    Agent is a rectangular robot navigating a randomized polygon field.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, a=2.0, b=1.0, render_mode=None, phase=1):
        super(RobotCoverageEnv, self).__init__()

        self.a = a
        self.b = b
        self.max_ray_dist = b + 0.05
        self.render_mode = render_mode
        self.phase = phase

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.observation_space = spaces.Dict(
            {
                "visual": spaces.Box(
                    low=0, high=255, shape=(2, 64, 64), dtype=np.uint8
                ),
                "sensors": spaces.Box(
                    low=np.array(
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0],
                        dtype=np.float32,
                    ),
                    high=np.array(
                        [
                            self.max_ray_dist,
                            self.max_ray_dist,
                            self.max_ray_dist,
                            self.max_ray_dist,
                            self.max_ray_dist,
                            self.max_ray_dist,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                        ],
                        dtype=np.float32,
                    ),
                    dtype=np.float32,
                ),
            }
        )

        self.field = None
        self.agent_pos = None
        self.agent_theta = 0.0
        self.last_v = 0.0
        self.last_w = 0.0

        self.grid_resolution = min(a, b) / 4.0
        self.visited_grid = None
        self.obstacle_grid = None
        self.grid_origin = (0, 0)
        self.grid_shape = (1, 1)
        self.total_cells = 1

        self.max_steps = MAX_STEPS
        self.current_step = 0

        self.window_size = 800
        self.window = None
        self.clock = None
        self.render_scale = 1.0
        self.render_offset_x = 0.0
        self.render_offset_y = 0.0

    def _init_grids(self, nav_field):
        minx, miny, maxx, maxy = nav_field.bounds
        res = self.grid_resolution
        nx = int((maxx - minx) / res) + 1
        ny = int((maxy - miny) / res) + 1
        self.grid_origin = (minx, miny)
        self.grid_shape = (nx, ny)
        self.visited_grid = np.zeros((nx, ny), dtype=np.bool_)
        self.obstacle_grid = np.zeros((nx, ny), dtype=np.bool_)

    def _world_to_grid(self, wx, wy):
        ox, oy = self.grid_origin
        res = self.grid_resolution
        gx = int((wx - ox) / res)
        gy = int((wy - oy) / res)
        return gx, gy

    def _in_bounds(self, gx, gy):
        nx, ny = self.grid_shape
        return 0 <= gx < nx and 0 <= gy < ny

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.last_v = 0.0
        self.last_w = 0.0

        for _ in range(MAX_FIELD_ATTEMPTS):
            self.field = self._generate_random_field()
            pos, theta = self._get_safe_spawn()
            if self._validate_field_accessibility(self.field, pos):
                self.agent_pos = pos
                self.agent_theta = theta
                break
        else:
            raise RuntimeError(
                f"Failed to generate accessible field after {MAX_FIELD_ATTEMPTS} attempts"
            )

        erosion = self.b / 2.0
        nav_field = self.field.buffer(-erosion)
        self._init_grids(nav_field)
        self._compute_total_cells(nav_field)

        minx, miny, maxx, maxy = self.field.bounds
        self.field_center = ((minx + maxx) / 2.0, (miny + maxy) / 2.0)
        self.field_radius = math.hypot(maxx - minx, maxy - miny) / 2.0

        pad = 5.0
        width = (maxx - minx) + 2 * pad
        height = (maxy - miny) + 2 * pad
        self.render_scale = self.window_size / max(width, height)
        self.render_offset_x = minx - pad
        self.render_offset_y = miny - pad

        self._update_coverage()

        obs = self._get_observation()
        info = {"coverage_cells": int(self.visited_grid.sum()), "total_cells": self.total_cells}

        return obs, info

    def step(self, action):
        self.current_step += 1

        self.last_v = action[0]
        self.last_w = action[1]
        v = (action[0] + 1.0) / 2.0 * ROBOT_SPEED_V
        w = action[1] * ROBOT_SPEED_W
        dt = DT

        self.agent_theta += w * dt
        self.agent_pos = (
            self.agent_pos[0] + v * math.cos(self.agent_theta) * dt,
            self.agent_pos[1] + v * math.sin(self.agent_theta) * dt,
        )

        body_poly = self._get_agent_polygon()
        crashed = not self.field.contains(body_poly)

        reward = REWARD_BASE_PENALTY + REWARD_FORWARD * v
        terminated = False

        if crashed:
            reward += REWARD_CRASH_PENALTY
            terminated = True
        else:
            new_cells = self._update_coverage()
            reward += REWARD_NEW_COVERAGE * new_cells

            if self.visited_grid.sum() > 0.95 * self.total_cells:
                terminated = True

        obs = self._get_observation()
        truncated = self.current_step >= self.max_steps
        info = {"coverage_cells": int(self.visited_grid.sum()), "total_cells": self.total_cells}

        return obs, float(reward), terminated, truncated, info

    def _generate_random_field(self):
        phase_rule = PHASE_RULES[self.phase]
        radii_low, radii_high = phase_rule["radii_bounds"]
        obst_min, obst_max = phase_rule["obst_range"]

        while True:
            angles = np.sort(self.np_random.uniform(0, 2 * np.pi, 12))
            radii = self.np_random.uniform(radii_low, radii_high, 12)

            points = [
                (r * math.cos(ang), r * math.sin(ang)) for r, ang in zip(radii, angles)
            ]
            outer_field = Polygon(points)
            outer_field = outer_field.buffer(2.0).simplify(1.0)

            num_obstacles = (
                self.np_random.integers(obst_min, obst_max + 1) if obst_max > 0 else 0
            )
            obstacles = []
            for _ in range(num_obstacles):
                ox = self.np_random.uniform(
                    outer_field.bounds[0] + 5, outer_field.bounds[2] - 5
                )
                oy = self.np_random.uniform(
                    outer_field.bounds[1] + 5, outer_field.bounds[3] - 5
                )
                obs_poly = (
                    Point(ox, oy).buffer(self.np_random.uniform(1.0, 3.0)).simplify(0.5)
                )

                if outer_field.contains(obs_poly):
                    obstacles.append(obs_poly)

            final_field = outer_field
            for obs in obstacles:
                final_field = final_field.difference(obs)

            if not isinstance(final_field, MultiPolygon):
                return final_field

    def _validate_field_accessibility(self, field, spawn_pos):
        erosion = self.b / 2.0
        nav_field = field.buffer(-erosion)
        if nav_field.is_empty:
            return False
        if isinstance(nav_field, MultiPolygon):
            return False

        minx, miny, maxx, maxy = nav_field.bounds
        res = self.grid_resolution
        nx = int((maxx - minx) / res) + 1
        ny = int((maxy - miny) / res) + 1

        grid = np.zeros((nx, ny), dtype=np.bool_)
        for i in range(nx):
            for j in range(ny):
                cx = minx + (i + 0.5) * res
                cy = miny + (j + 0.5) * res
                grid[i, j] = nav_field.contains(Point(cx, cy))

        total_free = int(grid.sum())
        if total_free == 0:
            return False

        si = int((spawn_pos[0] - minx) / res)
        sj = int((spawn_pos[1] - miny) / res)
        si = max(0, min(nx - 1, si))
        sj = max(0, min(ny - 1, sj))
        if not grid[si, sj]:
            nearest_free = np.argwhere(grid)
            dists = np.abs(nearest_free - np.array([si, sj])).sum(axis=1)
            idx = nearest_free[dists.argmin()]
            si, sj = int(idx[0]), int(idx[1])

        visited = np.zeros((nx, ny), dtype=np.bool_)
        queue = deque()
        queue.append((si, sj))
        visited[si, sj] = True
        reachable = 0

        while queue:
            ci, cj = queue.popleft()
            reachable += 1
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ni, nj = ci + di, cj + dj
                if (
                    0 <= ni < nx
                    and 0 <= nj < ny
                    and not visited[ni, nj]
                    and grid[ni, nj]
                ):
                    visited[ni, nj] = True
                    queue.append((ni, nj))

        if reachable < total_free * 0.95:
            return False

        return True

    def _get_safe_spawn(self):
        while True:
            boundary_coords = list(self.field.exterior.coords)
            num_edges = len(boundary_coords) - 1
            edge_idx = self.np_random.integers(0, num_edges)

            x1, y1 = boundary_coords[edge_idx]
            x2, y2 = boundary_coords[(edge_idx + 1) % num_edges]

            t = self.np_random.uniform(0.25, 0.25)
            px = x1 + t * (x2 - x1)
            py = y1 + t * (y2 - y1)

            dx = x2 - x1
            dy = y2 - y1
            edge_len = math.hypot(dx, dy)
            nx = -dy / edge_len
            ny = dx / edge_len

            inward = self.field.representative_point()
            if nx * (inward.x - px) + ny * (inward.y - py) < 0:
                nx, ny = -nx, -ny

            spawn_dist = self.a
            x = px + nx * spawn_dist
            y = py + ny * spawn_dist

            theta = math.atan2(dy, dx)

            self.agent_pos = (x, y)
            self.agent_theta = theta

            body_poly = self._get_agent_polygon()
            if self.field.contains(body_poly):
                obs = self._get_observation()
                if all(d > 1.0 for d in obs["sensors"][:6]):
                    return (x, y), theta

    def _get_agent_polygon(self):
        a, b = self.a, self.b
        rect = Polygon(
            [(-a / 2, -b / 2), (a / 2, -b / 2), (a / 2, b / 2), (-a / 2, b / 2)]
        )
        rect = rotate(rect, self.agent_theta, use_radians=True, origin=(0, 0))
        rect = translate(rect, self.agent_pos[0], self.agent_pos[1])
        return rect

    def _compute_total_cells(self, nav_field):
        if nav_field.is_empty:
            self.total_cells = 1
            return
        minx, miny, maxx, maxy = nav_field.bounds
        res = self.grid_resolution
        count = 0
        x = minx
        while x < maxx:
            y = miny
            while y < maxy:
                if nav_field.contains(Point(x + res / 2, y + res / 2)):
                    count += 1
                y += res
            x += res
        self.total_cells = max(count, 1)

    def _get_observation(self):
        a, b = self.a, self.b
        x, y = self.agent_pos
        theta = self.agent_theta

        def local_to_global(lx, ly):
            gx = x + lx * math.cos(theta) - ly * math.sin(theta)
            gy = y + lx * math.sin(theta) + ly * math.cos(theta)
            return Point(gx, gy)

        def local_dir_to_global(angle_offset):
            ang = theta + angle_offset
            return math.cos(ang), math.sin(ang)

        origins = [
            local_to_global(a / 2, b / 2),
            local_to_global(a / 2, 0),
            local_to_global(a / 2, -b / 2),
            local_to_global(0, b / 2),
            local_to_global(0, -b / 2),
            local_to_global(-a / 2, 0),
        ]
        angles = [math.pi / 4, 0.0, -math.pi / 4, math.pi / 2, -math.pi / 2, math.pi]

        dists = []
        hit_points = []
        for origin, ang in zip(origins, angles):
            dx, dy = local_dir_to_global(ang)
            end = Point(
                origin.x + dx * self.max_ray_dist,
                origin.y + dy * self.max_ray_dist,
            )
            ray = LineString([origin, end])
            dist, hit = self._calculate_ray_distance(ray, origin)
            dists.append(dist)
            hit_points.append(hit)

        min_front = min(dists[0], dists[1], dists[2]) / self.max_ray_dist
        asymmetry = (dists[0] - dists[2]) / self.max_ray_dist
        center_d = dists[1]
        min_front_raw = min(dists[0], dists[1], dists[2])
        wall_angle = math.atan2(dists[0] - dists[2], center_d - min_front_raw) / math.pi

        sensor_obs = np.array(
            [
                dists[0],
                dists[1],
                dists[2],
                dists[3],
                dists[4],
                dists[5],
                min_front,
                asymmetry,
                wall_angle,
                self.last_v,
                self.last_w,
            ],
            dtype=np.float32,
        )

        grid_size = 64
        visual_obs = np.zeros((2, grid_size, grid_size), dtype=np.uint8)

        half_grid = grid_size // 2
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        res = self.grid_resolution
        ox, oy = self.grid_origin

        for hp in hit_points:
            if hp is not None:
                gcx, gcy = self._world_to_grid(hp.x, hp.y)
                if self._in_bounds(gcx, gcy):
                    self.obstacle_grid[gcx, gcy] = True

        obs_cells = np.argwhere(self.obstacle_grid)
        if len(obs_cells) > 0:
            wx = (obs_cells[:, 0] + 0.5) * res + ox
            wy = (obs_cells[:, 1] + 0.5) * res + oy
            dx = wx - x
            dy = wy - y
            lx = (dx * cos_t + dy * sin_t) / res + half_grid
            ly = (-dx * sin_t + dy * cos_t) / res + half_grid
            valid = (lx >= 0) & (lx < grid_size) & (ly >= 0) & (ly < grid_size)
            ix = lx[valid].astype(np.int32)
            iy = ly[valid].astype(np.int32)
            visual_obs[0, ix, iy] = 255

        vis_cells = np.argwhere(self.visited_grid)
        if len(vis_cells) > 0:
            wx = (vis_cells[:, 0] + 0.5) * res + ox
            wy = (vis_cells[:, 1] + 0.5) * res + oy
            dx = wx - x
            dy = wy - y
            lx = (dx * cos_t + dy * sin_t) / res + half_grid
            ly = (-dx * sin_t + dy * cos_t) / res + half_grid
            valid = (lx >= 0) & (lx < grid_size) & (ly >= 0) & (ly < grid_size)
            ix = lx[valid].astype(np.int32)
            iy = ly[valid].astype(np.int32)
            visual_obs[1, ix, iy] = 255

        return {
            "visual": visual_obs,
            "sensors": sensor_obs,
        }

    def _calculate_ray_distance(self, ray, origin):
        boundary = self.field.boundary
        intersection = ray.intersection(boundary)

        if intersection.is_empty:
            return self.max_ray_dist, None
        if isinstance(intersection, Point):
            return origin.distance(intersection), intersection
        try:
            pts = list(intersection.geoms)
            closest = min(pts, key=lambda pt: origin.distance(pt))
            return origin.distance(closest), closest
        except AttributeError:
            return self.max_ray_dist, None

    def _update_coverage(self):
        body_poly = self._get_agent_polygon()
        minx, miny, maxx, maxy = body_poly.bounds
        grid_minx = int(minx // self.grid_resolution)
        grid_maxx = int(maxx // self.grid_resolution)
        grid_miny = int(miny // self.grid_resolution)
        grid_maxy = int(maxy // self.grid_resolution)

        new_count = 0
        res = self.grid_resolution
        ox, oy = self.grid_origin
        for i in range(grid_minx, grid_maxx + 1):
            for j in range(grid_miny, grid_maxy + 1):
                cx = (i + 0.5) * res
                cy = (j + 0.5) * res
                if body_poly.contains(Point(cx, cy)):
                    gi, gj = self._world_to_grid(cx, cy)
                    if self._in_bounds(gi, gj) and not self.visited_grid[gi, gj]:
                        self.visited_grid[gi, gj] = True
                        new_count += 1

        return new_count

    def _to_pygame(self, x, y):
        """Converts standard Cartesian math coordinates to Pygame Screen coordinates."""
        px = int((x - self.render_offset_x) * self.render_scale)
        py = int(self.window_size - (y - self.render_offset_y) * self.render_scale)
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
                pygame.display.set_caption("Robot Coverage Agent")
            else:
                self.window = pygame.Surface((self.window_size, self.window_size))

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        ex_points = [self._to_pygame(x, y) for x, y in self.field.exterior.coords]
        pygame.draw.polygon(canvas, (220, 220, 220), ex_points)

        rect_size = max(1, int(self.grid_resolution * self.render_scale))
        ox, oy = self.grid_origin
        res = self.grid_resolution
        vis_cells = np.argwhere(self.visited_grid)
        for idx in vis_cells:
            gx, gy = int(idx[0]), int(idx[1])
            px = (gx * res) + ox
            py = ((gy + 1) * res) + oy
            pg_coords = self._to_pygame(px, py)
            pygame.draw.rect(
                canvas,
                (100, 220, 100),
                (pg_coords[0], pg_coords[1], rect_size, rect_size),
            )

        pygame.draw.polygon(canvas, (0, 0, 0), ex_points, 3)
        for interior in self.field.interiors:
            in_points = [self._to_pygame(x, y) for x, y in interior.coords]
            pygame.draw.polygon(canvas, (255, 255, 255), in_points)
            pygame.draw.polygon(canvas, (255, 0, 0), in_points, 2)

        body = self._get_agent_polygon()
        body_points = [self._to_pygame(x, y) for x, y in body.exterior.coords]
        pygame.draw.polygon(canvas, (50, 50, 200), body_points)

        hx = self.agent_pos[0] + (self.a / 2) * math.cos(self.agent_theta)
        hy = self.agent_pos[1] + (self.a / 2) * math.sin(self.agent_theta)
        pygame.draw.line(
            canvas,
            (0, 255, 0),
            self._to_pygame(*self.agent_pos),
            self._to_pygame(hx, hy),
            3,
        )

        a, b = self.a, self.b
        theta = self.agent_theta
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        def local_to_global_pt(lx, ly):
            return (
                self.agent_pos[0] + lx * cos_t - ly * sin_t,
                self.agent_pos[1] + lx * sin_t + ly * cos_t,
            )

        origins = [
            local_to_global_pt(a / 2, b / 2),
            local_to_global_pt(a / 2, 0),
            local_to_global_pt(a / 2, -b / 2),
            local_to_global_pt(0, b / 2),
            local_to_global_pt(0, -b / 2),
            local_to_global_pt(-a / 2, 0),
        ]
        angles = [math.pi / 4, 0.0, -math.pi / 4, math.pi / 2, -math.pi / 2, math.pi]
        colors = [
            (255, 165, 0),
            (0, 255, 200),
            (255, 165, 0),
            (100, 100, 255),
            (100, 100, 255),
            (200, 200, 200),
        ]

        for origin, ang_off, color in zip(origins, angles, colors):
            ang = theta + ang_off
            end = (
                origin[0] + math.cos(ang) * self.max_ray_dist,
                origin[1] + math.sin(ang) * self.max_ray_dist,
            )
            pygame.draw.line(
                canvas,
                color,
                self._to_pygame(*origin),
                self._to_pygame(*end),
                1,
            )
            pygame.draw.circle(canvas, color, self._to_pygame(*origin), 3)

        if self.render_mode == "human":
            pygame.event.pump()
            if self.current_step % 4 == 0:
                self.window.blit(canvas, canvas.get_rect())
                pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def set_phase(self, phase):
        self.phase = phase

    def close_display(self):
        """Safely closes only the Pygame window, leaving the core environment intact."""
        if self.window is not None:
            if self.render_mode == "human":
                pygame.display.quit()
            self.window = None

    def close(self):
        """Full teardown for the end of the script."""
        self.close_display()
        if pygame.get_init():
            pygame.quit()
