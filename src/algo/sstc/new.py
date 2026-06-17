"""
Spiral Spanning Tree Coverage - Continuous Control Adaptation

Adapts the SSTC algorithm for continuous robot control in unknown environments.
The robot builds its world map from sensor data and replans SSTC paths as
new areas are discovered.

Based on: Spiral-STC by Gabriely et al.
Original implementation: src/algo/sstc/original.py
Environment reference: src/v2/robot_env.py
"""

import collections
import math

import cv2
import numpy as np
import pygame
from shapely.geometry import MultiPolygon, Point, Polygon

# --------------------------------------------------------------------------- #
#  Constants
# --------------------------------------------------------------------------- #

ROBOT_SIDE = 1.0
ROBOT_RADIUS = ROBOT_SIDE / 2.0
ROBOT_SPEED_V = 0.8
ROBOT_SPEED_W = 1.0
DT = 0.3
METERS_PER_PIXEL = 0.0375
RAY_MAX_DIST = 3.0

SUPER_CELL_SIZE = 2.0 * ROBOT_SIDE
BASE_CELL_SIZE = ROBOT_SIDE
WALL_DILATION = 1
MIN_SUPER_CELLS_FOR_SSTC = 4

MAX_STEPS = 15000
COVERAGE_GOAL = 0.90
NO_PROGRESS_LIMIT = 500

WINDOW_SIZE = 800
FPS = 60

COLOR_BG = (20, 24, 30)
COLOR_FIELD = (220, 220, 220)
COLOR_BOUNDARY = (0, 0, 0)
COLOR_ROBOT = (50, 50, 200)
COLOR_COVERED = (100, 220, 100)
COLOR_FREE = (100, 100, 100)
COLOR_WALL = (120, 60, 60)
COLOR_PATH = (241, 196, 15)
COLOR_RAY_HIT = (231, 76, 60)
COLOR_RAY_CLEAR = (46, 204, 113)
COLOR_TEXT = (236, 240, 241)
COLOR_SPANNING = (155, 89, 182)


# --------------------------------------------------------------------------- #
#  Field Generation
# --------------------------------------------------------------------------- #

def generate_random_field(rng):
    while True:
        angles = np.sort(rng.uniform(0, 2 * np.pi, 12))
        radii = rng.uniform(8, 14, 12)
        points = [(r * math.cos(a), r * math.sin(a)) for r, a in zip(radii, angles)]
        outer = Polygon(points).buffer(2.0).simplify(1.0)

        obstacles = []
        for _ in range(rng.integers(2, 6)):
            ox = rng.uniform(outer.bounds[0] + 5, outer.bounds[2] - 5)
            oy = rng.uniform(outer.bounds[1] + 5, outer.bounds[3] - 5)
            obs = Point(ox, oy).buffer(rng.uniform(1.0, 3.0)).simplify(0.5)
            if outer.contains(obs):
                obstacles.append(obs)

        field = outer
        for obs in obstacles:
            field = field.difference(obs)
        if not isinstance(field, MultiPolygon):
            return field


def rasterize_field(field, pixels_per_meter, offset):
    minx, miny, maxx, maxy = field.bounds
    pad = 5.0
    grid_size_m = max(maxx - minx, maxy - miny) + 2 * pad
    grid_size_p = max(1, int(grid_size_m * pixels_per_meter))

    grid = np.zeros((grid_size_p, grid_size_p), dtype=np.uint8)
    ext = np.array(field.exterior.coords, dtype=np.float32)
    cv2.fillPoly(grid, [((ext - offset) * pixels_per_meter).astype(np.int32)], 1)
    for interior in field.interiors:
        hole = np.array(interior.coords, dtype=np.float32)
        cv2.fillPoly(grid, [((hole - offset) * pixels_per_meter).astype(np.int32)], 0)
    return grid, grid_size_p


# --------------------------------------------------------------------------- #
#  Internal Map
# --------------------------------------------------------------------------- #

class InternalMap:
    UNKNOWN = 0
    FREE = 1
    COVERED = 2
    WALL = 3

    def __init__(self, field_grid, cell_size, offset, pixels_per_meter):
        self.cell_size = cell_size
        self.offset = offset
        self.ppm = pixels_per_meter

        field_h, field_w = field_grid.shape
        self.grid_h = int(math.ceil(field_h / pixels_per_meter / cell_size)) + 2
        self.grid_w = int(math.ceil(field_w / pixels_per_meter / cell_size)) + 2

        self.grid = np.full((self.grid_h, self.grid_w), self.UNKNOWN, dtype=np.int8)
        self.free_count = 0
        self.covered_count = 0

        self._inside = np.zeros((self.grid_h, self.grid_w), dtype=bool)
        for gy in range(self.grid_h):
            for gx in range(self.grid_w):
                cx = gx * cell_size + cell_size / 2 + offset[0]
                cy = gy * cell_size + cell_size / 2 + offset[1]
                fgx = int((cx - offset[0]) * pixels_per_meter)
                fgy = int((cy - offset[1]) * pixels_per_meter)
                if 0 <= fgx < field_w and 0 <= fgy < field_h:
                    self._inside[gy, gx] = field_grid[fgy, fgx] == 1

    def to_grid(self, pos_m):
        return int((pos_m[0] - self.offset[0]) / self.cell_size), \
               int((pos_m[1] - self.offset[1]) / self.cell_size)

    def to_world(self, gx, gy):
        return np.array([
            gx * self.cell_size + self.cell_size / 2 + self.offset[0],
            gy * self.cell_size + self.cell_size / 2 + self.offset[1],
        ])

    def in_bounds(self, gx, gy):
        return 0 <= gx < self.grid_w and 0 <= gy < self.grid_h

    def mark_ray(self, origin_m, angle, dist, is_hit):
        step = self.cell_size * 0.25
        dx, dy = math.cos(angle), math.sin(angle)
        num_steps = int(dist / step) + 1

        for i in range(num_steps):
            px = origin_m[0] + dx * step * i
            py = origin_m[1] + dy * step * i
            gx, gy = self.to_grid(np.array([px, py]))
            if self.in_bounds(gx, gy) and self.grid[gy, gx] == self.UNKNOWN:
                if not self._inside[gy, gx]:
                    self.grid[gy, gx] = self.WALL
                else:
                    self.grid[gy, gx] = self.FREE
                    self.free_count += 1

        if is_hit:
            gx, gy = self.to_grid(np.array([
                origin_m[0] + dx * dist, origin_m[1] + dy * dist]))
            if self.in_bounds(gx, gy) and self.grid[gy, gx] != self.COVERED:
                if self.grid[gy, gx] == self.FREE:
                    self.free_count -= 1
                self.grid[gy, gx] = self.WALL
                for ddy in range(-WALL_DILATION, WALL_DILATION + 1):
                    for ddx in range(-WALL_DILATION, WALL_DILATION + 1):
                        nx, ny = gx + ddx, gy + ddy
                        if self.in_bounds(nx, ny) and self.grid[ny, nx] == self.UNKNOWN:
                            self.grid[ny, nx] = self.WALL

    def mark_coverage(self, pos_m, heading):
        half = ROBOT_RADIUS
        cos_h, sin_h = math.cos(heading), math.sin(heading)
        x1 = max(0, int((pos_m[0] - half - self.offset[0]) / self.cell_size))
        y1 = max(0, int((pos_m[1] - half - self.offset[1]) / self.cell_size))
        x2 = min(self.grid_w, int((pos_m[0] + half - self.offset[0]) / self.cell_size) + 1)
        y2 = min(self.grid_h, int((pos_m[1] + half - self.offset[1]) / self.cell_size) + 1)

        for gy in range(y1, y2):
            for gx in range(x1, x2):
                if not self._inside[gy, gx]:
                    continue
                cx = gx * self.cell_size + self.cell_size / 2 + self.offset[0]
                cy = gy * self.cell_size + self.cell_size / 2 + self.offset[1]
                lx = (cx - pos_m[0]) * cos_h + (cy - pos_m[1]) * sin_h
                ly = -(cx - pos_m[0]) * sin_h + (cy - pos_m[1]) * cos_h
                if abs(lx) <= half and abs(ly) <= half:
                    if self.grid[gy, gx] == self.FREE:
                        self.free_count -= 1
                    if self.grid[gy, gx] != self.WALL:
                        if self.grid[gy, gx] != self.COVERED:
                            self.covered_count += 1
                        self.grid[gy, gx] = self.COVERED

    def get_coverage_percent(self):
        total = self.free_count + self.covered_count
        return self.covered_count / max(total, 1)

    def count_valid_super_cells(self, sstc_h, sstc_w):
        count = 0
        for si in range(sstc_h):
            for sj in range(sstc_w):
                if self._is_valid_sc(si, sj):
                    count += 1
        return count

    def _is_valid_sc(self, si, sj):
        sstc_h = self.grid_h // 2
        sstc_w = self.grid_w // 2
        if si < 0 or si >= sstc_h or sj < 0 or sj >= sstc_w:
            return False
        for dy in range(2):
            for dx in range(2):
                gy, gx = si * 2 + dy, sj * 2 + dx
                if not self._inside[gy, gx]:
                    return False
                if self.grid[gy, gx] in (self.WALL, self.UNKNOWN):
                    return False
        return True

    def find_frontier(self, pos_m):
        gx, gy = self.to_grid(pos_m)
        best_dist = float('inf')
        best_world = None

        visited = set()
        queue = collections.deque([(gx, gy, 0)])
        visited.add((gx, gy))

        while queue:
            cx, cy, depth = queue.popleft()
            if depth > 100:
                break
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if (nx, ny) in visited or not self.in_bounds(nx, ny):
                    continue
                visited.add((nx, ny))
                state = self.grid[ny, nx]
                if state == self.UNKNOWN:
                    for adx, ady in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        ax, ay = nx + adx, ny + ady
                        if self.in_bounds(ax, ay) and self.grid[ay, ax] in (self.FREE, self.COVERED):
                            dist = math.hypot(nx - gx, ny - gy)
                            if dist < best_dist:
                                best_dist = dist
                                best_world = self.to_world(nx, ny)
                            break
                elif state != self.WALL:
                    queue.append((nx, ny, depth + 1))

        return best_world


# --------------------------------------------------------------------------- #
#  SSTC Planner
# --------------------------------------------------------------------------- #

class SSTCPlanner:
    def __init__(self, imap):
        self.imap = imap
        self.sstc_h = imap.grid_h // 2
        self.sstc_w = imap.grid_w // 2
        self.edge_list = []
        self.waypoints = []
        self.current_waypoint_idx = 0
        self._steps_on_waypoint = 0
        self._wp_start_pos = None

    def find_start(self, pos_m):
        gx, gy = self.imap.to_grid(pos_m)
        si, sj = gy // 2, gx // 2

        for radius in range(0, min(self.sstc_h, self.sstc_w)):
            for di in range(-radius, radius + 1):
                for dj in range(-radius, radius + 1):
                    if abs(di) != radius and abs(dj) != radius:
                        continue
                    ni, nj = si + di, sj + dj
                    if self.imap._is_valid_sc(ni, nj):
                        return ni, nj
        return None

    def plan(self, start_si, start_sj):
        if not self.imap._is_valid_sc(start_si, start_sj):
            return False

        visit = np.zeros((self.sstc_h, self.sstc_w), dtype=int)
        visit[start_si][start_sj] = 1
        route = []
        self.edge_list = []
        self._stc(start_si, start_sj, visit, route)

        if len(route) < 2:
            return False

        self.waypoints = []
        seen = set()
        for si, sj in route:
            key = (si, sj)
            if key in seen:
                continue
            seen.add(key)
            gx, gy = sj * 2 + 1, si * 2 + 1
            self.waypoints.append(self.imap.to_world(gx, gy))

        self.current_waypoint_idx = 0
        return len(self.waypoints) > 1

    def _stc(self, si, sj, visit, route):
        order = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        found = False
        route.append((si, sj))

        for dsi, dsj in order:
            ni, nj = si + dsi, sj + dsj
            if self.imap._is_valid_sc(ni, nj) and visit[ni][nj] == 0:
                self.edge_list.append(((si, sj), (ni, nj)))
                found = True
                visit[ni][nj] += 1
                self._stc(ni, nj, visit, route)

        if not found:
            has_unvisited = False
            for node in reversed(route):
                if visit[node[0]][node[1]] == 2:
                    continue
                visit[node[0]][node[1]] += 1
                route.append(node)
                for dsi, dsj in order:
                    ni, nj = node[0] + dsi, node[1] + dsj
                    if self.imap._is_valid_sc(ni, nj) and visit[ni][nj] == 0:
                        has_unvisited = True
                        break
                if has_unvisited:
                    break

    def get_next_waypoint(self, current_pos_m):
        if not self.waypoints or self.current_waypoint_idx >= len(self.waypoints):
            return None

        target = self.waypoints[self.current_waypoint_idx]
        dist = np.linalg.norm(target - current_pos_m)

        # Skip if close enough
        if dist < SUPER_CELL_SIZE * 0.5:
            self.current_waypoint_idx += 1
            self._steps_on_waypoint = 0
            self._wp_start_pos = None
            if self.current_waypoint_idx >= len(self.waypoints):
                return None
            target = self.waypoints[self.current_waypoint_idx]

        # Track progress
        if self._wp_start_pos is None:
            self._wp_start_pos = current_pos_m.copy()
            self._steps_on_waypoint = 0
        self._steps_on_waypoint += 1

        # If stuck for too many steps, skip waypoint
        if self._steps_on_waypoint > 100:
            self.current_waypoint_idx += 1
            self._steps_on_waypoint = 0
            self._wp_start_pos = None
            if self.current_waypoint_idx >= len(self.waypoints):
                return None
            target = self.waypoints[self.current_waypoint_idx]

        return target

    def needs_replan(self):
        return not self.waypoints or self.current_waypoint_idx >= len(self.waypoints)


# --------------------------------------------------------------------------- #
#  Robot Controller
# --------------------------------------------------------------------------- #

class RobotController:
    def __init__(self):
        self.kp_h = 2.0
        self.kd_h = 0.15
        self.kp_v = 0.8
        self.prev_err = 0.0

    def compute(self, pos_m, heading, target_m):
        dx = target_m[0] - pos_m[0]
        dy = target_m[1] - pos_m[1]
        dist = math.hypot(dx, dy)
        desired = math.atan2(dy, dx)
        err = (desired - heading + math.pi) % (2 * math.pi) - math.pi

        d_err = err - self.prev_err
        ang = max(-1.0, min(1.0, self.kp_h * err + self.kd_h * d_err))
        self.prev_err = err

        turn_f = max(0.0, 1.0 - abs(err) / (math.pi / 2))
        dist_f = min(1.0, dist / SUPER_CELL_SIZE)
        lin = self.kp_v * turn_f * dist_f

        return lin, ang


# --------------------------------------------------------------------------- #
#  Sensors
# --------------------------------------------------------------------------- #

def local_to_global(lx, ly, pos, h):
    return (pos[0] + lx * math.cos(h) - ly * math.sin(h),
            pos[1] + lx * math.sin(h) + ly * math.cos(h))


def cast_ray(origin, angle, true_grid, gsp, ppm, offset):
    step = METERS_PER_PIXEL * 0.5
    dx, dy = math.cos(angle), math.sin(angle)
    for i in range(1, int(RAY_MAX_DIST / step) + 1):
        px = origin[0] + dx * step * i
        py = origin[1] + dy * step * i
        ix = int((px - offset[0]) * ppm)
        iy = int((py - offset[1]) * ppm)
        if ix < 0 or ix >= gsp or iy < 0 or iy >= gsp:
            return RAY_MAX_DIST, True
        if true_grid[iy, ix] == 0:
            return min(math.hypot(px - origin[0], py - origin[1]), RAY_MAX_DIST), True
    return RAY_MAX_DIST, False


def compute_sensors(pos, heading, true_grid, gsp, ppm, offset):
    a = ROBOT_SIDE
    ol = [(a/2, a/2), (a/2, 0), (a/2, -a/2), (0, a/2), (0, -a/2), (-a/2, 0)]
    ao = [math.pi/4, 0.0, -math.pi/4, math.pi/2, -math.pi/2, math.pi]
    dists, hits = [], []
    for (lx, ly), af in zip(ol, ao):
        ox, oy = local_to_global(lx, ly, pos, heading)
        d, h = cast_ray(np.array([ox, oy]), heading + af, true_grid, gsp, ppm, offset)
        dists.append(d)
        hits.append((ox, oy, heading + af, d, h))
    return dists, hits


# --------------------------------------------------------------------------- #
#  Simulation
# --------------------------------------------------------------------------- #

class SSTCSimulation:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.ppm = 1.0 / METERS_PER_PIXEL

        self.field = generate_random_field(self.rng)
        minx, miny, maxx, maxy = self.field.bounds
        pad = 5.0
        self.offset = np.array([minx - pad, miny - pad])

        self.true_grid, self.gsp = rasterize_field(self.field, self.ppm, self.offset)
        self.imap = InternalMap(self.true_grid, BASE_CELL_SIZE, self.offset, self.ppm)

        self.pos_m, self.heading = self._get_spawn()
        self.ctrl = RobotController()

        self.step_count = 0
        self.non_new = 0
        self.collisions = 0
        self.last_dists = []
        self.last_hits = []
        self.mode = "explore"
        self.explore_target = None
        self._sstc_coverage_start = 0
        self._steps_on_waypoint = 0
        self._wp_start_pos = None
        self._collision_window = []
        self._wall_follow_timer = 0
        self._wall_follow_steps = 0
        self._sstc_replans = 0

        self.sstc = SSTCPlanner(self.imap)

        self.imap.mark_coverage(self.pos_m, self.heading)
        self._update_sensors()

    def _get_spawn(self):
        for _ in range(100):
            boundary = list(self.field.exterior.coords)
            ne = len(boundary) - 1
            ei = self.rng.integers(0, ne)
            x1, y1 = boundary[ei]
            x2, y2 = boundary[(ei + 1) % ne]
            t = self.rng.uniform(0.25, 0.75)
            px, py = x1 + t * (x2 - x1), y1 + t * (y2 - y1)
            dx, dy = x2 - x1, y2 - y1
            el = math.hypot(dx, dy)
            nx, ny = -dy / el, dx / el
            inward = self.field.representative_point()
            if nx * (inward.x - px) + ny * (inward.y - py) < 0:
                nx, ny = -nx, -ny
            pos = np.array([px + nx * ROBOT_SIDE, py + ny * ROBOT_SIDE])
            if self._valid(pos):
                return pos, math.atan2(dy, dx)
        raise RuntimeError("No spawn")

    def _valid(self, pos):
        gx = int((pos[0] - self.offset[0]) * self.ppm)
        gy = int((pos[1] - self.offset[1]) * self.ppm)
        r = int(ROBOT_RADIUS * self.ppm)
        if gx - r < 0 or gx + r >= self.gsp or gy - r < 0 or gy + r >= self.gsp:
            return False
        return self.true_grid[gy, gx] == 1

    def _update_sensors(self):
        d, h = compute_sensors(self.pos_m, self.heading, self.true_grid, self.gsp, self.ppm, self.offset)
        a = ROBOT_SIDE
        ol = [(a/2, a/2), (a/2, 0), (a/2, -a/2), (0, a/2), (0, -a/2), (-a/2, 0)]
        ao = [math.pi/4, 0.0, -math.pi/4, math.pi/2, -math.pi/2, math.pi]
        for (lx, ly), af, dist in zip(ol, ao, d):
            ox, oy = local_to_global(lx, ly, self.pos_m, self.heading)
            self.imap.mark_ray(np.array([ox, oy]), self.heading + af, dist, dist < RAY_MAX_DIST)
        self.last_dists = d
        self.last_hits = h

    def _collides(self, new_pos):
        gx = int((new_pos[0] - self.offset[0]) * self.ppm)
        gy = int((new_pos[1] - self.offset[1]) * self.ppm)
        r = int(ROBOT_RADIUS * self.ppm)
        x1, x2 = max(0, gx - r), min(self.gsp, gx + r + 1)
        y1, y2 = max(0, gy - r), min(self.gsp, gy + r + 1)
        if x1 >= x2 or y1 >= y2:
            return True
        return self.true_grid[y1:y2, x1:x2].sum() < (y2 - y1) * (x2 - x1)

    def _try_sstc(self):
        n_sc = self.imap.count_valid_super_cells(self.sstc.sstc_h, self.sstc.sstc_w)
        if n_sc < MIN_SUPER_CELLS_FOR_SSTC:
            return False
        start = self.sstc.find_start(self.pos_m)
        if start is not None and self.sstc.plan(start[0], start[1]):
            self.mode = "sstc"
            self._sstc_coverage_start = self.imap.covered_count
            self._sstc_replans = 0
            return True
        return False

    def step(self):
        self.step_count += 1
        target = None

        if self.mode == "sstc":
            target = self.sstc.get_next_waypoint(self.pos_m)
            if target is None:
                new_covered = self.imap.covered_count - self._sstc_coverage_start
                if new_covered < 3:
                    self.mode = "explore"
                    self.explore_target = self.imap.find_frontier(self.pos_m)
                else:
                    if not self._try_sstc():
                        self.mode = "explore"
                        self.explore_target = self.imap.find_frontier(self.pos_m)

        if self.mode == "explore":
            # Track steps in explore mode
            if not hasattr(self, '_explore_steps'):
                self._explore_steps = 0
            self._explore_steps += 1

            if self.explore_target is None:
                self.explore_target = self.imap.find_frontier(self.pos_m)

            if self.explore_target is not None:
                dist_to_target = np.linalg.norm(self.explore_target - self.pos_m)
                if dist_to_target < SUPER_CELL_SIZE:
                    self.explore_target = None
                    self._explore_steps = 0
                    if self._try_sstc():
                        target = self.sstc.get_next_waypoint(self.pos_m)
                    else:
                        self.explore_target = self.imap.find_frontier(self.pos_m)
                # If stuck exploring for too long, try wall follow
                elif self._explore_steps > 300:
                    self.mode = "wall_follow"
                    self._wall_follow_steps = 0
                    self._wall_follow_timer = 300
                    self._explore_steps = 0
            else:
                if self._try_sstc():
                    target = self.sstc.get_next_waypoint(self.pos_m)
                else:
                    # No frontiers, no SSTC - wall follow
                    self.mode = "wall_follow"
                    self._wall_follow_steps = 0
                    self._wall_follow_timer = 300
                    self._explore_steps = 0

            if target is None:
                target = self.explore_target

        elif self.mode == "wall_follow":
            # Simple right-hand wall following
            self._wall_follow_steps += 1
            self._wall_follow_timer -= 1

            # Check right sensor
            right_angle = self.heading - math.pi / 2
            right_clear = True
            for d in self.last_dists:
                pass

            # Use the right sensor (index 4 is right-facing)
            right_dist = self.last_dists[4] if len(self.last_dists) > 4 else RAY_MAX_DIST
            front_dist = self.last_dists[1] if len(self.last_dists) > 1 else RAY_MAX_DIST

            if front_dist < ROBOT_SIDE * 1.5:
                # Obstacle ahead, turn left
                ang_cmd = 0.8
                lin_cmd = 0.2
            elif right_dist > ROBOT_SIDE * 2.0:
                # Wall too far, turn right
                ang_cmd = -0.5
                lin_cmd = 0.5
            elif right_dist < ROBOT_SIDE * 0.8:
                # Too close to wall, turn left
                ang_cmd = 0.4
                lin_cmd = 0.5
            else:
                # Good distance, go straight
                ang_cmd = 0.0
                lin_cmd = 0.8

            lin = max(0.0, min(1.0, lin_cmd)) * ROBOT_SPEED_V
            ang = max(-1.0, min(1.0, ang_cmd)) * ROBOT_SPEED_W

            new_h = (self.heading + ang * DT) % (2 * math.pi)
            mid_h = self.heading + ang * DT / 2
            new_pos = self.pos_m + np.array([lin * DT * math.cos(mid_h), lin * DT * math.sin(mid_h)])

            if self._collides(new_pos):
                self.collisions += 1
                self.heading = (self.heading + math.pi / 2) % (2 * math.pi)
                return True

            old_cov = self.imap.get_coverage_percent()
            self.pos_m = new_pos
            self.heading = new_h
            self._update_sensors()
            self.imap.mark_coverage(self.pos_m, self.heading)
            new_cov = self.imap.get_coverage_percent()

            # Switch back to explore if we found new area or timer expired
            if new_cov > old_cov or self._wall_follow_timer <= 0:
                self.mode = "explore"
                self.explore_target = self.imap.find_frontier(self.pos_m)
                if self.explore_target is None and self._try_sstc():
                    target = self.sstc.get_next_waypoint(self.pos_m)
                elif self.explore_target is None:
                    # No frontiers and no SSTC - try wall follow again
                    self.mode = "wall_follow"
                    self._wall_follow_timer = 400

            self.non_new = 0 if new_cov > old_cov else self.non_new + 1
            return True

        if target is None:
            return False

        lin, ang = self.ctrl.compute(self.pos_m, self.heading, target)
        lin = max(0.0, min(1.0, lin)) * ROBOT_SPEED_V
        ang = max(-1.0, min(1.0, ang)) * ROBOT_SPEED_W

        new_h = (self.heading + ang * DT) % (2 * math.pi)
        mid_h = self.heading + ang * DT / 2
        new_pos = self.pos_m + np.array([lin * DT * math.cos(mid_h), lin * DT * math.sin(mid_h)])

        if self._collides(new_pos):
            self.collisions += 1
            self._collision_window.append(self.step_count)
            self.heading = (self.heading + math.pi / 3) % (2 * math.pi)

            recent = [c for c in self._collision_window if self.step_count - c < 50]
            self._collision_window = recent
            if len(recent) >= 5:
                # Switch to wall-following mode for longer
                self.mode = "wall_follow"
                self._wall_follow_steps = 0
                self._wall_follow_timer = 400
                self.explore_target = None
                self.sstc.current_waypoint_idx = len(self.sstc.waypoints)
                self._collision_window.clear()
            return True

        old_cov = self.imap.get_coverage_percent()
        self.pos_m = new_pos
        self.heading = new_h
        self._update_sensors()
        self.imap.mark_coverage(self.pos_m, self.heading)
        new_cov = self.imap.get_coverage_percent()

        self.non_new = 0 if new_cov > old_cov else self.non_new + 1
        return True

    def is_done(self):
        return (self.imap.get_coverage_percent() >= COVERAGE_GOAL or
                self.step_count >= MAX_STEPS or
                self.non_new >= NO_PROGRESS_LIMIT)

    def get_stats(self):
        return {
            "step": self.step_count,
            "coverage": self.imap.get_coverage_percent(),
            "collisions": self.collisions,
            "mode": self.mode,
            "wp_idx": self.sstc.current_waypoint_idx,
            "wp_total": len(self.sstc.waypoints),
            "free": self.imap.free_count,
            "covered": self.imap.covered_count,
        }


# --------------------------------------------------------------------------- #
#  Visualization
# --------------------------------------------------------------------------- #

class Visualizer:
    def __init__(self, sim):
        self.sim = sim
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption("SSTC Continuous Coverage")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 18)

        minx, miny, maxx, maxy = sim.field.bounds
        pad = 5.0
        w, h = (maxx - minx) + 2 * pad, (maxy - miny) + 2 * pad
        self.scale = WINDOW_SIZE / max(w, h)
        self.offset = sim.offset
        self.grid_surf = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE), pygame.SRCALPHA)
        self.paint_surf = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE), pygame.SRCALPHA)
        self._dirty = True

    def _tp(self, x, y):
        return (int((x - self.offset[0]) * self.scale),
                int(WINDOW_SIZE - (y - self.offset[1]) * self.scale))

    def _update_grid(self):
        self.grid_surf.fill((0, 0, 0, 0))
        res = 1.0 / self.sim.ppm
        cpx = max(1, int(BASE_CELL_SIZE * self.sim.ppm * self.scale))
        grid = self.sim.imap.grid
        for gy in range(self.sim.imap.grid_h):
            for gx in range(self.sim.imap.grid_w):
                s = grid[gy, gx]
                if s in (InternalMap.COVERED, InternalMap.FREE, InternalMap.WALL):
                    px = gx * res + self.offset[0]
                    py = (gy + 1) * res + self.offset[1]
                    pg = self._tp(px, py)
                    c = {InternalMap.COVERED: (*COLOR_COVERED, 180),
                         InternalMap.FREE: (*COLOR_FREE, 80),
                         InternalMap.WALL: (*COLOR_WALL, 180)}[s]
                    pygame.draw.rect(self.grid_surf, c, (pg[0], pg[1], cpx, cpx))

    def render(self):
        self.screen.fill(COLOR_BG)
        ext = [self._tp(x, y) for x, y in self.sim.field.exterior.coords]
        pygame.draw.polygon(self.screen, COLOR_FIELD, ext)

        # Paint the robot's footprint on the persistent paint surface
        half = ROBOT_RADIUS
        ch, sh = math.cos(self.sim.heading), math.sin(self.sim.heading)
        paint_corners = []
        for lx, ly in [(-half, -half), (half, -half), (half, half), (-half, half)]:
            paint_corners.append(self._tp(
                self.sim.pos_m[0] + lx * ch - ly * sh,
                self.sim.pos_m[1] + lx * sh + ly * ch
            ))
        # Semi-transparent green paint that accumulates
        paint_mask = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE), pygame.SRCALPHA)
        pygame.draw.polygon(paint_mask, (100, 220, 100, 40), paint_corners)
        self.paint_surf.blit(paint_mask, (0, 0))

        self.screen.blit(self.paint_surf, (0, 0))

        if self.sim.step_count % 10 == 0 or self._dirty:
            self._update_grid()
            self._dirty = False
        self.screen.blit(self.grid_surf, (0, 0))

        pygame.draw.polygon(self.screen, COLOR_BOUNDARY, ext, 3)
        for interior in self.sim.field.interiors:
            ip = [self._tp(x, y) for x, y in interior.coords]
            pygame.draw.polygon(self.screen, (255, 255, 255), ip)
            pygame.draw.polygon(self.screen, (255, 0, 0), ip, 2)

        for (si1, sj1), (si2, sj2) in self.sim.sstc.edge_list:
            w1 = self.sim.imap.to_world(sj1 * 2 + 1, si1 * 2 + 1)
            w2 = self.sim.imap.to_world(sj2 * 2 + 1, si2 * 2 + 1)
            pygame.draw.line(self.screen, COLOR_SPANNING,
                             self._tp(w1[0], w1[1]), self._tp(w2[0], w2[1]), 2)

        rem = self.sim.sstc.waypoints[self.sim.sstc.current_waypoint_idx:]
        if len(rem) > 1:
            pygame.draw.lines(self.screen, COLOR_PATH, False,
                              [self._tp(w[0], w[1]) for w in rem], 2)

        half = ROBOT_RADIUS
        ch, sh = math.cos(self.sim.heading), math.sin(self.sim.heading)
        corners = []
        for lx, ly in [(-half, -half), (half, -half), (half, half), (-half, half)]:
            corners.append(self._tp(self.sim.pos_m[0] + lx * ch - ly * sh,
                                     self.sim.pos_m[1] + lx * sh + ly * ch))
        pygame.draw.polygon(self.screen, COLOR_ROBOT, corners)

        hx = self.sim.pos_m[0] + ROBOT_RADIUS * math.cos(self.sim.heading)
        hy = self.sim.pos_m[1] + ROBOT_RADIUS * math.sin(self.sim.heading)
        pygame.draw.line(self.screen, (0, 255, 0),
                         self._tp(*self.sim.pos_m), self._tp(hx, hy), 3)

        a = ROBOT_SIDE
        ct, st = math.cos(self.sim.heading), math.sin(self.sim.heading)
        def lpt(lx, ly):
            return (self.sim.pos_m[0] + lx * ct - ly * st,
                    self.sim.pos_m[1] + lx * st + ly * ct)
        ol = [lpt(a/2, a/2), lpt(a/2, 0), lpt(a/2, -a/2),
              lpt(0, a/2), lpt(0, -a/2), lpt(-a/2, 0)]
        af = [math.pi/4, 0.0, -math.pi/4, math.pi/2, -math.pi/2, math.pi]
        for o, afi, d in zip(ol, af, self.sim.last_dists):
            ang = self.sim.heading + afi
            end = (o[0] + math.cos(ang) * d, o[1] + math.sin(ang) * d)
            c = COLOR_RAY_HIT if d < RAY_MAX_DIST else COLOR_RAY_CLEAR
            pygame.draw.line(self.screen, c, self._tp(*o), self._tp(*end), 1)

        s = self.sim.get_stats()
        for i, line in enumerate([
            f"Step: {s['step']}", f"Coverage: {s['coverage']:.1%}",
            f"Mode: {s['mode']}", f"Collisions: {s['collisions']}",
            f"WP: {s['wp_idx']}/{s['wp_total']}",
            f"Free: {s['free']}  Covered: {s['covered']}",
        ]):
            self.screen.blit(self.font.render(line, True, COLOR_TEXT), (10, 10 + i * 24))

        pygame.display.flip()

    def close(self):
        pygame.quit()


def main():
    print("SSTC Continuous Coverage")
    sim = SSTCSimulation(seed=42)
    viz = Visualizer(sim)
    print(f"Grid: {sim.imap.grid_h}x{sim.imap.grid_w}")

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_r:
                    sim = SSTCSimulation(seed=None)
                    viz = Visualizer(sim)

        if not sim.is_done():
            sim.step()
        viz.render()
        viz.clock.tick(FPS)

        if sim.step_count % 100 == 0:
            s = sim.get_stats()
            print(f"Step {s['step']}: {s['coverage']:.1%} mode={s['mode']} coll={s['collisions']}")

    s = sim.get_stats()
    print(f"\nDone: step={s['step']}, coverage={s['coverage']:.1%}, collisions={s['collisions']}")
    viz.close()


if __name__ == "__main__":
    main()
