"""
ε⋆ (epsilon-star) Online Coverage Path Planning Algorithm

Implementation of the algorithm from:
  "ε⋆: An Online Coverage Path Planning Algorithm"
  Junnan Song, Shalabh Gupta — University of Connecticut

Adapted for the v2 robot environment (src/v2/robot_env.py) with:
  - 6-ray sensors, 1m range
  - Noisy localization (σ_pos=0.01m, σ_heading=0.05rad)
  - Continuous differential-drive control
  - Bounding-box tiling over random polygonal fields
"""

from __future__ import annotations

import enum
import logging
import math

import numpy as np

logger = logging.getLogger("EStar")
logger.setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EPSILON_M = 1.0
SAFETY_BUFFER_CELLS = 1
LOCAL_NEIGHBORHOOD_L0 = 7
LOCAL_NEIGHBORHOOD_HL = 3
LOOKAHEAD_M = 0.8
KP_HEADING = 1.5
KP_THROTTLE = 0.6
SAFETY_THRESHOLD_M = 0.45
FLOOD_FILL_INTERVAL = 10


class CellState(enum.IntEnum):
    O = 0
    F = 1
    E = 2
    U = 3


# ---------------------------------------------------------------------------
# Bresenham
# ---------------------------------------------------------------------------
def _bresenham(x0: int, y0: int, x1: int, y1: int):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    cx, cy = x0, y0
    while True:
        yield cx, cy
        if cx == x1 and cy == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            cx += sx
        if e2 < dx:
            err += dx
            cy += sy


# ---------------------------------------------------------------------------
# Epsilon Tiling
# ---------------------------------------------------------------------------
class EpsilonTiling:
    def __init__(
        self,
        field_grid: np.ndarray,
        render_offset: np.ndarray,
        pixels_per_meter: float,
        epsilon_m: float = EPSILON_M,
    ):
        self.field_grid = field_grid
        self.render_offset = render_offset
        self.pixels_per_meter = pixels_per_meter
        self.epsilon_m = epsilon_m
        self.pixels_per_cell = max(1, int(round(epsilon_m * pixels_per_meter)))

        h_grid, w_grid = field_grid.shape
        self.cols = max(1, w_grid // self.pixels_per_cell)
        self.rows = max(1, h_grid // self.pixels_per_cell)
        self.num_cells = self.rows * self.cols
        self.cell_size_m = self.pixels_per_cell / pixels_per_meter

        self.cell_states = np.full(self.num_cells, CellState.U, dtype=np.int8)
        self.cell_centroids_m = self._compute_centroids()
        self._mark_initial_obstacles()
        self.levels = self._build_hierarchy()
        self.exogenous = self._build_exogenous_field()
        self.forbidden = np.zeros(self.num_cells, dtype=bool)

        self._obstacle_2d_dirty = True
        self._obstacle_2d_cache: np.ndarray | None = None

        logger.info(
            "Tiling: %dx%d = %d cells, epsilon=%.2fm, %d levels",
            self.cols, self.rows, self.num_cells, self.epsilon_m, len(self.levels),
        )

    # -- coordinate helpers (vectorized) -----------------------------------

    def _compute_centroids(self) -> np.ndarray:
        rs = np.arange(self.rows)
        cs = np.arange(self.cols)
        rr, cc = np.meshgrid(rs, cs, indexing="ij")
        rr = rr.ravel()
        cc = cc.ravel()
        px = (cc + 0.5) * self.pixels_per_cell
        py = (rr + 0.5) * self.pixels_per_cell
        cx_m = px / self.pixels_per_meter + self.render_offset[0]
        cy_m = py / self.pixels_per_meter + self.render_offset[1]
        return np.column_stack([cx_m, cy_m])

    def world_to_cell(self, pos_m) -> int:
        px = (pos_m[0] - self.render_offset[0]) * self.pixels_per_meter
        py = (pos_m[1] - self.render_offset[1]) * self.pixels_per_meter
        c = int(px / self.pixels_per_cell)
        r = int(py / self.pixels_per_cell)
        c = max(0, min(c, self.cols - 1))
        r = max(0, min(r, self.rows - 1))
        return r * self.cols + c

    def pixel_to_cell(self, px: float, py: float) -> int:
        c = int(px / self.pixels_per_cell)
        r = int(py / self.pixels_per_cell)
        c = max(0, min(c, self.cols - 1))
        r = max(0, min(r, self.rows - 1))
        return r * self.cols + c

    def cell_to_world(self, idx: int) -> np.ndarray:
        return self.cell_centroids_m[idx].copy()

    def cell_rc(self, idx: int) -> tuple[int, int]:
        return divmod(idx, self.cols)

    def rc_to_cell(self, r: int, c: int) -> int | None:
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return r * self.cols + c
        return None

    # -- initial obstacle marking ------------------------------------------

    def _mark_initial_obstacles(self):
        ppc = self.pixels_per_cell
        h_grid, w_grid = self.field_grid.shape
        for idx in range(self.num_cells):
            r, c = divmod(idx, self.cols)
            y1 = r * ppc
            y2 = min((r + 1) * ppc, h_grid)
            x1 = c * ppc
            x2 = min((c + 1) * ppc, w_grid)
            if y1 >= y2 or x1 >= x2:
                self.cell_states[idx] = CellState.O
                continue
            if self.field_grid[y1:y2, x1:x2].mean() < 0.5:
                self.cell_states[idx] = CellState.O

    # -- hierarchy ---------------------------------------------------------

    def _build_hierarchy(self) -> list[list[tuple[int, int, int, int]]]:
        levels: list[list[tuple[int, int, int, int]]] = []

        levels.append([
            (r, c, r + 1, c + 1)
            for r in range(self.rows)
            for c in range(self.cols)
        ])

        prev = levels[0]
        group_size = 2
        while len(prev) > 1:
            seen = {}
            nxt = []
            for r0, c0, r_end, c_end in prev:
                rr0 = (r0 // group_size) * group_size
                cc0 = (c0 // group_size) * group_size
                key = (rr0, cc0)
                if key in seen:
                    i = seen[key]
                    old_r0, old_c0, old_re, old_ce = nxt[i]
                    nxt[i] = (old_r0, old_c0, max(old_re, r_end), max(old_ce, c_end))
                else:
                    seen[key] = len(nxt)
                    nxt.append((r0, c0, r_end, c_end))
            if len(nxt) == len(prev):
                break
            levels.append(nxt)
            prev = nxt
            group_size *= 2
        return levels

    def _build_exogenous_field(self) -> np.ndarray:
        B = np.zeros(self.num_cells, dtype=np.float64)
        for idx in range(self.num_cells):
            _, c = divmod(idx, self.cols)
            B[idx] = self.cols - c
        return B

    # -- cell operations ---------------------------------------------------

    def mark_obstacle(self, idx: int):
        self.cell_states[idx] = CellState.O
        self._obstacle_2d_dirty = True

    def mark_forbidden(self, idx: int):
        if self.cell_states[idx] == CellState.U:
            self.cell_states[idx] = CellState.F
            self.forbidden[idx] = True
            self._obstacle_2d_dirty = True

    def mark_explored(self, idx: int):
        if self.cell_states[idx] == CellState.U:
            self.cell_states[idx] = CellState.E

    def get_obstacle_2d(self) -> np.ndarray:
        if self._obstacle_2d_dirty or self._obstacle_2d_cache is None:
            s = self.cell_states
            self._obstacle_2d_cache = (s == CellState.O) | (s == CellState.F)
            self._obstacle_2d_cache = self._obstacle_2d_cache.reshape(self.rows, self.cols)
            self._obstacle_2d_dirty = False
        return self._obstacle_2d_cache

    def dilate_obstacles(self, radius_cells: int):
        to_mark_f = []
        for idx in range(self.num_cells):
            if self.cell_states[idx] == CellState.O:
                r0, c0 = self.cell_rc(idx)
                for dr in range(-radius_cells, radius_cells + 1):
                    for dc in range(-radius_cells, radius_cells + 1):
                        if dr * dr + dc * dc > radius_cells * radius_cells:
                            continue
                        ni = self.rc_to_cell(r0 + dr, c0 + dc)
                        if ni is not None and self.cell_states[ni] == CellState.U:
                            to_mark_f.append(ni)
        for ni in to_mark_f:
            self.cell_states[ni] = CellState.F
            self.forbidden[ni] = True
        self._obstacle_2d_dirty = True

    def flood_fill_obstacles(self):
        visited = np.zeros(self.num_cells, dtype=bool)
        for idx in range(self.num_cells):
            if visited[idx] or self.cell_states[idx] != CellState.U:
                continue
            region = []
            stack = [idx]
            touches_boundary = False
            while stack:
                ci = stack.pop()
                if visited[ci]:
                    continue
                visited[ci] = True
                region.append(ci)
                r, c = self.cell_rc(ci)
                for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                    ni = self.rc_to_cell(r + dr, c + dc)
                    if ni is None:
                        touches_boundary = True
                    elif not visited[ni] and self.cell_states[ni] != CellState.O:
                        stack.append(ni)
            if not touches_boundary and region:
                for ci in region:
                    self.cell_states[ci] = CellState.O
                    self.exogenous[ci] = -1
        self._obstacle_2d_dirty = True

    def get_neighborhood(self, cell_idx: int, level: int, size: int) -> list[int]:
        r, c = self.cell_rc(cell_idx)
        half = size // 2
        result = []
        r_min = max(0, r - half)
        r_max = min(self.rows - 1, r + half)
        c_min = max(0, c - half)
        c_max = min(self.cols - 1, c + half)
        for rr in range(r_min, r_max + 1):
            base = rr * self.cols
            for cc in range(c_min, c_max + 1):
                result.append(base + cc)
        return result

    def find_coarse_containing(self, cell_idx: int, level: int) -> int | None:
        r, c = self.cell_rc(cell_idx)
        for i, (r0, c0, rows, cols) in enumerate(self.levels[level]):
            if r0 <= r < r0 + rows and c0 <= c < c0 + cols:
                return i
        return None

    def get_cells_in_coarse(self, coarse_idx: int, level: int) -> list[int]:
        r0, c0, r_end, c_end = self.levels[level][coarse_idx]
        r_end = min(r_end, self.rows)
        c_end = min(c_end, self.cols)
        cells = []
        for r in range(r0, r_end):
            base = r * self.cols
            for c in range(c0, c_end):
                cells.append(base + c)
        return cells


# ---------------------------------------------------------------------------
# MAPS
# ---------------------------------------------------------------------------
class MAPS:
    def __init__(self, tiling: EpsilonTiling):
        self.tiling = tiling
        self.num_levels = len(tiling.levels)
        self.potentials: list[np.ndarray] = []
        self.p_u: list[np.ndarray] = []

        for level in range(self.num_levels):
            n_coarse = len(tiling.levels[level])
            self.potentials.append(np.zeros(n_coarse, dtype=np.float64))
            self.p_u.append(np.ones(n_coarse, dtype=np.float64))

        self._init_level_0()

    def _init_level_0(self):
        t = self.tiling
        states = t.cell_states
        B = t.exogenous

        mask_obs_forb = (states == CellState.O) | (states == CellState.F)
        mask_explored = states == CellState.E
        mask_unexplored = states == CellState.U

        self.potentials[0][mask_obs_forb] = -1.0
        self.potentials[0][mask_explored] = 0.0
        self.potentials[0][mask_unexplored] = B[mask_unexplored]

        self._recompute_all_levels()

    def _recompute_all_levels(self):
        t = self.tiling
        states = t.cell_states
        for level in range(1, self.num_levels):
            n_coarse = len(t.levels[level])
            for ci in range(n_coarse):
                cells = t.get_cells_in_coarse(ci, level)
                n_total = len(cells)
                n_u = sum(1 for c in cells if states[c] == CellState.U)
                p_u = n_u / n_total if n_total > 0 else 0.0
                self.p_u[level][ci] = p_u
                mean_B = np.mean([t.exogenous[c] for c in cells])
                self.potentials[level][ci] = p_u * mean_B

    def update_obstacle(self, cell_idx: int):
        self.potentials[0][cell_idx] = -1.0
        self._propagate_up(cell_idx)

    def update_explored(self, cell_idx: int):
        self.potentials[0][cell_idx] = 0.0
        self._propagate_up(cell_idx)

    def _propagate_up(self, cell_idx: int):
        t = self.tiling
        for level in range(1, self.num_levels):
            coarse_idx = t.find_coarse_containing(cell_idx, level)
            if coarse_idx is None:
                continue
            cells = t.get_cells_in_coarse(coarse_idx, level)
            n_total = len(cells)
            n_u = sum(1 for ci in cells if t.cell_states[ci] == CellState.U)
            p_u = n_u / n_total if n_total > 0 else 0.0
            self.p_u[level][coarse_idx] = p_u
            mean_B = np.mean([t.exogenous[ci] for ci in cells])
            self.potentials[level][coarse_idx] = p_u * mean_B


# ---------------------------------------------------------------------------
# ETM
# ---------------------------------------------------------------------------
class ETMState(enum.Enum):
    ST = "start"
    CP0 = "compute_level0"
    CPL = "compute_higher"
    WT = "wait"
    FN = "finish"


class ETM:
    def __init__(self, tiling: EpsilonTiling, maps: MAPS):
        self.tiling = tiling
        self.maps = maps
        self.state = ETMState.ST
        self.current_cell = -1
        self.previous_waypoint = -1
        self.cycle_count = 0
        self.escape_level = 0

    def initialize(self, start_cell: int):
        self.current_cell = start_cell
        self.previous_waypoint = -1
        self.state = ETMState.CP0

    def update_from_sensors(
        self,
        robot_pos_m: np.ndarray,
        hit_points_world: list[tuple[float, float] | None],
    ):
        t = self.tiling
        new_obstacles = False

        for hp in hit_points_world:
            if hp is None:
                continue
            cell_idx = t.world_to_cell(np.array(hp))
            if t.cell_states[cell_idx] == CellState.U:
                t.mark_obstacle(cell_idx)
                self.maps.update_obstacle(cell_idx)
                new_obstacles = True

                r, c = t.cell_rc(cell_idx)
                for dr in range(-SAFETY_BUFFER_CELLS, SAFETY_BUFFER_CELLS + 1):
                    for dc in range(-SAFETY_BUFFER_CELLS, SAFETY_BUFFER_CELLS + 1):
                        if dr * dr + dc * dc > (SAFETY_BUFFER_CELLS + 0.5) ** 2:
                            continue
                        ni = t.rc_to_cell(r + dr, c + dc)
                        if ni is not None and t.cell_states[ni] == CellState.U:
                            t.mark_forbidden(ni)
                            self.maps.potentials[0][ni] = -1.0
                            self.maps._propagate_up(ni)

        self.current_cell = t.world_to_cell(robot_pos_m)

        self.cycle_count += 1
        if self.cycle_count % FLOOD_FILL_INTERVAL == 0 and new_obstacles:
            t.flood_fill_obstacles()
            self.maps._recompute_all_levels()

    def mark_current_explored(self):
        t = self.tiling
        if self.current_cell >= 0 and t.cell_states[self.current_cell] == CellState.U:
            t.mark_explored(self.current_cell)
            self.maps.update_explored(self.current_cell)

    def compute_waypoint(self) -> tuple[int, str]:
        if self.state == ETMState.FN:
            return self.current_cell, "sp"
        if self.state == ETMState.CP0:
            return self._compute_cp0()
        if self.state == ETMState.CPL:
            return self._compute_cpl()
        return self.current_cell, "id"

    def _compute_cp0(self) -> tuple[int, str]:
        t = self.tiling
        maps = self.maps
        lam = self.current_cell
        if lam < 0:
            return self.current_cell, "id"

        obstacle_2d = t.get_obstacle_2d()
        rows, cols_s = obstacle_2d.shape
        neighborhood = t.get_neighborhood(lam, 0, LOCAL_NEIGHBORHOOD_L0)

        lam_r, lam_c = t.cell_rc(lam)

        dr_set = []
        for n in neighborhood:
            if t.cell_states[n] == CellState.U and maps.potentials[0][n] > 0:
                cr, cc = t.cell_rc(n)
                blocked = False
                for bx, by in _bresenham(lam_c, lam_r, cc, cr):
                    if 0 <= by < rows and 0 <= bx < cols_s and obstacle_2d[by, bx]:
                        blocked = True
                        break
                if not blocked:
                    dr_set.append(n)

        lam_u = t.cell_states[lam] == CellState.U

        if lam_u:
            up_idx = t.rc_to_cell(lam_r - 1, lam_c)
            down_idx = t.rc_to_cell(lam_r + 1, lam_c)

            up_eligible = (
                up_idx is not None
                and t.cell_states[up_idx] == CellState.U
                and maps.potentials[0][up_idx] > 0
            )
            down_eligible = (
                down_idx is not None
                and t.cell_states[down_idx] == CellState.U
                and maps.potentials[0][down_idx] > 0
            )

            if up_eligible and down_eligible:
                best_wp = -1
                best_pot = -1.0
                for n in neighborhood:
                    ns = t.cell_states[n]
                    if ns in (CellState.F, CellState.E):
                        nr, nc = t.cell_rc(n)
                        if abs(nc - lam_c) <= 1 and abs(nr - lam_r) <= 1:
                            p = maps.potentials[0][n]
                            if p > best_pot:
                                best_pot = p
                                best_wp = n
                if best_wp < 0:
                    best_wp = up_idx if up_idx is not None else down_idx
                self.previous_waypoint = best_wp
                return best_wp, "mv"
            else:
                self.previous_waypoint = lam
                return lam, "tk"

        if dr_set:
            best_wp = max(dr_set, key=lambda n: maps.potentials[0][n])
            self.previous_waypoint = best_wp
            return best_wp, "mv"

        if self.previous_waypoint >= 0:
            pw = self.previous_waypoint
            if t.cell_states[pw] == CellState.U and maps.potentials[0][pw] > 0:
                return pw, "mv"

        self.state = ETMState.CPL
        self.escape_level = 1
        return self._compute_cpl()

    def _compute_cpl(self) -> tuple[int, str]:
        t = self.tiling
        maps = self.maps
        lam = self.current_cell

        while self.escape_level < self.maps.num_levels:
            level = self.escape_level
            coarse_lam = t.find_coarse_containing(lam, level)
            if coarse_lam is None:
                self.escape_level += 1
                continue

            n_coarse = len(t.levels[level])
            c0, r0 = divmod(coarse_lam, max(1, int(math.isqrt(n_coarse))))
            half = LOCAL_NEIGHBORHOOD_HL // 2
            grid_side = max(1, int(math.isqrt(n_coarse)))

            neighborhood = []
            for dr in range(-half, half + 1):
                for dc in range(-half, half + 1):
                    nr = r0 + dr
                    nc = c0 + dc
                    if 0 <= nr < grid_side and 0 <= nc < grid_side:
                        idx = nr * grid_side + nc
                        if idx < n_coarse:
                            neighborhood.append(idx)
            if coarse_lam not in neighborhood:
                neighborhood.append(coarse_lam)

            best_coarse = -1
            best_pot = -1.0
            for n in neighborhood:
                if n < len(maps.potentials[level]):
                    p = maps.potentials[level][n]
                    if p > best_pot:
                        best_pot = p
                        best_coarse = n

            if best_coarse >= 0 and best_pot > 0:
                cells = t.get_cells_in_coarse(best_coarse, level)
                unexplored = [c for c in cells if t.cell_states[c] == CellState.U]
                if unexplored:
                    wp = unexplored[0]
                    self.state = ETMState.CP0
                    self.previous_waypoint = wp
                    return wp, "mv"

            self.escape_level += 1

        self.state = ETMState.FN
        return self.current_cell, "sp"


# ---------------------------------------------------------------------------
# Waypoint Follower
# ---------------------------------------------------------------------------
class WaypointFollower:
    def __init__(self, kp_heading: float = KP_HEADING, kp_throttle: float = KP_THROTTLE):
        self.kp_heading = kp_heading
        self.kp_throttle = kp_throttle
        self.target_pos: np.ndarray | None = None
        self.wall_follow_active = False
        self._wf_steps = 0
        self._wf_max_steps = 80

    def set_target(self, target_pos_m: np.ndarray):
        self.target_pos = target_pos_m.copy()

    def start_wall_follow(self):
        self.wall_follow_active = True
        self._wf_steps = 0

    def stop_wall_follow(self):
        self.wall_follow_active = False
        self._wf_steps = 0

    def compute_action(
        self,
        robot_pos_m: np.ndarray,
        robot_heading: float,
        sensor_dists: list[float],
    ) -> tuple[float, float]:
        if self.target_pos is None:
            return 0.0, 0.0

        dx = self.target_pos[0] - robot_pos_m[0]
        dy = self.target_pos[1] - robot_pos_m[1]
        dist = math.hypot(dx, dy)

        if dist < EPSILON_M * 0.3:
            self.stop_wall_follow()
            return 0.0, 0.0

        min_front = min(sensor_dists[0], sensor_dists[1], sensor_dists[2])
        r_right = sensor_dists[4]
        r_left = sensor_dists[3]

        if self.wall_follow_active:
            self._wf_steps += 1
            if self._wf_steps > self._wf_max_steps:
                self.stop_wall_follow()
            elif min_front < 0.35:
                return 0.0, -1.0
            elif r_right > 0.6:
                return 0.3, -0.5
            elif r_right < 0.25:
                return 0.3, 0.5
            else:
                return 0.5, 0.0

        if min_front < SAFETY_THRESHOLD_M:
            self.start_wall_follow()
            if r_right > r_left:
                return 0.0, -1.0
            else:
                return 0.0, 1.0

        target_heading = math.atan2(dy, dx)
        heading_error = target_heading - robot_heading
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi

        steering = self.kp_heading * heading_error
        steering = max(-1.0, min(1.0, steering))

        throttle = self.kp_throttle * min(dist / LOOKAHEAD_M, 1.0)
        throttle = max(0.0, min(1.0, throttle))

        if min_front < SAFETY_THRESHOLD_M * 2.0:
            throttle *= max(0.3, min_front / (SAFETY_THRESHOLD_M * 2.0))

        return throttle * 2.0 - 1.0, steering


# ---------------------------------------------------------------------------
# EStarController
# ---------------------------------------------------------------------------
class EStarController:
    """
    Drop-in classical controller for RobotCoverageEnv.

    Usage:
        controller = EStarController(env)
        obs, info = env.reset()
        while not done:
            action = controller.step(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
    """

    def __init__(self, env, epsilon_m: float = EPSILON_M):
        self.env = env
        self.epsilon_m = epsilon_m

        self.tiling: EpsilonTiling | None = None
        self.maps: MAPS | None = None
        self.etm: ETM | None = None
        self.follower = WaypointFollower()

        self.initialized = False
        self.waypoint_world: np.ndarray | None = None
        self._seen_cells: set[int] = set()
        self._prev_collisions = 0
        self._blocked_waypoints: set[int] = set()
        self._wp_collision_count: dict[int, int] = {}

    def _init_tiling(self, field_grid, render_offset, pixels_per_meter):
        self.tiling = EpsilonTiling(
            field_grid, render_offset, pixels_per_meter, self.epsilon_m
        )
        self.tiling.dilate_obstacles(SAFETY_BUFFER_CELLS)
        self.maps = MAPS(self.tiling)
        self.etm = ETM(self.tiling, self.maps)

    def _get_sensor_world_hits(self, obs) -> list[tuple[float, float] | None]:
        env = self.env
        a = 1.0
        pos = env.agent_pos_m
        heading = env.agent_heading
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)

        origins_local = [
            (a / 2, a / 2),  (a / 2, 0),  (a / 2, -a / 2),
            (0, a / 2),      (0, -a / 2), (-a / 2, 0),
        ]
        angles_offset = [
            math.pi / 4, 0.0, -math.pi / 4,
            math.pi / 2, -math.pi / 2, math.pi,
        ]

        sensor_dists = obs["sensors"][:6]
        hit_points = []
        for i, ((lx, ly), ang_off) in enumerate(zip(origins_local, angles_offset)):
            gx = pos[0] + lx * cos_h - ly * sin_h
            gy = pos[1] + lx * sin_h + ly * cos_h
            sd = float(sensor_dists[i])
            if sd < 1.0 - 1e-3:
                hit_points.append((
                    gx + sd * math.cos(heading + ang_off),
                    gy + sd * math.sin(heading + ang_off),
                ))
            else:
                hit_points.append(None)
        return hit_points

    def _update_covered_cells(self):
        env = self.env
        pos = env.agent_pos_m
        t = self.tiling

        cell = t.world_to_cell(pos)
        if cell >= 0 and t.cell_states[cell] == CellState.U:
            t.mark_explored(cell)
            self.maps.update_explored(cell)

        ppc = t.pixels_per_cell
        pos_p = env._m_to_p(pos)
        half = max(1, ppc // 2)
        cx_p, cy_p = int(pos_p[0]), int(pos_p[1])
        gsp = env.grid_size_p
        coverage = env.coverage_map
        p2m_scale = 1.0 / env.pixels_per_meter
        ro = env.render_offset

        y1 = max(0, cy_p - half)
        y2 = min(gsp, cy_p + half + 1)
        x1 = max(0, cx_p - half)
        x2 = min(gsp, cx_p + half + 1)

        for py in range(y1, y2):
            for px in range(x1, x2):
                if coverage[py, px] > 0:
                    world_x = px * p2m_scale + ro[0]
                    world_y = py * p2m_scale + ro[1]
                    cidx = t.pixel_to_cell(px, py)
                    if t.cell_states[cidx] == CellState.U:
                        t.mark_explored(cidx)
                        self.maps.update_explored(cidx)

    def step(self, obs, info) -> np.ndarray:
        env = self.env

        if not self.initialized:
            self._init_tiling(
                env.field_grid, env.render_offset, env.pixels_per_meter
            )
            start_cell = self.tiling.world_to_cell(env.agent_pos_m)
            self.etm.initialize(start_cell)
            self.initialized = True
            self._prev_collisions = 0
            self._blocked_waypoints.clear()
            self._wp_collision_count.clear()

        cur_collisions = info.get("num_collisions", 0)
        if cur_collisions > self._prev_collisions:
            if self.waypoint_world is not None:
                wp_cell = self.tiling.world_to_cell(self.waypoint_world)
                self._blocked_waypoints.add(wp_cell)
                self._wp_collision_count[wp_cell] = (
                    self._wp_collision_count.get(wp_cell, 0) + 1
                )
                if self._wp_collision_count[wp_cell] >= 3:
                    self.tiling.mark_forbidden(wp_cell)
                    self.maps.potentials[0][wp_cell] = -1.0
                    self.maps._propagate_up(wp_cell)
            self._prev_collisions = cur_collisions

        hit_points = self._get_sensor_world_hits(obs)
        self.etm.update_from_sensors(env.agent_pos_m, hit_points)

        if self.waypoint_world is not None:
            wp_cell = self.tiling.world_to_cell(self.waypoint_world)
            if self.tiling.cell_states[wp_cell] != CellState.U:
                self.etm.mark_current_explored()

        self._update_covered_cells()

        wp_idx, command = self.etm.compute_waypoint()

        if command == "sp":
            self.initialized = False
            return np.array([0.0, 0.0], dtype=np.float32)

        if command == "tk":
            self.etm.mark_current_explored()
            wp_idx = self.etm.current_cell

        if wp_idx in self._blocked_waypoints and wp_idx != self.etm.current_cell:
            self.etm.previous_waypoint = -1
            wp_idx, command = self.etm.compute_waypoint()
            if command == "sp":
                self.initialized = False
                return np.array([0.0, 0.0], dtype=np.float32)

        if wp_idx != self.etm.current_cell:
            self.waypoint_world = self.tiling.cell_to_world(wp_idx)
            self.follower.set_target(self.waypoint_world)
        elif self.waypoint_world is None:
            self.waypoint_world = self.tiling.cell_to_world(wp_idx)
            self.follower.set_target(self.waypoint_world)

        sensor_dists = [float(obs["sensors"][i]) for i in range(6)]
        throttle, steering = self.follower.compute_action(
            env.agent_pos_m, env.agent_heading, sensor_dists
        )

        return np.array([throttle, steering], dtype=np.float32)

    def reset(self):
        self.tiling = None
        self.maps = None
        self.etm = None
        self.follower = WaypointFollower()
        self.initialized = False
        self.waypoint_world = None
        self._seen_cells.clear()
        self._prev_collisions = 0
        self._blocked_waypoints.clear()
        self._wp_collision_count.clear()
