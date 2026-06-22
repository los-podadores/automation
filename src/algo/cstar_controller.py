"""
C* Coverage Path Planning Algorithm

Implementation of the algorithm from:
  "C*: A Coverage Path Planning Algorithm for Unknown Environments
   using Rapidly Covering Graphs"
  Zongyuan Shen, James P. Wilson, Shalabh Gupta — University of Connecticut

Adapted for the v2 robot environment (src/v2/robot_env.py) with:
  - 6-ray sensors, 1m range
  - Noisy localization (σ_pos=0.01m, σ_heading=0.05rad)
  - Continuous differential-drive control
  - Random polygonal fields

Reference: IEEE Transactions on Robotics, DOI 10.1109/TRO.2026.3661719
"""

from __future__ import annotations

import enum
import heapq
import logging
import math
from dataclasses import dataclass, field

import numpy as np
import cv2

logger = logging.getLogger("CStar")
logger.setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
W_DEFAULT = 0.9             # sampling resolution (lap spacing)
SAFETY_THRESHOLD_M = 0.45
KP_HEADING = 1.5
KP_THROTTLE = 0.6
LOOKAHEAD_M = 0.8
WAYPOINT_REACHED_DIST = 0.15
OBSTACLE_PATCH_RADIUS = 3
UNKNOWN_THRESHOLD = 0.5     # cells with discovered < this are unknown


class NodeState(enum.IntEnum):
    OPEN = 0
    CLOSED = 1


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------
@dataclass
class RCGNode:
    id: int
    pos: np.ndarray
    lap_idx: int
    along_lap: float
    state: NodeState = NodeState.OPEN
    is_link: bool = False

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, RCGNode) and self.id == other.id


class RCG:
    """Rapidly Covering Graph."""

    def __init__(self):
        self.nodes: dict[int, RCGNode] = {}
        self.edges: dict[int, set[int]] = {}
        self._next_id = 0

    def add_node(self, pos: np.ndarray, lap_idx: int, along_lap: float,
                 is_link: bool = False) -> RCGNode:
        nid = self._next_id
        self._next_id += 1
        node = RCGNode(nid, pos.copy(), lap_idx, along_lap, NodeState.OPEN, is_link)
        self.nodes[nid] = node
        self.edges[nid] = set()
        return node

    def remove_node(self, nid: int):
        for other_id in list(self.edges.get(nid, set())):
            self.edges.get(other_id, set()).discard(nid)
        self.edges.pop(nid, None)
        self.nodes.pop(nid, None)

    def add_edge(self, n1: int, n2: int):
        if n1 in self.nodes and n2 in self.nodes:
            self.edges.setdefault(n1, set()).add(n2)
            self.edges.setdefault(n2, set()).add(n1)

    def has_edge(self, n1: int, n2: int) -> bool:
        return n2 in self.edges.get(n1, set())

    def neighbors(self, nid: int) -> set[int]:
        return self.edges.get(nid, set()).copy()

    def same_lap_neighbors(self, nid: int) -> tuple[int | None, int | None]:
        """Return (prev_along, next_along) neighbors on the same lap."""
        node = self.nodes[nid]
        same = sorted(
            (self.nodes[n] for n in self.edges.get(nid, set())
             if self.nodes[n].lap_idx == node.lap_idx and n != nid),
            key=lambda n: n.along_lap,
        )
        prev_n = None
        next_n = None
        for n in same:
            if n.along_lap < node.along_lap:
                prev_n = n
            elif n.along_lap > node.along_lap and next_n is None:
                next_n = n
        return prev_n, next_n


class WaypointFollower:
    """Pure-pursuit waypoint follower with obstacle avoidance."""

    def __init__(self, kp_heading=KP_HEADING, kp_throttle=KP_THROTTLE):
        self.kp_heading = kp_heading
        self.kp_throttle = kp_throttle
        self.target_pos: np.ndarray | None = None

    def set_target(self, target_pos_m: np.ndarray):
        self.target_pos = target_pos_m.copy()

    def compute_action(self, robot_pos_m: np.ndarray, robot_heading: float,
                       sensor_dists: list[float]) -> tuple[float, float]:
        if self.target_pos is None:
            return 0.0, 0.0

        dx = self.target_pos[0] - robot_pos_m[0]
        dy = self.target_pos[1] - robot_pos_m[1]
        dist = math.hypot(dx, dy)

        if dist < WAYPOINT_REACHED_DIST:
            return 0.0, 0.0

        min_front = min(sensor_dists[0], sensor_dists[1], sensor_dists[2])
        if min_front < SAFETY_THRESHOLD_M:
            r_right = sensor_dists[4]
            r_left = sensor_dists[3]
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
# C* Controller
# ---------------------------------------------------------------------------
class CStarController:
    """
    Drop-in classical controller for RobotCoverageEnv implementing the C*
    online coverage path planning algorithm.

    Usage:
        controller = CStarController(env)
        obs, info = env.reset()
        controller.reset()
        while not done:
            action = controller.step(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
    """

    def __init__(self, env, w: float = W_DEFAULT,
                 safety_threshold_m: float = SAFETY_THRESHOLD_M,
                 kp_heading: float = KP_HEADING,
                 kp_throttle: float = KP_THROTTLE):
        self.env = env
        self.w = w
        self.safety_threshold_m = safety_threshold_m

        # Internal state
        self.rcg = RCG()
        self.current_node: RCGNode | None = None
        self.goal_node: RCGNode | None = None
        self.follower = WaypointFollower(kp_heading, kp_throttle)
        self.initialized = False

        # Geometry (set on init from initial heading)
        self.sweep_dir = np.array([1.0, 0.0])
        self.lap_dir = np.array([0.0, 1.0])
        self.lap_origin = np.array([0.0, 0.0])

        # Laps: lap_idx -> list of node IDs ordered by along_lap
        self.laps: dict[int, list[int]] = {}

        # Internal map
        self._internal_obstacles: np.ndarray | None = None
        self._discovered: np.ndarray | None = None
        self._sampled: np.ndarray | None = None  # cells that have been assigned to a lap sample
        self._prev_discovered: np.ndarray | None = None  # snapshot of _discovered at last sampling
        self._prev_discovered_count = 0

        # Track placed sample positions to avoid re-sampling
        self._placed_samples: list[np.ndarray] = []
        self._placed_laps: list[int] = []

        # Retreat nodes for dead-end escape
        self._retreat_nodes: set[int] = set()

        # TSP waypoint queue
        self._tsp_queue: list[np.ndarray] = []

        # Coverage hole detection
        self._last_hole_check_step = 0
        self._hole_check_interval = 10

        # Waypoint tracking
        self._waypoint_world: np.ndarray | None = None
        self._prev_collisions = 0
        self._blocked_waypoints: set = set()

    # ---------------------------------------------------------------
    # Coordinate helpers
    # ---------------------------------------------------------------
    def _m_to_p(self, pos_m: np.ndarray) -> np.ndarray:
        return (pos_m - self.env.render_offset) * self.env.pixels_per_meter

    def _p_to_m(self, pos_p: np.ndarray) -> np.ndarray:
        return pos_p / self.env.pixels_per_meter + self.env.render_offset

    def _lap_index_of(self, pos_m: np.ndarray) -> int:
        diff = pos_m - self.lap_origin
        perp = np.dot(diff, self.lap_dir)
        return int(round(perp / self.w))

    def _along_lap_of(self, pos_m: np.ndarray) -> float:
        diff = pos_m - self.lap_origin
        return float(np.dot(diff, self.sweep_dir))

    def _is_obstacle(self, pos_m: np.ndarray) -> bool:
        pp = self._m_to_p(pos_m)
        ix, iy = int(pp[0]), int(pp[1])
        gsp = self.env.grid_size_p
        if ix < 0 or ix >= gsp or iy < 0 or iy >= gsp:
            return True
        return self._internal_obstacles[iy, ix] > 0

    def _edge_collision_free(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        d = math.dist(p1, p2)
        if d < 1e-6:
            return True
        steps = max(2, int(d / (self.w * 0.3)))
        for i in range(steps + 1):
            t = i / steps
            pt = p1 * (1 - t) + p2 * t
            if self._is_obstacle(pt):
                return False
        return True

    # ---------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------
    def _init_internal_map(self):
        env = self.env
        gsp = env.grid_size_p
        self._internal_obstacles = np.zeros((gsp, gsp), dtype=np.float32)
        self._discovered = np.zeros((gsp, gsp), dtype=bool)
        self._sampled = np.zeros((gsp, gsp), dtype=bool)

        # Seed boundary from environment obstacle map
        self._internal_obstacles[env.obstacle_map > 0] = 1.0
        self._discovered[env.obstacle_map > 0] = True

    def _init_geometry(self):
        self.sweep_dir = np.array([
            math.cos(self.env.agent_heading),
            math.sin(self.env.agent_heading),
        ])
        self.lap_dir = np.array([-self.sweep_dir[1], self.sweep_dir[0]])
        self.lap_origin = self.env.agent_pos_m.copy()

    # ---------------------------------------------------------------
    # Internal map updates
    # ---------------------------------------------------------------
    def _update_internal_map_from_sensors(self, sensor_dists: list[float]):
        env = self.env
        pos = env.agent_pos_m
        heading = env.agent_heading
        a = 1.0  # ROBOT_SIDE

        origins_local = [
            (a / 2, a / 2), (a / 2, 0), (a / 2, -a / 2),
            (0, a / 2), (0, -a / 2), (-a / 2, 0),
        ]
        angles_offset = [
            math.pi / 4, 0.0, -math.pi / 4,
            math.pi / 2, -math.pi / 2, math.pi,
        ]

        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        ppm = env.pixels_per_meter
        gsp = env.grid_size_p

        for (lx, ly), ang_off, sd in zip(origins_local, angles_offset, sensor_dists):
            gx = pos[0] + lx * cos_h - ly * sin_h
            gy = pos[1] + lx * sin_h + ly * cos_h
            ray_angle = heading + ang_off

            if sd < 1.0 - 1e-3:
                hit_x = gx + sd * math.cos(ray_angle)
                hit_y = gy + sd * math.sin(ray_angle)
                pp = self._m_to_p(np.array([hit_x, hit_y]))
                ix, iy = int(pp[0]), int(pp[1])
                r = OBSTACLE_PATCH_RADIUS
                y1 = max(0, iy - r)
                y2 = min(gsp, iy + r + 1)
                x1 = max(0, ix - r)
                x2 = min(gsp, ix + r + 1)
                if y1 < y2 and x1 < x2:
                    self._internal_obstacles[y1:y2, x1:x2] = 1.0

            # Mark ray path as discovered
            step_m = 0.05
            for i in range(1, int(1.0 / step_m) + 1):
                px = gx + math.cos(ray_angle) * step_m * i
                py = gy + math.sin(ray_angle) * step_m * i
                pp = self._m_to_p(np.array([px, py]))
                ix, iy = int(pp[0]), int(pp[1])
                if 0 <= ix < gsp and 0 <= iy < gsp:
                    self._discovered[iy, ix] = True
                else:
                    break

    # ---------------------------------------------------------------
    # Sampling front and frontier samples
    # ---------------------------------------------------------------
    def _compute_sampling_front(self) -> np.ndarray:
        """Return uncovered free cells available for new RCG samples.

        Uses the environment's coverage map. If RCG is small, returns all
        uncovered free cells. When RCG grows large, limits to cells near
        the coverage frontier (cells adjacent to unknown space).
        """
        gsp = self.env.grid_size_p

        # Free cells that haven't been covered
        env_cov = self.env.coverage_map > 0
        uncovered_free = self._discovered.copy()
        uncovered_free[self._internal_obstacles > 0] = False
        uncovered_free[env_cov] = False

        if not uncovered_free.any():
            return uncovered_free

        # For large RCG, limit to frontier cells (near unknown space)
        if len(self.rcg.nodes) > 50:
            unknown = ~self._discovered & (self._internal_obstacles == 0)
            k3 = np.ones((3, 3), dtype=np.uint8)
            unknown_dilated = cv2.dilate(unknown.astype(np.uint8), k3, iterations=1)
            # Only keep uncovered free cells near unknown space
            uncovered_free = uncovered_free & (unknown_dilated > 0)

        return uncovered_free

    def _mark_sampled(self, pos: np.ndarray):
        """Mark cells very close to pos as sampled (narrow band along path)."""
        ppm = self.env.pixels_per_meter
        # Mark a narrow band (quarter of w) rather than full w radius
        r_px = max(2, int(self.w * ppm * 0.25))
        pp = self._m_to_p(pos)
        cx, cy = int(pp[0]), int(pp[1])
        gsp = self.env.grid_size_p
        y1 = max(0, cy - r_px)
        y2 = min(gsp, cy + r_px + 1)
        x1 = max(0, cx - r_px)
        x2 = min(gsp, cx + r_px + 1)
        self._sampled[y1:y2, x1:x2] = True

    def _identify_laps_in_front(self, front: np.ndarray) -> dict[int, np.ndarray]:
        """Identify which cells in the sampling front belong to which lap."""
        gsp = self.env.grid_size_p
        ppm = self.env.pixels_per_meter
        ro = self.env.render_offset

        lap_cells: dict[int, list[tuple[int, int]]] = {}

        ys, xs = np.where(front)
        for y, x in zip(ys, xs):
            world_x = x / ppm + ro[0]
            world_y = y / ppm + ro[1]
            pos_m = np.array([world_x, world_y])
            li = self._lap_index_of(pos_m)
            lap_cells.setdefault(li, []).append((x, y))

        # Convert to arrays of world positions
        lap_positions: dict[int, np.ndarray] = {}
        for li, cells in lap_cells.items():
            positions = []
            for cx, cy in cells:
                wx = cx / ppm + ro[0]
                wy = cy / ppm + ro[1]
                positions.append([wx, wy])
            lap_positions[li] = np.array(positions)

        return lap_positions

    def _place_frontier_samples(self, lap_positions: dict[int, np.ndarray]) -> list[tuple[np.ndarray, int, float]]:
        """Place frontier samples on laps. Returns list of (pos, lap_idx, along_lap)."""
        samples = []
        ppm = self.env.pixels_per_meter
        w_px = self.w * ppm

        for li, positions in lap_positions.items():
            if len(positions) == 0:
                continue

            # Project positions onto sweep direction to get along_lap coordinates
            diffs = positions - self.lap_origin
            along_laps = diffs @ self.sweep_dir

            # Sort by along_lap
            order = np.argsort(along_laps)
            sorted_positions = positions[order]
            sorted_along = along_laps[order]

            # Place samples at intervals of w
            if len(samples) == 0:
                # First sample: closest to lap origin
                ref_along = 0.0
            else:
                # Find existing nodes on this lap to reference
                existing = [n for n in self.rcg.nodes.values() if n.lap_idx == li]
                if existing:
                    ref_along = existing[0].along_lap
                else:
                    ref_along = sorted_along[0]

            # Snap to grid aligned with ref_along
            start_along = ref_along
            while start_along > sorted_along[0]:
                start_along -= self.w

            sample_along = start_along
            idx = 0
            while sample_along <= sorted_along[-1] + self.w * 0.1:
                # Find closest position to this along_lap value
                while idx < len(sorted_along) - 1 and sorted_along[idx] < sample_along - self.w * 0.1:
                    idx += 1

                if idx < len(sorted_along):
                    best_idx = idx
                    best_dist = abs(sorted_along[best_idx] - sample_along)
                    for j in range(idx, min(idx + 3, len(sorted_along))):
                        d = abs(sorted_along[j] - sample_along)
                        if d < best_dist:
                            best_dist = d
                            best_idx = j

                    if best_dist < self.w * 0.6:
                        pos = sorted_positions[best_idx]
                        # Check if it's a frontier: ball of radius w contains unknown or obstacle
                        if self._is_frontier_sample(pos):
                            # Check not too close to any already-placed sample on SAME lap
                            too_close = False
                            for sp, sp_lap in zip(self._placed_samples, self._placed_laps):
                                if sp_lap == li and math.dist(pos, sp) < self.w * 0.4:
                                    too_close = True
                                    break
                            if not too_close:
                                samples.append((pos.copy(), li, sample_along))
                                self._placed_samples.append(pos.copy())
                                self._placed_laps.append(li)

                sample_along += self.w

        return samples

    def _is_frontier_sample(self, pos: np.ndarray) -> bool:
        """Check if a ball of radius w around pos contains unknown or obstacle area."""
        ppm = self.env.pixels_per_meter
        r_px = int(self.w * ppm) + 1
        pp = self._m_to_p(pos)
        cx, cy = int(pp[0]), int(pp[1])
        gsp = self.env.grid_size_p

        y1 = max(0, cy - r_px)
        y2 = min(gsp, cy + r_px + 1)
        x1 = max(0, cx - r_px)
        x2 = min(gsp, cx + r_px + 1)

        if y1 >= y2 or x1 >= x2:
            return False

        # Check if any cell in the ball is unknown or obstacle
        region_obs = self._internal_obstacles[y1:y2, x1:x2]
        region_disc = self._discovered[y1:y2, x1:x2]

        if region_obs.any():
            return True

        # Unknown = not discovered and not obstacle
        unknown = np.logical_not(region_disc) & (region_obs == 0)
        return bool(unknown.any())

    def _mark_sampled(self, pos: np.ndarray):
        """Mark cells within w of pos as sampled so they don't appear in the front."""
        ppm = self.env.pixels_per_meter
        r_px = int(self.w * ppm) + 1
        pp = self._m_to_p(pos)
        cx, cy = int(pp[0]), int(pp[1])
        gsp = self.env.grid_size_p
        y1 = max(0, cy - r_px)
        y2 = min(gsp, cy + r_px + 1)
        x1 = max(0, cx - r_px)
        x2 = min(gsp, cx + r_px + 1)
        self._sampled[y1:y2, x1:x2] = True

    # ---------------------------------------------------------------
    # RCG expansion and pruning
    # ---------------------------------------------------------------
    def _expand_rcg(self, samples: list[tuple[np.ndarray, int, float]]):
        """Add frontier samples as nodes and create edges."""
        new_node_ids = []
        for pos, li, along in samples:
            node = self.rcg.add_node(pos, li, along)
            self.laps.setdefault(li, []).append(node.id)
            new_node_ids.append(node.id)
            logger.debug("Added node %d on lap %d at along=%.2f", node.id, li, along)

        # Sort each lap by along_lap
        for li in self.laps:
            self.laps[li].sort(key=lambda nid: self.rcg.nodes[nid].along_lap)

        # Connect new nodes to neighbors
        for nid in new_node_ids:
            node = self.rcg.nodes[nid]

            # Same-lap neighbors (adjacent in sorted order)
            lap_nodes = self.laps.get(node.lap_idx, [])
            idx = lap_nodes.index(nid) if nid in lap_nodes else -1
            if idx > 0:
                prev_id = lap_nodes[idx - 1]
                if self._edge_collision_free(node.pos, self.rcg.nodes[prev_id].pos):
                    self.rcg.add_edge(nid, prev_id)
            if idx < len(lap_nodes) - 1:
                next_id = lap_nodes[idx + 1]
                if self._edge_collision_free(node.pos, self.rcg.nodes[next_id].pos):
                    self.rcg.add_edge(nid, next_id)

            # Adjacent-lap neighbors
            for adj_li in [node.lap_idx - 1, node.lap_idx + 1]:
                adj_lap_nodes = self.laps.get(adj_li, [])
                for adj_nid in adj_lap_nodes:
                    adj_node = self.rcg.nodes[adj_nid]
                    d = math.dist(node.pos, adj_node.pos)
                    if d <= math.sqrt(2) * self.w:
                        if self._edge_collision_free(node.pos, adj_node.pos):
                            self.rcg.add_edge(nid, adj_nid)

        # Also connect existing boundary nodes to new nodes
        for nid in new_node_ids:
            node = self.rcg.nodes[nid]
            for adj_li in [node.lap_idx - 1, node.lap_idx + 1]:
                adj_lap_nodes = self.laps.get(adj_li, [])
                for adj_nid in adj_lap_nodes:
                    if adj_nid in self.rcg.nodes and not self.rcg.has_edge(nid, adj_nid):
                        adj_node = self.rcg.nodes[adj_nid]
                        d = math.dist(node.pos, adj_node.pos)
                        if d <= math.sqrt(2) * self.w:
                            if self._edge_collision_free(node.pos, adj_node.pos):
                                self.rcg.add_edge(nid, adj_nid)

    def _prune_rcg(self):
        """Prune inessential nodes and edges from the RCG."""
        if not self.rcg.nodes:
            return

        # --- Node pruning ---
        inessential_nodes = set()
        for nid, node in list(self.rcg.nodes.items()):
            if node.is_link:
                continue
            if not self._is_essential_node(nid):
                inessential_nodes.add(nid)

        # Prune inessential nodes: merge same-lap neighbors
        for nid in list(inessential_nodes):
            if nid not in self.rcg.nodes:
                continue
            node = self.rcg.nodes[nid]
            prev_n, next_n = self.rcg.same_lap_neighbors(nid)

            # Remove cross-lap edges
            for other_id in list(self.rcg.neighbors(nid)):
                other_node = self.rcg.nodes.get(other_id)
                if other_node and other_node.lap_idx != node.lap_idx:
                    self.rcg.edges[nid].discard(other_id)
                    self.rcg.edges[other_id].discard(nid)

            # Merge same-lap neighbors
            if prev_n and next_n:
                if not self.rcg.has_edge(prev_n.id, next_n.id):
                    if self._edge_collision_free(prev_n.pos, next_n.pos):
                        self.rcg.add_edge(prev_n.id, next_n.id)

            self.rcg.remove_node(nid)
            # Update lap list
            lap_list = self.laps.get(node.lap_idx, [])
            if nid in lap_list:
                lap_list.remove(nid)

        # --- Edge pruning ---
        edges_to_prune = set()
        for nid, node in self.rcg.nodes.items():
            if node.is_link:
                continue
            for other_id in list(self.rcg.neighbors(nid)):
                if other_id not in self.rcg.nodes:
                    continue
                other_node = self.rcg.nodes[other_id]
                if not self._is_essential_edge(nid, other_id):
                    edges_to_prune.add((min(nid, other_id), max(nid, other_id)))

        for n1, n2 in edges_to_prune:
            self.rcg.edges.get(n1, set()).discard(n2)
            self.rcg.edges.get(n2, set()).discard(n1)

    def _is_essential_node(self, nid: int) -> bool:
        """Check if node is essential per Definition III.8."""
        node = self.rcg.nodes.get(nid)
        if node is None:
            return False

        # Condition 1: adjacent to unknown area
        if self._is_frontier_sample(node.pos):
            return True

        # Condition 2: end-node of a lap
        lap_list = self.laps.get(node.lap_idx, [])
        if lap_list and (nid == lap_list[0] or nid == lap_list[-1]):
            return True

        # Condition 3: connected to end-node of adjacent lap
        for other_id in self.rcg.neighbors(nid):
            other = self.rcg.nodes.get(other_id)
            if other is None or other.lap_idx == node.lap_idx:
                continue
            other_lap = self.laps.get(other.lap_idx, [])
            if not other_lap:
                continue
            if other_id == other_lap[0] or other_id == other_lap[-1]:
                # other is an end-node; check conditions 3a/3b
                # Find other's neighbors on node's lap
                other_on_my_lap = [
                    n for n in self.rcg.neighbors(other_id)
                    if self.rcg.nodes[n].lap_idx == node.lap_idx and n != nid
                ]
                if not other_on_my_lap:
                    return True  # 3a
                # 3b: check if edge (nid, other_id) is closest to obstacle
                if all(not self.rcg.nodes[n].is_link for n in other_on_my_lap):
                    return True  # Simplified: accept as essential

        return False

    def _is_essential_edge(self, n1_id: int, n2_id: int) -> bool:
        """Check if edge is essential per Definition III.9."""
        n1 = self.rcg.nodes.get(n1_id)
        n2 = self.rcg.nodes.get(n2_id)
        if n1 is None or n2 is None:
            return False

        # Both must be essential
        if not self._is_essential_node(n1_id) or not self._is_essential_node(n2_id):
            return False

        # Condition 1: same lap
        if n1.lap_idx == n2.lap_idx:
            return True

        # Condition 2: adjacent laps
        if abs(n1.lap_idx - n2.lap_idx) != 1:
            return False

        # 2a: both are end nodes
        lap1 = self.laps.get(n1.lap_idx, [])
        lap2 = self.laps.get(n2.lap_idx, [])
        n1_is_end = lap1 and (n1_id == lap1[0] or n1_id == lap1[-1])
        n2_is_end = lap2 and (n2_id == lap2[0] or n2_id == lap2[-1])

        if n1_is_end and n2_is_end:
            return True

        # 2b: one is end, other is not
        if n1_is_end and not n2_is_end:
            other_on_my_lap = [
                n for n in self.rcg.neighbors(n1_id)
                if self.rcg.nodes[n].lap_idx == n2.lap_idx and n != n2_id
            ]
            if not other_on_my_lap:
                return True
            return True  # Simplified

        if n2_is_end and not n1_is_end:
            other_on_my_lap = [
                n for n in self.rcg.neighbors(n2_id)
                if self.rcg.nodes[n].lap_idx == n1.lap_idx and n != n1_id
            ]
            if not other_on_my_lap:
                return True
            return True  # Simplified

        return False

    # ---------------------------------------------------------------
    # Goal node selection (Algorithm 1)
    # ---------------------------------------------------------------
    def _select_goal_node(self) -> RCGNode | None:
        """Select next goal node using priority: left, up, down, right.

        For left/right (adjacent lap), picks the nearest OPEN node.
        For up/down (same lap), picks the immediate neighbor.
        """
        if self.current_node is None:
            return None

        cn = self.current_node
        lap_list = self.laps.get(cn.lap_idx, [])
        idx = lap_list.index(cn.id) if cn.id in lap_list else -1

        # Left neighbor: nearest OPEN node on left adjacent lap
        left_lap = self.laps.get(cn.lap_idx - 1, [])
        left_candidates = []
        for lid in left_lap:
            ln = self.rcg.nodes[lid]
            if ln.state == NodeState.OPEN:
                dist = abs(ln.along_lap - cn.along_lap)
                if dist < self.w * 1.5:
                    left_candidates.append((dist, ln))
        if left_candidates:
            left_candidates.sort(key=lambda x: x[0])
            return left_candidates[0][1]

        # Up neighbor (higher along_lap on same lap)
        if idx >= 0 and idx < len(lap_list) - 1:
            up_id = lap_list[idx + 1]
            up_node = self.rcg.nodes[up_id]
            if up_node.state == NodeState.OPEN:
                return up_node

        # Down neighbor (lower along_lap on same lap)
        if idx > 0:
            down_id = lap_list[idx - 1]
            down_node = self.rcg.nodes[down_id]
            if down_node.state == NodeState.OPEN:
                return down_node

        # Right neighbor: nearest OPEN node on right adjacent lap
        right_lap = self.laps.get(cn.lap_idx + 1, [])
        right_candidates = []
        for rid in right_lap:
            rn = self.rcg.nodes[rid]
            if rn.state == NodeState.OPEN:
                dist = abs(rn.along_lap - cn.along_lap)
                if dist < self.w * 1.5:
                    right_candidates.append((dist, rn))
        if right_candidates:
            right_candidates.sort(key=lambda x: x[0])
            return right_candidates[0][1]

        return None  # Dead-end

    # ---------------------------------------------------------------
    # State update (Algorithm 2)
    # ---------------------------------------------------------------
    def _update_state(self, goal: RCGNode):
        """Update node states and create link nodes if needed (Algorithm 2)."""
        cn = self.current_node
        if cn is None:
            return

        lap_list = self.laps.get(cn.lap_idx, [])
        idx = lap_list.index(cn.id) if cn.id in lap_list else -1

        prev_n = self.rcg.nodes[lap_list[idx - 1]] if idx > 0 else None
        next_n = self.rcg.nodes[lap_list[idx + 1]] if idx < len(lap_list) - 1 else None

        # Mark current node as Closed
        cn.state = NodeState.CLOSED

        # Create link node when moving to left lap and current is Closed
        # Link node ensures uncovered lap segment between current and Open neighbor
        if goal.lap_idx != cn.lap_idx:
            if goal.lap_idx < cn.lap_idx:
                # Moving to left lap
                if next_n and next_n.state == NodeState.OPEN:
                    d = math.dist(cn.pos, next_n.pos)
                    if d > self.w:
                        link_pos = cn.pos + self.sweep_dir * self.w
                        link = self.rcg.add_node(link_pos, cn.lap_idx,
                                                 self._along_lap_of(link_pos), is_link=True)
                        self.laps.setdefault(cn.lap_idx, []).append(link.id)
                        self.laps[cn.lap_idx].sort(key=lambda nid: self.rcg.nodes[nid].along_lap)
                        if self._edge_collision_free(cn.pos, link.pos):
                            self.rcg.add_edge(cn.id, link.id)
                        if self._edge_collision_free(link.pos, next_n.pos):
                            self.rcg.add_edge(link.id, next_n.id)
            else:
                # Moving to right lap
                if prev_n and prev_n.state == NodeState.OPEN:
                    d = math.dist(cn.pos, prev_n.pos)
                    if d > self.w:
                        link_pos = cn.pos - self.sweep_dir * self.w
                        link = self.rcg.add_node(link_pos, cn.lap_idx,
                                                 self._along_lap_of(link_pos), is_link=True)
                        self.laps.setdefault(cn.lap_idx, []).append(link.id)
                        self.laps[cn.lap_idx].sort(key=lambda nid: self.rcg.nodes[nid].along_lap)
                        if self._edge_collision_free(cn.pos, link.pos):
                            self.rcg.add_edge(cn.id, link.id)
                        if self._edge_collision_free(link.pos, prev_n.pos):
                            self.rcg.add_edge(link.id, prev_n.id)

    # ---------------------------------------------------------------
    # Retreat node management
    # ---------------------------------------------------------------
    def _update_retreat_nodes(self):
        """Maintain retreat nodes: all Open nodes within sqrt(2)*w of robot trajectory."""
        if self.current_node is None:
            return

        robot_pos = self.env.agent_pos_m
        max_dist = math.sqrt(2) * self.w

        to_remove = set()
        for nid in self._retreat_nodes:
            if nid not in self.rcg.nodes:
                to_remove.add(nid)
            elif self.rcg.nodes[nid].state != NodeState.OPEN:
                to_remove.add(nid)
            elif math.dist(robot_pos, self.rcg.nodes[nid].pos) > max_dist * 3:
                to_remove.add(nid)

        self._retreat_nodes -= to_remove

        for nid, node in self.rcg.nodes.items():
            if node.state == NodeState.OPEN and nid not in self._retreat_nodes:
                if math.dist(robot_pos, node.pos) <= max_dist:
                    self._retreat_nodes.add(nid)

    # ---------------------------------------------------------------
    # Dead-end escape
    # ---------------------------------------------------------------
    def _escape_dead_end(self) -> RCGNode | None:
        """Find nearest retreat node using A* on RCG."""
        if not self._retreat_nodes or self.current_node is None:
            return None

        best_node = None
        best_cost = float('inf')

        for retreat_id in self._retreat_nodes:
            if retreat_id not in self.rcg.nodes:
                continue
            path = self._astar_rcg(self.current_node.id, retreat_id)
            if path is not None:
                cost = sum(
                    math.dist(self.rcg.nodes[path[i]].pos, self.rcg.nodes[path[i + 1]].pos)
                    for i in range(len(path) - 1)
                )
                if cost < best_cost:
                    best_cost = cost
                    best_node = self.rcg.nodes[retreat_id]

        return best_node

    def _astar_rcg(self, start_id: int, goal_id: int) -> list[int] | None:
        """A* on the RCG graph."""
        if start_id == goal_id:
            return [start_id]

        open_set = [(0, start_id)]
        came_from: dict[int, int] = {}
        g_score: dict[int, float] = {start_id: 0}

        while open_set:
            _, curr = heapq.heappop(open_set)
            if curr == goal_id:
                path = [curr]
                while curr in came_from:
                    curr = came_from[curr]
                    path.append(curr)
                path.reverse()
                return path

            for nxt in self.rcg.neighbors(curr):
                if nxt not in self.rcg.nodes:
                    continue
                tentative = g_score[curr] + math.dist(
                    self.rcg.nodes[curr].pos, self.rcg.nodes[nxt].pos
                )
                if tentative < g_score.get(nxt, float('inf')):
                    came_from[nxt] = curr
                    g_score[nxt] = tentative
                    h = math.dist(self.rcg.nodes[nxt].pos, self.rcg.nodes[goal_id].pos)
                    heapq.heappush(open_set, (tentative + h, nxt))

        return None

    # ---------------------------------------------------------------
    # Coverage hole detection and TSP
    # ---------------------------------------------------------------
    def _detect_coverage_holes(self) -> list[set[int]]:
        """Detect coverage holes using flood-fill on RCG Open nodes."""
        if self.current_node is None:
            return []

        visited = set()
        holes = []

        for start_id in list(self.rcg.nodes.keys()):
            if start_id in visited:
                continue
            node = self.rcg.nodes[start_id]
            if node.state != NodeState.OPEN:
                visited.add(start_id)
                continue
            if start_id == self.current_node.id:
                visited.add(start_id)
                continue

            # BFS to find connected component
            component = set()
            queue = [start_id]
            has_unknown_neighbor = False

            while queue:
                curr = queue.pop(0)
                if curr in visited or curr in component:
                    continue
                component.add(curr)
                visited.add(curr)

                curr_node = self.rcg.nodes[curr]
                if self._is_frontier_sample(curr_node.pos):
                    has_unknown_neighbor = True

                for nbr in self.rcg.neighbors(curr):
                    if nbr in self.rcg.nodes:
                        nbr_node = self.rcg.nodes[nbr]
                        if nbr_node.state == NodeState.OPEN and nbr not in component:
                            queue.append(nbr)

            # Check if this component is a coverage hole
            if not has_unknown_neighbor and len(component) > 0:
                # Verify it's surrounded by obstacles and closed nodes
                holes.append(component)

        return holes

    def _setup_and_solve_tsp(self, hole_nodes: set[int]) -> list[np.ndarray] | None:
        """Set up and solve TSP for a coverage hole."""
        if self.current_node is None:
            return None

        # Collect all nodes in the TSP
        tsp_node_ids = list(hole_nodes)
        if self.current_node.id not in tsp_node_ids:
            tsp_node_ids.append(self.current_node.id)

        # Determine end node
        end_node_id = None
        for nid in self.rcg.neighbors(self.current_node.id):
            n = self.rcg.nodes[nid]
            if n.state == NodeState.OPEN and nid not in hole_nodes:
                end_node_id = self.current_node.id
                break

        if end_node_id is None and self.goal_node:
            for nid in self.rcg.neighbors(self.goal_node.id):
                n = self.rcg.nodes[nid]
                if n.state == NodeState.OPEN and nid not in hole_nodes:
                    end_node_id = self.goal_node.id
                    break

        if end_node_id is None:
            end_node_id = self.current_node.id

        if end_node_id not in tsp_node_ids:
            tsp_node_ids.append(end_node_id)

        if len(tsp_node_ids) < 2:
            return None

        # Build distance matrix using A*
        n = len(tsp_node_ids)
        dist_matrix = np.zeros((n, n))
        path_cache: dict[tuple[int, int], list[int] | None] = {}

        for i in range(n):
            for j in range(i + 1, n):
                ni, nj = tsp_node_ids[i], tsp_node_ids[j]
                if self.rcg.has_edge(ni, nj):
                    d = math.dist(self.rcg.nodes[ni].pos, self.rcg.nodes[nj].pos)
                    dist_matrix[i, j] = d
                    dist_matrix[j, i] = d
                    path_cache[(ni, nj)] = [ni, nj]
                    path_cache[(nj, ni)] = [nj, ni]
                else:
                    path = self._astar_rcg(ni, nj)
                    if path is not None:
                        d = sum(
                            math.dist(self.rcg.nodes[path[k]].pos, self.rcg.nodes[path[k + 1]].pos)
                            for k in range(len(path) - 1)
                        )
                        dist_matrix[i, j] = d
                        dist_matrix[j, i] = d
                        path_cache[(ni, nj)] = path
                        path_cache[(nj, ni)] = list(reversed(path))
                    else:
                        dist_matrix[i, j] = float('inf')
                        dist_matrix[j, i] = float('inf')

        # Nearest-neighbor TSP
        start_idx = tsp_node_ids.index(end_node_id)
        route = [start_idx]
        unvisited = set(range(n)) - {start_idx}

        while unvisited:
            curr = route[-1]
            nearest = min(unvisited, key=lambda j: dist_matrix[curr, j])
            route.append(nearest)
            unvisited.remove(nearest)

        # 2-opt improvement
        route = self._two_opt(route, dist_matrix)

        # Convert to positions
        path_positions = []
        for i in range(len(route) - 1):
            ni = tsp_node_ids[route[i]]
            nj = tsp_node_ids[route[i + 1]]
            key = (ni, nj)
            if key in path_cache and path_cache[key] is not None:
                for nid in path_cache[key][1:]:
                    path_positions.append(self.rcg.nodes[nid].pos.copy())
            else:
                path_positions.append(self.rcg.nodes[nj].pos.copy())

        return path_positions

    def _two_opt(self, route: list[int], dist_matrix: np.ndarray) -> list[int]:
        """Improve TSP route using 2-opt."""
        improved = True
        best = route[:]
        while improved:
            improved = False
            for i in range(1, len(best) - 1):
                for j in range(i + 1, len(best)):
                    new_route = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                    old_dist = sum(dist_matrix[best[k], best[k + 1]] for k in range(len(best) - 1))
                    new_dist = sum(dist_matrix[new_route[k], new_route[k + 1]] for k in range(len(new_route) - 1))
                    if new_dist < old_dist - 1e-6:
                        best = new_route
                        improved = True
        return best

    # ---------------------------------------------------------------
    # Initialization and reset
    # ---------------------------------------------------------------
    def reset(self):
        self.rcg = RCG()
        self.current_node = None
        self.goal_node = None
        self.follower = WaypointFollower()
        self.initialized = False
        self.sweep_dir = np.array([1.0, 0.0])
        self.lap_dir = np.array([0.0, 1.0])
        self.lap_origin = np.array([0.0, 0.0])
        self.laps = {}
        self._internal_obstacles = None
        self._discovered = None
        self._sampled = None
        self._prev_discovered_count = 0
        self._prev_discovered = None
        self._placed_samples = []
        self._placed_laps = []
        self._retreat_nodes = set()
        self._tsp_queue = []
        self._last_hole_check_step = 0
        self._waypoint_world = None
        self._prev_collisions = 0
        self._blocked_waypoints = set()

    def _first_init(self):
        """One-time initialization on first step."""
        self._init_internal_map()
        self._init_geometry()

        start_pos = self.env.agent_pos_m.copy()
        li = self._lap_index_of(start_pos)
        along = self._along_lap_of(start_pos)

        # Create initial node at start position
        start_node = self.rcg.add_node(start_pos, li, along)
        self.laps.setdefault(li, []).append(start_node.id)
        self.laps[li].sort(key=lambda nid: self.rcg.nodes[nid].along_lap)
        self.current_node = start_node

        # Snapshot discovered map for incremental sampling
        self._prev_discovered = self._discovered.copy()

        self.initialized = True
        logger.info("C* initialized: start=(%.2f,%.2f) lap=%d sweep=(%.2f,%.2f)",
                     start_pos[0], start_pos[1], li, self.sweep_dir[0], self.sweep_dir[1])

    # ---------------------------------------------------------------
    # Sampling and expansion helper
    # ---------------------------------------------------------------
    def _do_sampling_and_expand(self):
        """Detect sampling front, place frontier samples, expand RCG."""
        front = self._compute_sampling_front()
        new_disc = np.sum(front)
        if new_disc < 5:
            return

        lap_positions = self._identify_laps_in_front(front)
        samples = self._place_frontier_samples(lap_positions)

        if samples:
            self._expand_rcg(samples)
            self._prune_rcg()
            self._update_retreat_nodes()

        # Snapshot current discovered map for next iteration
        self._prev_discovered = self._discovered.copy()
        self._prev_discovered_count = int(np.sum(self._discovered))

    # ---------------------------------------------------------------
    # Main step
    # ---------------------------------------------------------------
    def step(self, obs: dict, info: dict) -> np.ndarray:
        env = self.env

        if not self.initialized:
            self._first_init()
            return self._compute_action(obs)

        # 1. Update internal map from sensors
        sensor_dists = [float(obs["sensors"][i]) for i in range(6)]
        self._update_internal_map_from_sensors(sensor_dists)

        # 2. Check collisions
        cur_collisions = info.get("num_collisions", 0)
        if cur_collisions > self._prev_collisions:
            if self._waypoint_world is not None:
                self._blocked_waypoints.add(tuple(self._waypoint_world))
            self._prev_collisions = cur_collisions

        # 3. Always try to expand RCG with newly discovered area
        self._do_sampling_and_expand()

        # 4. Check if waypoint reached
        waypoint_reached = False
        if self._waypoint_world is not None:
            d = math.dist(env.agent_pos_m, self._waypoint_world)
            if d < WAYPOINT_REACHED_DIST:
                waypoint_reached = True

        if self.goal_node is None and len(self.rcg.nodes) > 1:
            waypoint_reached = True

        # 5. If waypoint reached: select next goal
        if waypoint_reached:
            if self.current_node:
                logger.debug("Reached node %d at (%.2f,%.2f)",
                             self.current_node.id, self.current_node.pos[0], self.current_node.pos[1])

            # Update retreat nodes
            self._update_retreat_nodes()

            # Sample newly discovered area and expand RCG
            self._do_sampling_and_expand()

            # Check for coverage holes
            current_step = env.current_step
            if current_step - self._last_hole_check_step >= self._hole_check_interval:
                self._last_hole_check_step = current_step
                holes = self._detect_coverage_holes()
                if holes:
                    largest_hole = max(holes, key=len)
                    tsp_path = self._setup_and_solve_tsp(largest_hole)
                    if tsp_path and len(tsp_path) > 0:
                        self._tsp_queue = tsp_path
                        logger.info("Coverage hole detected with %d nodes, TSP path has %d waypoints",
                                    len(largest_hole), len(tsp_path))

            # If TSP queue has waypoints, follow them
            if self._tsp_queue:
                next_pos = self._tsp_queue.pop(0)
                self.follower.set_target(next_pos)
                self._waypoint_world = next_pos.copy()
                return self._compute_action(obs)

            # Select next goal node
            goal = self._select_goal_node()

            if goal is None:
                # Dead-end: try retreat nodes
                goal = self._escape_dead_end()
                if goal is None:
                    # No retreat nodes; if RCG has nodes, try any open node
                    open_nodes = [n for n in self.rcg.nodes.values()
                                  if n.state == NodeState.OPEN and n.id != self.current_node.id]
                    if open_nodes:
                        goal = min(open_nodes,
                                   key=lambda n: math.dist(env.agent_pos_m, n.pos))
                    else:
                        logger.info("Coverage complete: no more open nodes")
                        return np.array([0.0, 0.0], dtype=np.float32)
                logger.debug("Escaping dead-end to node %d at (%.2f,%.2f)",
                             goal.id, goal.pos[0], goal.pos[1])

            # Update state and set new goal
            self._update_state(goal)
            self.current_node = goal
            self.goal_node = goal
            self.follower.set_target(goal.pos)
            self._waypoint_world = goal.pos.copy()

        return self._compute_action(obs)

    def _compute_action(self, obs: dict) -> np.ndarray:
        """Compute [throttle, steering] from current waypoint."""
        sensor_dists = [float(obs["sensors"][i]) for i in range(6)]

        # Check if current waypoint is blocked
        if self._waypoint_world is not None:
            wp_tuple = tuple(self._waypoint_world)
            if wp_tuple in self._blocked_waypoints:
                # Force re-selection on next step
                self._waypoint_world = None
                if self.goal_node:
                    self.goal_node.state = NodeState.OPEN

        throttle, steering = self.follower.compute_action(
            self.env.agent_pos_m, self.env.agent_heading, sensor_dists
        )
        return np.array([throttle, steering], dtype=np.float32)
