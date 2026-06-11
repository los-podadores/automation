"""
Standalone 2D Robot Simulation in Pygame-ce.
Upgraded with "Bumper Escape Cooldown" to prevent micro-stuttering on acute corners.
"""

import heapq
import logging
import math

import pygame
from pygame.math import Vector2
from shapely.geometry import LineString, Point

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("HardwareSimulation")

# --- CONFIGURATION & CONSTANTS ---
WINDOW_SIZE = (1000, 800)
FPS = 60

COLOR_BG = (20, 24, 30)
COLOR_BOUNDARY = (46, 204, 113)
COLOR_OBSTACLE = (231, 76, 60)
COLOR_ROBOT = (52, 152, 219)
COLOR_RAY_HIT = (231, 76, 60)
COLOR_RAY_OR = (46, 204, 113)
COLOR_PATH_ACTUAL = (241, 196, 15)
COLOR_VIRTUAL_WALL = (155, 89, 182)
COLOR_TEXT = (236, 240, 241)
COLOR_PAINT = (52, 152, 219, 30)

ROBOT_RADIUS = 15
D_REF = 25.0
OR_VALUE = 60.0

V_NORMAL = 1.2
SAFETY_THRES = 35.0

KP = 0.015
KD = 0.05

GRID_SIZE = 15
MIN_FRONTIER_SIZE = 8

# --- GEOMETRY UTILITIES ---


def intersect_ray_segment(ray_origin, ray_dir, p1, p2):
    v = ray_dir
    ba = p2 - p1
    ao = p1 - ray_origin
    det = v.x * (-ba.y) - v.y * (-ba.x)
    if abs(det) < 1e-6:
        return None
    dist = (ao.x * (-ba.y) - ao.y * (-ba.x)) / det
    t = (v.x * ao.y - v.y * ao.x) / det
    if dist >= 0 and 0 <= t <= 1:
        return dist
    return None


def get_distance_to_geometry(origin, direction, segments, max_dist):
    min_d = max_dist
    for p1, p2 in segments:
        d = intersect_ray_segment(origin, direction, p1, p2)
        if d is not None and d < min_d:
            min_d = d
    return min_d


# --- SIMULATION CLASSES ---


class Robot:
    def __init__(self, x, y, heading_deg, boundary_pts):
        self.pos = Vector2(x, y)
        self.heading_angle = math.radians(heading_deg)
        self.radius = ROBOT_RADIUS
        self.map_boundary = boundary_pts

        self.phase = 1
        self.state = "FIND_WALL"
        self.wf_state = "NONE"

        self.r_front = OR_VALUE
        self.r_left = OR_VALUE
        self.r_right = OR_VALUE

        self.path_actual = [Vector2(self.pos)]
        self.current_layer_path = [Vector2(self.pos)]

        self.virtual_walls = []

        self.internal_map = {}
        self.waypoints = []
        self.current_waypoint_idx = 0

        self.v = 0.0
        self.omega = 0.0
        self._lost_timer = 0
        self.prev_error = 0.0
        self.bump_escape_timer = 0  # NEW: Cooldown timer for blind spot snag escapes

    def get_grid_coords(self, position):
        return int(position.x // GRID_SIZE), int(position.y // GRID_SIZE)

    def _mark_ray_on_map(self, origin, direction, dist, is_hit):
        steps = int(dist // (GRID_SIZE / 2.0))
        for i in range(steps + 1):
            pt = origin + direction * (i * (GRID_SIZE / 2.0))
            gx, gy = self.get_grid_coords(pt)
            if self.internal_map.get((gx, gy)) != "COVERED":
                self.internal_map[(gx, gy)] = "FREE"

        if is_hit:
            pt = origin + direction * dist
            gx, gy = self.get_grid_coords(pt)
            if self.internal_map.get((gx, gy)) != "COVERED":
                self.internal_map[(gx, gy)] = "WALL"

    def update_sensors(self, boundary_segments, obstacle_segments):
        h_vec = Vector2(math.cos(self.heading_angle), math.sin(self.heading_angle))

        all_segs = boundary_segments + obstacle_segments
        if self.state != "TRANSIT_TO_FRONTIER":
            all_segs += self.virtual_walls

        self.r_front = get_distance_to_geometry(self.pos, h_vec, all_segs, OR_VALUE)
        self._mark_ray_on_map(self.pos, h_vec, self.r_front, self.r_front < OR_VALUE)

        left_vec = h_vec.rotate(-90)
        self.r_left = get_distance_to_geometry(self.pos, left_vec, all_segs, OR_VALUE)
        self._mark_ray_on_map(self.pos, left_vec, self.r_left, self.r_left < OR_VALUE)

        right_vec = h_vec.rotate(90)
        self.r_right = get_distance_to_geometry(self.pos, right_vec, all_segs, OR_VALUE)
        self._mark_ray_on_map(
            self.pos, right_vec, self.r_right, self.r_right < OR_VALUE
        )

        gx, gy = self.get_grid_coords(self.pos)
        self.internal_map[(gx, gy)] = "COVERED"
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if Vector2(dx * GRID_SIZE, dy * GRID_SIZE).length() <= self.radius:
                    self.internal_map[(gx + dx, gy + dy)] = "COVERED"

    def get_valid_frontiers(self):
        free_cells = set(
            [pos for pos, state in self.internal_map.items() if state == "FREE"]
        )
        visited = set()
        targets = []

        gx, gy = self.get_grid_coords(self.pos)

        for cell in free_cells:
            if cell not in visited:
                cluster = []
                queue = [cell]
                visited.add(cell)

                while queue:
                    curr = queue.pop(0)
                    cluster.append(curr)
                    cx, cy = curr

                    for dx, dy in [
                        (0, 1),
                        (1, 0),
                        (0, -1),
                        (-1, 0),
                        (1, 1),
                        (-1, -1),
                        (1, -1),
                        (-1, 1),
                    ]:
                        nxt = (cx + dx, cy + dy)
                        if nxt in free_cells and nxt not in visited:
                            visited.add(nxt)
                            queue.append(nxt)

                if len(cluster) >= MIN_FRONTIER_SIZE:
                    best_dist = float("inf")
                    best_cell = None
                    for cx, cy in cluster:
                        dist = math.hypot(cx - gx, cy - gy)
                        if dist < best_dist:
                            best_dist = dist
                            best_cell = (cx, cy)
                    targets.append((best_dist, best_cell))

        targets.sort(key=lambda x: x[0])
        return [t[1] for t in targets]

    def plan_path_through_memory(self, start_pos, target_cell):
        start_cell = self.get_grid_coords(start_pos)

        frontier = [(0, start_cell)]
        came_from = {start_cell: None}
        cost_so_far = {start_cell: 0}

        while frontier:
            _, curr = heapq.heappop(frontier)

            if curr == target_cell:
                break

            for dx, dy in [
                (0, 1),
                (1, 0),
                (0, -1),
                (-1, 0),
                (1, 1),
                (-1, -1),
                (1, -1),
                (-1, 1),
            ]:
                nxt = (curr[0] + dx, curr[1] + dy)
                state = self.internal_map.get(nxt, "UNKNOWN")

                if state not in ["FREE", "COVERED"]:
                    continue

                wall_penalty = 0
                for nx in range(-2, 3):
                    for ny in range(-2, 3):
                        if self.internal_map.get((nxt[0] + nx, nxt[1] + ny)) == "WALL":
                            dist = math.hypot(nx, ny)
                            if dist < 2.5:
                                wall_penalty += 50 / (dist + 1)

                new_cost = cost_so_far[curr] + math.hypot(dx, dy) + wall_penalty
                if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                    cost_so_far[nxt] = new_cost
                    priority = new_cost + math.hypot(
                        target_cell[0] - nxt[0], target_cell[1] - nxt[1]
                    )
                    heapq.heappush(frontier, (priority, nxt))
                    came_from[nxt] = curr

        if target_cell not in came_from:
            return []

        path = []
        curr = target_cell
        while curr != start_cell:
            world_pos = Vector2(
                curr[0] * GRID_SIZE + GRID_SIZE / 2, curr[1] * GRID_SIZE + GRID_SIZE / 2
            )
            path.append(world_pos)
            curr = came_from[curr]
        path.reverse()
        return path

    def step(self, dt, boundary_segments, obstacle_segments):
        if self.state == "FINISHED":
            self.v, self.omega = 0.0, 0.0
            return

        self.update_sensors(boundary_segments, obstacle_segments)

        self.heading_angle += self.omega * dt

        # PHYSICAL BUMPER CHECK
        movement_vec = (
            Vector2(math.cos(self.heading_angle), math.sin(self.heading_angle))
            * self.v
            * dt
        )
        intended_pos = self.pos + movement_vec

        collision = False
        pt = Point(intended_pos.x, intended_pos.y)
        for p1, p2 in boundary_segments + obstacle_segments:
            line = LineString([(p1.x, p1.y), (p2.x, p2.y)])
            if pt.distance(line) < self.radius - 0.5:
                collision = True
                break

        if collision:
            self.v = 0.0
            recoil_dir = Vector2(
                math.cos(self.heading_angle), math.sin(self.heading_angle)
            )
            self.pos -= recoil_dir * 2.0

            if self.state == "TRANSIT_TO_FRONTIER":
                logger.warning(
                    "Transit collision! Aborting path and tracking obstacle."
                )
                self.state = "TURN_CONCAVE"
                self.wf_state = "TURN_CONCAVE"
                self.current_layer_path = []
                self.bump_escape_timer = 15
            else:
                if self.r_left <= self.radius + 5.0:
                    self.r_right = OR_VALUE + 50.0
                else:
                    # BLIND SPOT CORNER SNAG.
                    # Force the robot to pivot left for a few frames to clear the snag
                    # and sweep its sensor back onto the wall.
                    self.bump_escape_timer = 15
        else:
            self.pos = intended_pos

        if self.state == "TRANSIT_TO_FRONTIER":
            self.logic_transit()
        else:
            self.logic_coverage()

        if self.pos.distance_to(self.path_actual[-1]) > 5:
            self.path_actual.append(Vector2(self.pos))
            if self.state != "TRANSIT_TO_FRONTIER":
                if (
                    len(self.current_layer_path) == 0
                    or self.pos.distance_to(self.current_layer_path[-1]) > 5
                ):
                    self.current_layer_path.append(Vector2(self.pos))

    def logic_transit(self):
        if self.current_waypoint_idx >= len(self.waypoints):
            logger.info("Arrived at new frontier! Resuming concentric coverage.")
            self.state = "FIND_WALL"
            self.v, self.omega = 0.0, 0.0
            self.phase += 1
            return

        target = self.waypoints[self.current_waypoint_idx]
        dist_to_target = self.pos.distance_to(target)

        if dist_to_target < 10:
            self.current_waypoint_idx += 1
            return

        desired_vec = target - self.pos
        desired_angle = math.atan2(desired_vec.y, desired_vec.x)
        angle_diff = (desired_angle - self.heading_angle + math.pi) % (
            2 * math.pi
        ) - math.pi

        if abs(angle_diff) > math.radians(10):
            self.v = 0.0
            self.omega = math.copysign(0.25, angle_diff)
        else:
            self.v = V_NORMAL * 1.5
            self.omega = max(-0.1, min(0.1, 0.5 * angle_diff))

    def logic_coverage(self):
        # NEW: Override coverage logic if we are actively escaping a blind-spot bump
        if self.bump_escape_timer > 0:
            self.bump_escape_timer -= 1
            self.v = 0.0
            self.omega = -0.30  # Pivot hard left to clear the snag on the right chassis
            self.wf_state = "ESCAPE_BUMP"
            self.state = "ESCAPE_BUMP"
            self.prev_error = 0.0
            return

        # DYNAMIC D_REF (CORRIDOR CENTERING)
        current_d_ref = D_REF
        if self.r_left < OR_VALUE and self.r_right < OR_VALUE:
            current_d_ref = min(D_REF, (self.r_left + self.r_right) / 2.0)

        if self.state == "FIND_WALL":
            self._lost_timer = 0
            self.prev_error = 0.0
            if self.r_right < OR_VALUE:
                self.state = "TRACKING_WALL"
            elif self.r_front < SAFETY_THRES:
                self.v, self.omega = 0.0, -0.30
            else:
                self.v, self.omega = V_NORMAL, 0.0
            return

        # REACTIVE WALL FOLLOWER
        if self.wf_state == "TURN_CONCAVE":
            if self.r_front < SAFETY_THRES + 5:
                self.v, self.omega = 0.0, -0.30
                return
            else:
                self.wf_state = "TRACKING_WALL"
                self.prev_error = 0.0

        if self.r_front < SAFETY_THRES:
            self.wf_state = "TURN_CONCAVE"
            self.v, self.omega, self._lost_timer, self.prev_error = 0.0, -0.30, 0, 0.0

        elif self.r_right == OR_VALUE:
            self._lost_timer += 1
            if self._lost_timer < 12:
                self.wf_state = "CLEARING_CORNER"
                self.v, self.omega, self.prev_error = V_NORMAL, 0.0, 0.0
            else:
                self.wf_state = "TURN_CONVEX"
                self.v, self.omega, self.prev_error = V_NORMAL * 0.2, 0.30, 0.0

        elif self.r_right > current_d_ref + 10:
            self.wf_state = "ADJUST_RIGHT"
            self.v, self.omega, self._lost_timer, self.prev_error = (
                V_NORMAL * 0.8,
                0.15,
                0,
                0.0,
            )

        else:
            self.wf_state = "TRACKING_WALL"
            self.v = V_NORMAL
            error = self.r_right - current_d_ref
            derivative = error - self.prev_error
            self.omega = max(-0.15, min(0.15, (KP * error) + (KD * derivative)))
            self.prev_error = error
            self._lost_timer = 0

        self.state = self.wf_state

        if self.wf_state == "TURN_CONVEX" and self._lost_timer > 120:
            self.state = "FIND_WALL"

        if self.state == "TRACKING_WALL" and len(self.current_layer_path) > 60:
            for i in range(len(self.current_layer_path) - 45):
                if self.pos.distance_to(self.current_layer_path[i]) < 15:
                    loop_pts = self.current_layer_path[i:]
                    min_x, max_x = (
                        min(p.x for p in loop_pts),
                        max(p.x for p in loop_pts),
                    )
                    min_y, max_y = (
                        min(p.y for p in loop_pts),
                        max(p.y for p in loop_pts),
                    )

                    simplified_line = LineString(
                        [(p.x, p.y) for p in loop_pts]
                    ).simplify(3.0)
                    coords = list(simplified_line.coords)
                    new_segments = [
                        (Vector2(coords[j]), Vector2(coords[j + 1]))
                        for j in range(len(coords) - 1)
                    ]
                    if len(coords) > 2:
                        new_segments.append((Vector2(coords[-1]), Vector2(coords[0])))
                    self.virtual_walls.extend(new_segments)

                    if (max_x - min_x) < 80 or (max_y - min_y) < 80:
                        logger.info(
                            "Local pocket complete! Searching memory for valid frontiers..."
                        )
                        targets = self.get_valid_frontiers()

                        for target in targets:
                            waypoints = self.plan_path_through_memory(self.pos, target)
                            if waypoints:
                                self.waypoints = waypoints
                                self.state = "TRANSIT_TO_FRONTIER"
                                self.current_waypoint_idx = 0
                                self.current_layer_path = []
                                return

                        logger.info(
                            "No reachable unpainted space left. Coverage Complete!"
                        )
                        self.state = "FINISHED"
                        self.v, self.omega = 0.0, 0.0
                        return

                    logger.info(f"Layer closed! Solidifying virtual walls.")
                    left_vec = Vector2(
                        math.cos(self.heading_angle - math.pi / 2),
                        math.sin(self.heading_angle - math.pi / 2),
                    )
                    intended_correction = self.pos + left_vec * 5.0
                    gx, gy = self.get_grid_coords(intended_correction)
                    if self.internal_map.get((gx, gy)) != "WALL":
                        self.pos = intended_correction

                    self.current_layer_path = []
                    return


def main():
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Bumper Escape Reflex")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 18)

    coverage_surface = pygame.Surface(WINDOW_SIZE, pygame.SRCALPHA)
    memory_surface = pygame.Surface(WINDOW_SIZE, pygame.SRCALPHA)

    boundary_pts = [
        Vector2(100, 100),
        Vector2(900, 100),
        Vector2(900, 700),
        Vector2(550, 700),
        Vector2(550, 450),
        Vector2(450, 450),
        Vector2(450, 700),
        Vector2(100, 700),
    ]
    obstacles = [
        [Vector2(250, 200), Vector2(400, 200), Vector2(400, 350), Vector2(250, 350)],
        [Vector2(650, 300), Vector2(800, 350), Vector2(750, 550), Vector2(600, 500)],
    ]

    boundary_segments = [
        (boundary_pts[i], boundary_pts[(i + 1) % len(boundary_pts)])
        for i in range(len(boundary_pts))
    ]
    obstacle_segments = [
        (obs[i], obs[(i + 1) % len(obs)]) for obs in obstacles for i in range(len(obs))
    ]

    robot = Robot(150, 200, -90, boundary_pts)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        robot.step(1.0, boundary_segments, obstacle_segments)

        pygame.draw.circle(
            coverage_surface,
            COLOR_PAINT,
            (int(robot.pos.x), int(robot.pos.y)),
            robot.radius,
        )

        screen.fill(COLOR_BG)
        screen.blit(coverage_surface, (0, 0))

        memory_surface.fill((0, 0, 0, 0))
        for (gx, gy), state in robot.internal_map.items():
            px, py = gx * GRID_SIZE, gy * GRID_SIZE
            if state == "FREE":
                pygame.draw.circle(
                    memory_surface,
                    (255, 255, 255, 100),
                    (px + GRID_SIZE // 2, py + GRID_SIZE // 2),
                    2,
                )
            elif state == "COVERED":
                pygame.draw.rect(
                    memory_surface, (52, 152, 219, 40), (px, py, GRID_SIZE, GRID_SIZE)
                )
        screen.blit(memory_surface, (0, 0))

        if robot.state == "TRANSIT_TO_FRONTIER" and len(robot.waypoints) > 1:
            pygame.draw.lines(screen, (231, 76, 60), False, robot.waypoints, 2)
            for wp in robot.waypoints:
                pygame.draw.circle(screen, (231, 76, 60), (int(wp.x), int(wp.y)), 3)

        if len(robot.path_actual) > 1:
            pygame.draw.lines(screen, COLOR_PATH_ACTUAL, False, robot.path_actual, 2)

        pygame.draw.polygon(screen, COLOR_BOUNDARY, boundary_pts, 4)
        for obs in obstacles:
            pygame.draw.polygon(screen, COLOR_OBSTACLE, obs, 3)
            s = pygame.Surface(WINDOW_SIZE, pygame.SRCALPHA)
            pygame.draw.polygon(s, (231, 76, 60, 60), obs)
            screen.blit(s, (0, 0))

        if len(robot.virtual_walls) > 0:
            for p1, p2 in robot.virtual_walls:
                pygame.draw.line(screen, COLOR_VIRTUAL_WALL, p1, p2, 2)

        pygame.draw.circle(
            screen, COLOR_ROBOT, (int(robot.pos.x), int(robot.pos.y)), robot.radius
        )

        h_vec = Vector2(math.cos(robot.heading_angle), math.sin(robot.heading_angle))

        for vec, dist in [
            (h_vec, robot.r_front),
            (h_vec.rotate(-90), robot.r_left),
            (h_vec.rotate(90), robot.r_right),
        ]:
            pygame.draw.line(
                screen,
                COLOR_RAY_HIT if dist < OR_VALUE else COLOR_RAY_OR,
                robot.pos,
                robot.pos + vec * dist,
                1,
            )

        for i, text in enumerate(
            [
                f"LAYER: {robot.phase}",
                f"STATE: {robot.state}",
                f"WF: {robot.wf_state}",
                f"V_WALLS: {len(robot.virtual_walls)}",
            ]
        ):
            screen.blit(font.render(text, True, COLOR_TEXT), (20, 20 + i * 25))

        pygame.display.flip()
        clock.tick(FPS)
    pygame.quit()


if __name__ == "__main__":
    main()
