"""
Standalone 2D Robot Simulation in Pygame-ce.
Upgraded with Slower Speeds and Zero-Radius "Point-and-Shoot" Pivoting.
"""

import heapq
import logging
import math

import pygame
from pygame.math import Vector2
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

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
COLOR_PATH_PLANNED = (127, 140, 141)
COLOR_PATH_ACTUAL = (241, 196, 15)
COLOR_TEXT = (236, 240, 241)
COLOR_PAINT = (52, 152, 219, 30)

ROBOT_RADIUS = 15
D_REF = 25.0
OR_VALUE = 50.0

# SPEED ADJUSTMENTS
V_NORMAL = 1.2  # Reduced for realism and sensor reliability
SAFETY_THRES = 35.0

# PD Controller Tuning
KP = 0.015
KD = 0.05

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


# --- BLIND PATHFINDING (SHAPELY) ---


def astar_safe_path(start_pos, goal_pos, safe_poly, step=15):
    def round_pt(v):
        return (int(round(v.x / step) * step), int(round(v.y / step) * step))

    start, goal = round_pt(start_pos), round_pt(goal_pos)
    if start == goal:
        return [goal_pos]

    frontier = [(0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}

    while frontier:
        _, curr = heapq.heappop(frontier)
        if curr == goal or math.dist(curr, goal) < step * 1.5:
            goal = curr
            break

        for dx, dy in [
            (0, step),
            (0, -step),
            (step, 0),
            (-step, 0),
            (step, step),
            (-step, -step),
            (step, -step),
            (-step, step),
        ]:
            nxt = (curr[0] + dx, curr[1] + dy)
            if not Point(nxt).within(safe_poly):
                continue
            new_cost = cost_so_far[curr] + math.dist(curr, nxt)
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                heapq.heappush(frontier, (new_cost + math.dist(nxt, goal), nxt))
                came_from[nxt] = curr

    path = []
    curr = goal
    while curr != start and curr in came_from:
        path.append(Vector2(curr[0], curr[1]))
        curr = came_from[curr]
    path.reverse()
    path.append(goal_pos)
    return path


def generate_coverage_path(boundary_pts, radius, start_pos):
    poly_bound = Polygon([(p.x, p.y) for p in boundary_pts])
    safe_poly = poly_bound.buffer(-(radius + 2.0))

    if safe_poly.is_empty:
        return []
    if safe_poly.geom_type == "MultiPolygon":
        start_pt = Point(start_pos.x, start_pos.y)
        for geom in safe_poly.geoms:
            if geom.distance(start_pt) < 10:
                safe_poly = geom
                break

    minx, miny, maxx, maxy = safe_poly.bounds
    spacing = radius * 2.0
    y = miny + radius

    sweep_segments = []
    while y <= maxy:
        line = LineString([(minx - 10, y), (maxx + 10, y)])
        intersected = safe_poly.intersection(line)
        if not intersected.is_empty:
            if intersected.geom_type == "LineString":
                sweep_segments.append(intersected)
            elif intersected.geom_type == "MultiLineString":
                for geom in intersected.geoms:
                    sweep_segments.append(geom)
        y += spacing

    waypoints = []
    curr_pos = Vector2(start_pos)

    while sweep_segments:
        best_dist = float("inf")
        best_idx = -1
        go_to_start = True

        for i, geom in enumerate(sweep_segments):
            p1, p2 = Vector2(geom.coords[0]), Vector2(geom.coords[-1])
            if curr_pos.distance_to(p1) < best_dist:
                best_dist = curr_pos.distance_to(p1)
                best_idx = i
                go_to_start = True
            if curr_pos.distance_to(p2) < best_dist:
                best_dist = curr_pos.distance_to(p2)
                best_idx = i
                go_to_start = False

        geom = sweep_segments.pop(best_idx)
        p1, p2 = Vector2(geom.coords[0]), Vector2(geom.coords[-1])
        target_start = p1 if go_to_start else p2
        target_end = p2 if go_to_start else p1

        if LineString([curr_pos, target_start]).within(safe_poly):
            waypoints.append(target_start)
        else:
            waypoints.extend(astar_safe_path(curr_pos, target_start, safe_poly))

        waypoints.append(target_end)
        curr_pos = target_end

    return waypoints


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
        self.r_front_obs = OR_VALUE

        self.path_actual = [Vector2(self.pos)]
        self.waypoints = []
        self.current_waypoint_idx = 0

        self.avoid_start_dist = 0
        self.avoid_path = []

        self.v = 0.0
        self.omega = 0.0
        self._lost_timer = 0
        self.prev_error = 0.0

    def update_sensors(self, boundary_segments, obstacle_segments):
        h_vec = Vector2(math.cos(self.heading_angle), math.sin(self.heading_angle))
        all_segs = boundary_segments + obstacle_segments

        self.r_front = get_distance_to_geometry(self.pos, h_vec, all_segs, OR_VALUE)
        self.r_left = get_distance_to_geometry(
            self.pos, h_vec.rotate(-90), all_segs, OR_VALUE
        )
        self.r_right = get_distance_to_geometry(
            self.pos, h_vec.rotate(90), all_segs, OR_VALUE
        )
        self.r_front_obs = get_distance_to_geometry(
            self.pos, h_vec, obstacle_segments, OR_VALUE
        )

    def step(self, dt, boundary_segments, obstacle_segments):
        self.update_sensors(boundary_segments, obstacle_segments)

        if self.phase == 1:
            self.logic_phase1()
        else:
            self.logic_phase2()

        self.heading_angle += self.omega * dt
        self.pos += (
            Vector2(math.cos(self.heading_angle), math.sin(self.heading_angle))
            * self.v
            * dt
        )

        if self.pos.distance_to(self.path_actual[-1]) > 5:
            self.path_actual.append(Vector2(self.pos))

    def do_reactive_wall_following(self):
        """Shared logic: Wall Tracking with Pivoting and Clamped PD Controller"""

        # Hysteresis Lock for Left Turns
        if self.wf_state == "TURN_CONCAVE":
            if self.r_front < SAFETY_THRES + 10:
                self.v = 0.0
                self.omega = -0.30  # Faster Pivot Left
                return
            else:
                self.wf_state = "TRACKING_WALL"
                self.prev_error = 0.0

        # Standard Reactive Checks
        if self.r_front < SAFETY_THRES:
            self.wf_state = "TURN_CONCAVE"
            self.v = 0.0
            self.omega = -0.30  # Pivot Left
            self._lost_timer = 0
            self.prev_error = 0.0

        elif self.r_right == OR_VALUE:
            self.wf_state = "TURN_CONVEX"
            # Tight arc around right-hand corners
            self.v = V_NORMAL * 0.2
            self.omega = 0.30
            self._lost_timer += 1
            self.prev_error = 0.0

        elif self.r_right > D_REF + 10:
            self.wf_state = "ADJUST_RIGHT"
            self.v = V_NORMAL * 0.8
            self.omega = 0.15
            self._lost_timer = 0
            self.prev_error = 0.0

        else:
            self.wf_state = "TRACKING_WALL"
            self.v = V_NORMAL

            error = self.r_right - D_REF
            derivative = error - self.prev_error
            raw_omega = (KP * error) + (KD * derivative)

            # Smooth steering clamp
            self.omega = max(-0.15, min(0.15, raw_omega))

            self.prev_error = error
            self._lost_timer = 0

    def logic_phase1(self):
        if self.state == "FIND_WALL":
            self._lost_timer = 0
            self.prev_error = 0.0
            if self.r_right < OR_VALUE:
                self.state = "TRACKING_WALL"
            elif self.r_front < SAFETY_THRES:
                self.v, self.omega = 0.0, -0.30  # Pivot Left
            else:
                self.v, self.omega = V_NORMAL, 0.0
            return

        self.do_reactive_wall_following()
        self.state = self.wf_state

        if self.wf_state == "TURN_CONVEX" and self._lost_timer > 120:
            self.state = "FIND_WALL"

        if self.state == "TRACKING_WALL" and len(self.path_actual) > 300:
            for i in range(len(self.path_actual) - 150):
                if self.pos.distance_to(self.path_actual[i]) < 10:
                    logger.info("Map closed! Generating blind path...")
                    self.phase = 2
                    self.state = "FOLLOWING_PATH"
                    self.waypoints = generate_coverage_path(
                        self.map_boundary, self.radius, self.pos
                    )
                    return

    def logic_phase2(self):
        if self.current_waypoint_idx >= len(self.waypoints):
            self.v, self.omega, self.state = 0.0, 0.0, "FINISHED"
            return

        target = self.waypoints[self.current_waypoint_idx]
        dist_to_target = self.pos.distance_to(target)

        if dist_to_target < 15:
            self.current_waypoint_idx += 1
            return

        if self.state == "FOLLOWING_PATH":
            desired_vec = target - self.pos
            desired_angle = math.atan2(desired_vec.y, desired_vec.x)
            angle_diff = (desired_angle - self.heading_angle + math.pi) % (
                2 * math.pi
            ) - math.pi

            # ZERO-RADIUS NAVIGATION ("Point and Shoot")
            # If off target by more than ~10 degrees, pivot in place!
            if abs(angle_diff) > math.radians(10):
                self.v = 0.0
                self.omega = math.copysign(0.25, angle_diff)
            else:
                # Drive straight, applying micro-corrections
                self.v = V_NORMAL
                self.omega = max(-0.1, min(0.1, 0.5 * angle_diff))

            if self.r_front_obs < SAFETY_THRES:
                logger.info(f"Obstacle encountered! Switching to Bug2 Avoidance.")
                self.state = "AVOID_OBSTACLE"
                self.avoid_start_dist = dist_to_target
                self.avoid_path = []
                self.prev_error = 0.0

        elif self.state == "AVOID_OBSTACLE":
            self.do_reactive_wall_following()

            if (
                len(self.avoid_path) == 0
                or self.pos.distance_to(self.avoid_path[-1]) > 5
            ):
                self.avoid_path.append(Vector2(self.pos))

            if len(self.avoid_path) > 100:
                for i in range(len(self.avoid_path) - 50):
                    if self.pos.distance_to(self.avoid_path[i]) < 10:
                        logger.warning(
                            "Looped around obstacle! Waypoint unreachable. Skipping."
                        )
                        self.current_waypoint_idx += 1
                        self.state = "FOLLOWING_PATH"
                        return

            if (
                dist_to_target < self.avoid_start_dist - 15
                and self.r_front_obs > SAFETY_THRES + 10
            ):
                logger.info("Cleared obstacle! Resuming path.")
                self.state = "FOLLOWING_PATH"


def main():
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Point-and-Shoot Bug2 Navigation")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 18)

    coverage_surface = pygame.Surface(WINDOW_SIZE, pygame.SRCALPHA)

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

        if robot.phase == 2 and len(robot.waypoints) > 1:
            pygame.draw.lines(screen, COLOR_PATH_PLANNED, False, robot.waypoints, 2)
            for wp in robot.waypoints:
                pygame.draw.circle(screen, (155, 89, 182), (int(wp.x), int(wp.y)), 3)

        if len(robot.path_actual) > 1:
            pygame.draw.lines(screen, COLOR_PATH_ACTUAL, False, robot.path_actual, 2)

        pygame.draw.polygon(screen, COLOR_BOUNDARY, boundary_pts, 4)
        for obs in obstacles:
            pygame.draw.polygon(screen, COLOR_OBSTACLE, obs, 3)
            s = pygame.Surface(WINDOW_SIZE, pygame.SRCALPHA)
            pygame.draw.polygon(s, (231, 76, 60, 60), obs)
            screen.blit(s, (0, 0))

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
            [f"PHASE: {robot.phase}", f"STATE: {robot.state}", f"WF: {robot.wf_state}"]
        ):
            screen.blit(font.render(text, True, COLOR_TEXT), (20, 20 + i * 25))

        pygame.display.flip()
        clock.tick(FPS)
    pygame.quit()


if __name__ == "__main__":
    main()
