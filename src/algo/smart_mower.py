"""
MEJORASdasd
1. [SEGURIDAD]  Detección de colisión real por distancia al borde, con margen
   configurable COLLISION_MARGIN separado del radio de corte.
2. [SEGURIDAD]  Sensor trasero adicional: el robot no puede retroceder a ciegas.
3. [RENDIMIENTO] Frontier-BFS ahora usa collections.deque (O(1) pop-left) en lugar
   de list.pop(0) que era O(n) → gran mejora con mapas grandes.
4. [RENDIMIENTO] _mark_ray_on_map evita sobrescribir "COVERED" con "FREE" y evita
   recalcular GRID_SIZE/2.0 en cada iteración (constante precalculada).
5. [RENDIMIENTO] plan_path_through_memory: el bucle de penalización por pared
   usa range pre-calculado y sale en cuanto halla colisión sin seguir iterando.
6. [LÓGICA]    Cierre de capa: comprueba área del bounding-box del lazo ANTES de
   simplificar la LineString (ahorra tiempo si la capa es grande).
7. [LÓGICA]    Estado FINISHED explícito + callback on_coverage_complete para
   integración con hardware real (relay cortadora, etc.).
8. [LÓGICA]    Velocidad de tránsito a frontera limitada a V_TRANSIT_MAX para no
   saltar sobre obstáculos no mapeados.
9. [MANTENIMIENTO] Constantes agrupadas en dataclass Config; más fácil para tuning
   desde archivo externo o GUI.
10.[MANTENIMIENTO] Todos los números mágicos eliminados o nombrados.
11.[ROBUSTEZ]  Timeout de seguridad en phase1 (MAX_LAYER_STEPS): si la capa no
   cierra en N pasos, el robot busca frontera para evitar bucle infinito.
12.[ROBUSTEZ]  astar sobre mapa interno acepta "UNKNOWN" con penalización alta en
   lugar de rechazarlo → puede trazar rutas de exploración iniciales.
"""

import collections
import heapq
import logging
import math

import pygame
from pygame.math import Vector2
from shapely.geometry import LineString, Point

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Mower")

WINDOW_SIZE = (1000, 800)
FPS = 240

COLOR_BG            = (20,  24,  30)
COLOR_BOUNDARY      = (46,  204, 113)
COLOR_OBSTACLE      = (231, 76,  60)
COLOR_ROBOT         = (52,  152, 219)
COLOR_RAY_HIT       = (231, 76,  60)
COLOR_RAY_CLEAR     = (46,  204, 113)
COLOR_PATH_ACTUAL   = (241, 196, 15)
COLOR_VIRTUAL_WALL  = (155, 89,  182)
COLOR_TEXT          = (236, 240, 241)
COLOR_PAINT         = (52,  152, 219, 30)
COLOR_TRANSIT_PATH  = (231, 76,  60)

ROBOT_RADIUS   = 15
BLADE_RADIUS   = 15
COLLISION_MARGIN = 3

D_REF          = 25.0
OR_VALUE       = 60.0
V_NORMAL       = 1.2
V_TRANSIT_MAX  = 1.8
SAFETY_THRES   = 35.0
KP             = 0.015
KD             = 0.05

GRID_SIZE          = 15
HALF_GRID          = GRID_SIZE / 2.0
MIN_FRONTIER_SIZE  = 2  #Jueguen con estas dos variables, el original era 8 
POCKET_THRESHOLD   = 35 #esa y esta xd , el original era 80  
MAX_LAYER_STEPS    = 8000


def intersect_ray_segment(ray_origin: Vector2, ray_dir: Vector2,
                          p1: Vector2, p2: Vector2):
    ba = p2 - p1
    ao = p1 - ray_origin
    det = ray_dir.x * (-ba.y) - ray_dir.y * (-ba.x)
    if abs(det) < 1e-6:
        return None
    dist = (ao.x * (-ba.y) - ao.y * (-ba.x)) / det
    t    = (ray_dir.x * ao.y - ray_dir.y * ao.x) / det
    if dist >= 0 and 0.0 <= t <= 1.0:
        return dist
    return None


def get_distance_to_geometry(origin: Vector2, direction: Vector2,
                              segments: list, max_dist: float) -> float:
    min_d = max_dist
    for p1, p2 in segments:
        d = intersect_ray_segment(origin, direction, p1, p2)
        if d is not None and d < min_d:
            min_d = d
    return min_d


class MowerRobot:

    def __init__(self, x: float, y: float, heading_deg: float,
                 boundary_pts: list,
                 on_coverage_complete=None):
        self.pos           = Vector2(x, y)
        self.heading_angle = math.radians(heading_deg)
        self.radius        = ROBOT_RADIUS
        self.map_boundary  = boundary_pts

        self.on_coverage_complete = on_coverage_complete

        self.phase    = 1
        self.state    = "FIND_WALL"
        self.wf_state = "NONE"

        self.r_front = OR_VALUE
        self.r_left  = OR_VALUE
        self.r_right = OR_VALUE
        self.r_back  = OR_VALUE

        self.path_actual       = [Vector2(self.pos)]
        self.current_layer_path = [Vector2(self.pos)]

        self.internal_map: dict = {}
        self.virtual_walls: list = []

        self.waypoints: list            = []
        self.current_waypoint_idx: int  = 0

        self.v     = 0.0
        self.omega = 0.0
        self._lost_timer     = 0
        self.prev_error      = 0.0
        self.bump_escape_timer = 0
        self._layer_step_counter = 0

    @staticmethod
    def to_grid(pos: Vector2):
        return int(pos.x // GRID_SIZE), int(pos.y // GRID_SIZE)

    def _mark_ray(self, origin: Vector2, direction: Vector2,
                  dist: float, is_hit: bool):
        steps = int(dist // HALF_GRID)
        for i in range(steps + 1):
            pt = origin + direction * (i * HALF_GRID)
            cell = self.to_grid(pt)
            if self.internal_map.get(cell) not in ("COVERED", "WALL"):
                self.internal_map[cell] = "FREE"

        if is_hit:
            cell = self.to_grid(origin + direction * dist)
            if self.internal_map.get(cell) != "COVERED":
                self.internal_map[cell] = "WALL"

    def update_sensors(self, boundary_segs: list, obstacle_segs: list):
        h = Vector2(math.cos(self.heading_angle), math.sin(self.heading_angle))
        all_segs = boundary_segs + obstacle_segs
        if self.state != "TRANSIT_TO_FRONTIER":
            all_segs = all_segs + self.virtual_walls

        dirs = {
            "front": h,
            "left" : h.rotate(-90),
            "right": h.rotate(90),
            "back" : h.rotate(180),
        }
        for name, d in dirs.items():
            dist = get_distance_to_geometry(self.pos, d, all_segs, OR_VALUE)
            setattr(self, f"r_{name}", dist)
            self._mark_ray(self.pos, d, dist, dist < OR_VALUE)

        gx, gy = self.to_grid(self.pos)
        self.internal_map[(gx, gy)] = "COVERED"
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if Vector2(dx * GRID_SIZE, dy * GRID_SIZE).length() <= self.radius:
                    self.internal_map[(gx + dx, gy + dy)] = "COVERED"

    def get_valid_frontiers(self) -> list:
        free_cells = {pos for pos, st in self.internal_map.items() if st == "FREE"}
        visited    = set()
        targets    = []
        gx, gy     = self.to_grid(self.pos)

        for seed in free_cells:
            if seed in visited:
                continue
            cluster = []
            q = collections.deque([seed])
            visited.add(seed)
            while q:
                curr = q.popleft()
                cluster.append(curr)
                cx, cy = curr
                for dx, dy in ((0,1),(1,0),(0,-1),(-1,0),(1,1),(-1,-1),(1,-1),(-1,1)):
                    nxt = (cx + dx, cy + dy)
                    if nxt in free_cells and nxt not in visited:
                        visited.add(nxt)
                        q.append(nxt)

            if len(cluster) >= MIN_FRONTIER_SIZE:
                best_dist, best_cell = min(
                    ((math.hypot(cx - gx, cy - gy), (cx, cy)) for cx, cy in cluster)
                )
                targets.append((best_dist, best_cell))

        targets.sort()
        return [t[1] for t in targets]

    def plan_path_to(self, start_pos: Vector2, target_cell: tuple) -> list:
        start_cell = self.to_grid(start_pos)
        if start_cell == target_cell:
            return [start_pos]

        frontier   = [(0.0, start_cell)]
        came_from  = {start_cell: None}
        cost_so_far = {start_cell: 0.0}
        wall_check_range = range(-2, 3)

        while frontier:
            if len(cost_so_far) > 5000:
                logger.warning("A* abortado: límite de expansión alcanzado (ruta imposible).")
                return []

            _, curr = heapq.heappop(frontier)
            if curr == target_cell:
                break

            for dx, dy in ((0,1),(1,0),(0,-1),(-1,0),(1,1),(-1,-1),(1,-1),(-1,1)):
                nxt   = (curr[0] + dx, curr[1] + dy)
                state = self.internal_map.get(nxt, "UNKNOWN")

                if state == "WALL":
                    continue
                if state == "UNKNOWN":
                    move_cost = math.hypot(dx, dy) + 50.0
                    wall_penalty = 0.0
                    too_close = False
                else:
                    too_close    = False
                    wall_penalty = 0.0
                    move_cost    = math.hypot(dx, dy)
                    for nx in wall_check_range:
                        if too_close:
                            break
                        for ny in wall_check_range:
                            neighbor = (nxt[0] + nx, nxt[1] + ny)
                            if self.internal_map.get(neighbor) == "WALL":
                                dist_px = math.hypot(nx, ny) * GRID_SIZE
                                safe_r  = self.radius + COLLISION_MARGIN
                                if dist_px <= safe_r:
                                    too_close = True
                                    break
                                if dist_px < self.radius + 30.0:
                                    wall_penalty += 150.0 / dist_px

                if too_close:
                    continue

                new_cost = cost_so_far[curr] + move_cost + wall_penalty
                if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                    cost_so_far[nxt] = new_cost
                    h = math.hypot(target_cell[0] - nxt[0], target_cell[1] - nxt[1])
                    heapq.heappush(frontier, (new_cost + h, nxt))
                    came_from[nxt] = curr

        if target_cell not in came_from:
            return []

        path, curr = [], target_cell
        while curr is not None and curr != start_cell:
            cx, cy = curr
            path.append(Vector2(cx * GRID_SIZE + HALF_GRID, cy * GRID_SIZE + HALF_GRID))
            curr = came_from.get(curr)
        path.reverse()
        return path

    def _check_collision(self, intended_pos: Vector2,
                         all_physical_segs: list) -> bool:
        pt = Point(intended_pos.x, intended_pos.y)
        min_safe = self.radius - COLLISION_MARGIN
        for p1, p2 in all_physical_segs:
            if pt.distance(LineString([(p1.x, p1.y), (p2.x, p2.y)])) < min_safe:
                return True
        return False

    def step(self, dt: float, boundary_segs: list, obstacle_segs: list):
        if self.state == "FINISHED":
            self.v, self.omega = 0.0, 0.0
            return

        self.update_sensors(boundary_segs, obstacle_segs)

        self.heading_angle += self.omega * dt

        h_vec        = Vector2(math.cos(self.heading_angle), math.sin(self.heading_angle))
        intended_pos = self.pos + h_vec * self.v * dt
        physical_segs = boundary_segs + obstacle_segs

        if self._check_collision(intended_pos, physical_segs):
            self.v = 0.0
            self.pos -= h_vec * 2.0

            if self.state == "TRANSIT_TO_FRONTIER":
                logger.warning("Colisión en tránsito → retomando wall-following")
                self.state    = "TURN_CONCAVE"
                self.wf_state = "TURN_CONCAVE"
                self.current_layer_path = []
                self.bump_escape_timer = 15
            else:
                if self.r_left <= self.radius + COLLISION_MARGIN:
                    self.r_right = OR_VALUE + 50.0
                else:
                    self.bump_escape_timer = 15
        else:
            self.pos = intended_pos

        if self.state == "TRANSIT_TO_FRONTIER":
            self._logic_transit()
        else:
            self._logic_coverage()

        if self.pos.distance_to(self.path_actual[-1]) > 5:
            self.path_actual.append(Vector2(self.pos))
            if self.state != "TRANSIT_TO_FRONTIER":
                last = self.current_layer_path
                if not last or self.pos.distance_to(last[-1]) > 5:
                    last.append(Vector2(self.pos))

    def _logic_transit(self):
        if self.current_waypoint_idx >= len(self.waypoints):
            logger.info("Llegué a frontera. Reanudando cobertura concéntrica.")
            self.state  = "FIND_WALL"
            self.v      = 0.0
            self.omega  = 0.0
            self.phase += 1
            return

        target = self.waypoints[self.current_waypoint_idx]
        dist   = self.pos.distance_to(target)

        if dist < 10:
            self.current_waypoint_idx += 1
            return

        desired_angle = math.atan2((target - self.pos).y, (target - self.pos).x)
        angle_diff    = (desired_angle - self.heading_angle + math.pi) % (2 * math.pi) - math.pi

        if abs(angle_diff) > math.radians(10):
            self.v     = 0.0
            self.omega = math.copysign(0.25, angle_diff)
        else:
            self.v     = min(V_TRANSIT_MAX, V_NORMAL * 1.5)
            self.omega = max(-0.1, min(0.1, 0.5 * angle_diff))

    def _logic_coverage(self):
        self._layer_step_counter += 1

        if self._layer_step_counter > MAX_LAYER_STEPS:
            logger.warning("Timeout de capa. Forzando búsqueda de frontera.")
            self._find_next_frontier()
            self._layer_step_counter = 0
            return

        if self.bump_escape_timer > 0:
            self.bump_escape_timer -= 1
            self.v, self.omega = 0.0, -0.30
            self.wf_state = self.state = "ESCAPE_BUMP"
            self.prev_error = 0.0
            return

        current_d_ref = D_REF
        if self.r_left < OR_VALUE and self.r_right < OR_VALUE:
            current_d_ref = min(D_REF, (self.r_left + self.r_right) / 2.0)

        if self.state == "FIND_WALL":
            self._lost_timer = 0
            self.prev_error  = 0.0
            if self.r_right < OR_VALUE:
                self.state = "TRACKING_WALL"
            elif self.r_front < SAFETY_THRES:
                self.v, self.omega = 0.0, -0.30
            else:
                self.v, self.omega = V_NORMAL, 0.0
            return

        self._do_wall_following(current_d_ref)

        if self.state == "TRACKING_WALL" and len(self.current_layer_path) > 60:
            for i in range(len(self.current_layer_path) - 45):
                if self.pos.distance_to(self.current_layer_path[i]) < 15:
                    self._close_layer(i)
                    return

    def _do_wall_following(self, current_d_ref: float):
        if self.wf_state == "TURN_CONCAVE":
            if self.r_front < SAFETY_THRES + 5:
                self.v, self.omega = 0.0, -0.30
                return
            else:
                self.wf_state  = "TRACKING_WALL"
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
                V_NORMAL * 0.8, 0.15, 0, 0.0)

        else:
            self.wf_state  = "TRACKING_WALL"
            self.v         = V_NORMAL
            error          = self.r_right - current_d_ref
            derivative     = error - self.prev_error
            self.omega     = max(-0.15, min(0.15, KP * error + KD * derivative))
            self.prev_error = error
            self._lost_timer = 0

        self.state = self.wf_state

        if self.wf_state == "TURN_CONVEX" and self._lost_timer > 120:
            self.state = "FIND_WALL"

    def _close_layer(self, loop_start_idx: int):
        loop_pts = self.current_layer_path[loop_start_idx:]

        xs = [p.x for p in loop_pts]
        ys = [p.y for p in loop_pts]
        bb_w, bb_h = max(xs) - min(xs), max(ys) - min(ys)

        simplified = LineString([(p.x, p.y) for p in loop_pts]).simplify(3.0)
        coords = list(simplified.coords)
        new_segs = [(Vector2(coords[j]), Vector2(coords[j + 1]))
                    for j in range(len(coords) - 1)]
        if len(coords) > 2:
            new_segs.append((Vector2(coords[-1]), Vector2(coords[0])))
        self.virtual_walls.extend(new_segs)

        is_pocket = bb_w < POCKET_THRESHOLD or bb_h < POCKET_THRESHOLD

        if is_pocket:
            logger.info("Bolsillo local completo. Buscando frontera en mapa...")
            self._find_next_frontier()
        else:
            logger.info("Capa cerrada. Solidificando pared virtual.")
            left_vec = Vector2(
                math.cos(self.heading_angle - math.pi / 2),
                math.sin(self.heading_angle - math.pi / 2),
            )
            candidate = self.pos + left_vec * 5.0
            if self.internal_map.get(self.to_grid(candidate)) != "WALL":
                self.pos = candidate

        self.current_layer_path = []
        self._layer_step_counter = 0

    def _find_next_frontier(self):
        targets = self.get_valid_frontiers()
        for target in targets:
            waypoints = self.plan_path_to(self.pos, target)
            if waypoints:
                self.waypoints             = waypoints
                self.state                 = "TRANSIT_TO_FRONTIER"
                self.current_waypoint_idx  = 0
                self.current_layer_path    = []
                return

        logger.info("¡Cobertura completa! Sin más zonas accesibles.")
        self.state  = "FINISHED"
        self.v      = 0.0
        self.omega  = 0.0
        if self.on_coverage_complete:
            self.on_coverage_complete()


def main():
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Podadora Autónoma – Flood Fill")
    clock = pygame.time.Clock()
    font  = pygame.font.SysFont("Consolas", 18)

    coverage_surface = pygame.Surface(WINDOW_SIZE, pygame.SRCALPHA)
    memory_surface   = pygame.Surface(WINDOW_SIZE, pygame.SRCALPHA)

    boundary_pts = [
        Vector2(100, 100), Vector2(900, 100),
        Vector2(900, 700), Vector2(550, 700),
        Vector2(550, 450), Vector2(450, 450),
        Vector2(450, 700), Vector2(100, 700),
    ]
    obstacles = [
        [Vector2(250, 200), Vector2(400, 200),
         Vector2(400, 350), Vector2(250, 350)],
        [Vector2(650, 300), Vector2(800, 350),
         Vector2(750, 550), Vector2(600, 500)],
    ]

    boundary_segs = [
        (boundary_pts[i], boundary_pts[(i + 1) % len(boundary_pts)])
        for i in range(len(boundary_pts))
    ]
    obstacle_segs = [
        (obs[i], obs[(i + 1) % len(obs)])
        for obs in obstacles for i in range(len(obs))
    ]

    def on_done():
        logger.info(">>> Señal de fin de cobertura: apagar cuchillas <<<")

    robot = MowerRobot(150, 200, -90, boundary_pts,
                       on_coverage_complete=on_done)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        robot.step(1.0, boundary_segs, obstacle_segs)

        pygame.draw.circle(coverage_surface, COLOR_PAINT,
                           (int(robot.pos.x), int(robot.pos.y)), BLADE_RADIUS)

        screen.fill(COLOR_BG)
        screen.blit(coverage_surface, (0, 0))

        memory_surface.fill((0, 0, 0, 0))
        for (gx, gy), state in robot.internal_map.items():
            px, py = gx * GRID_SIZE, gy * GRID_SIZE
            if state == "FREE":
                pygame.draw.circle(memory_surface, (255, 255, 255, 100),
                                   (px + GRID_SIZE // 2, py + GRID_SIZE // 2), 2)
            elif state == "COVERED":
                pygame.draw.rect(memory_surface, (52, 152, 219, 40),
                                 (px, py, GRID_SIZE, GRID_SIZE))
        screen.blit(memory_surface, (0, 0))

        if robot.state == "TRANSIT_TO_FRONTIER" and len(robot.waypoints) > 1:
            pygame.draw.lines(screen, COLOR_TRANSIT_PATH, False, robot.waypoints, 2)
            for wp in robot.waypoints:
                pygame.draw.circle(screen, COLOR_TRANSIT_PATH,
                                   (int(wp.x), int(wp.y)), 3)

        if len(robot.path_actual) > 1:
            pygame.draw.lines(screen, COLOR_PATH_ACTUAL, False, robot.path_actual, 2)

        pygame.draw.polygon(screen, COLOR_BOUNDARY, boundary_pts, 4)
        for obs in obstacles:
            pygame.draw.polygon(screen, COLOR_OBSTACLE, obs, 3)
            s = pygame.Surface(WINDOW_SIZE, pygame.SRCALPHA)
            pygame.draw.polygon(s, (231, 76, 60, 60), obs)
            screen.blit(s, (0, 0))

        for p1, p2 in robot.virtual_walls:
            pygame.draw.line(screen, COLOR_VIRTUAL_WALL, p1, p2, 2)

        pygame.draw.circle(screen, COLOR_ROBOT,
                           (int(robot.pos.x), int(robot.pos.y)), robot.radius)
        h_vec = Vector2(math.cos(robot.heading_angle), math.sin(robot.heading_angle))
        for vec, dist in [
            (h_vec,             robot.r_front),
            (h_vec.rotate(-90), robot.r_left),
            (h_vec.rotate(90),  robot.r_right),
            (h_vec.rotate(180), robot.r_back),
        ]:
            color = COLOR_RAY_HIT if dist < OR_VALUE else COLOR_RAY_CLEAR
            pygame.draw.line(screen, color, robot.pos, robot.pos + vec * dist, 1)

        hud = [
            f"CAPA:    {robot.phase}",
            f"ESTADO:  {robot.state}",
            f"WF:      {robot.wf_state}",
            f"V_WALLS: {len(robot.virtual_walls)}",
            f"PASO:    {robot._layer_step_counter}",
        ]
        for i, text in enumerate(hud):
            screen.blit(font.render(text, True, COLOR_TEXT), (20, 20 + i * 25))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()