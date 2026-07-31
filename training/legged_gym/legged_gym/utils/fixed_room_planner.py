"""Deterministic 2-D route templates for the fixed stair room."""

from __future__ import annotations

import heapq
import math

import numpy as np


def build_occupancy_grid(
    obstacle_boxes,
    *,
    room_size=10.0,
    resolution=0.10,
    obstacle_inflation=0.42,
    boundary_margin=0.40,
):
    """Rasterize wall clearance and full-size obstacle AABBs."""
    cells = int(round(float(room_size) / float(resolution))) + 1
    coordinates = np.arange(cells, dtype=np.float64) * float(resolution)
    x, y = np.meshgrid(coordinates, coordinates, indexing="ij")
    occupied = (
        (x < boundary_margin)
        | (x > room_size - boundary_margin)
        | (y < boundary_margin)
        | (y > room_size - boundary_margin)
    )
    for center_x, center_y, size_x, size_y, _ in obstacle_boxes:
        occupied |= (
            (np.abs(x - float(center_x)) <= 0.5 * float(size_x) + obstacle_inflation)
            & (np.abs(y - float(center_y)) <= 0.5 * float(size_y) + obstacle_inflation)
        )
    return occupied


def _grid_point(point, resolution):
    # Use deterministic half-up rounding. Python's bankers rounding can map an
    # exact half-cell differently after float32 conversion, making a segment
    # appear clear during planning but occupied when stored as a torch route.
    return tuple(
        int(math.floor(float(value) / float(resolution) + 0.5 + 1.0e-9))
        for value in point
    )


def _world_point(cell, resolution):
    return tuple(float(value) * float(resolution) for value in cell)


def astar_path(occupied, start, goal, *, resolution=0.10):
    """Return an 8-connected collision-free path without diagonal corner cuts."""
    start_cell = _grid_point(start, resolution)
    goal_cell = _grid_point(goal, resolution)
    rows, cols = occupied.shape
    for name, cell in (("start", start_cell), ("goal", goal_cell)):
        if not (0 <= cell[0] < rows and 0 <= cell[1] < cols):
            raise ValueError(f"Route {name} {cell} is outside the room grid")
        if occupied[cell]:
            raise ValueError(f"Route {name} {cell} is occupied")

    moves = tuple(
        (dx, dy, math.hypot(dx, dy))
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        if dx != 0 or dy != 0
    )
    distance = {start_cell: 0.0}
    previous = {}
    queue = [(math.dist(start_cell, goal_cell), start_cell)]
    while queue:
        _, current = heapq.heappop(queue)
        if current == goal_cell:
            break
        for dx, dy, cost in moves:
            neighbor = (current[0] + dx, current[1] + dy)
            if not (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols):
                continue
            if occupied[neighbor]:
                continue
            if dx != 0 and dy != 0 and (
                occupied[current[0] + dx, current[1]]
                or occupied[current[0], current[1] + dy]
            ):
                continue
            candidate = distance[current] + cost
            if candidate >= distance.get(neighbor, math.inf):
                continue
            distance[neighbor] = candidate
            previous[neighbor] = current
            priority = candidate + math.dist(neighbor, goal_cell)
            heapq.heappush(queue, (priority, neighbor))
    if goal_cell not in distance:
        raise RuntimeError(f"No fixed-map route from {start} to {goal}")

    cells = [goal_cell]
    while cells[-1] != start_cell:
        cells.append(previous[cells[-1]])
    cells.reverse()
    return [_world_point(cell, resolution) for cell in cells]


def segment_is_free(occupied, start, goal, *, resolution=0.10):
    distance = math.dist(start, goal)
    sample_count = max(2, int(math.ceil(distance / (0.25 * resolution))))
    rows, cols = occupied.shape
    for alpha in np.linspace(0.0, 1.0, sample_count):
        point = (
            start[0] + alpha * (goal[0] - start[0]),
            start[1] + alpha * (goal[1] - start[1]),
        )
        cell = _grid_point(point, resolution)
        if not (0 <= cell[0] < rows and 0 <= cell[1] < cols) or occupied[cell]:
            return False
    return True


def simplify_path(occupied, path, *, resolution=0.10):
    """String-pull an A* path while retaining inflated-map clearance."""
    if len(path) <= 2:
        return list(path)
    simplified = [path[0]]
    index = 0
    while index < len(path) - 1:
        candidate = len(path) - 1
        while candidate > index + 1 and not segment_is_free(
            occupied, path[index], path[candidate], resolution=resolution
        ):
            candidate -= 1
        simplified.append(path[candidate])
        index = candidate
    return simplified


def _append_distinct(route, points, tolerance=1.0e-5):
    for point in points:
        point = (float(point[0]), float(point[1]))
        if not route or math.dist(route[-1], point) > tolerance:
            route.append(point)


def build_bidirectional_route_templates(
    obstacle_boxes,
    *,
    start_y_range,
    route_bins,
    low_start_x,
    high_start_x,
    up_goal,
    down_goal,
    low_staging=(4.40, 4.55),
    high_staging=(7.10, 4.85),
    room_size=10.0,
    resolution=0.10,
    obstacle_inflation=0.42,
    boundary_margin=0.40,
):
    """Build padded [direction, start-bin, waypoint, xy] route templates."""
    if int(route_bins) < 2:
        raise ValueError("route_bins must be at least two")
    occupied = build_occupancy_grid(
        obstacle_boxes,
        room_size=room_size,
        resolution=resolution,
        obstacle_inflation=obstacle_inflation,
        boundary_margin=boundary_margin,
    )
    start_values = np.linspace(float(start_y_range[0]), float(start_y_range[1]), int(route_bins))
    up_routes = []
    down_routes = []
    down_low_path = simplify_path(
        occupied,
        astar_path(occupied, low_staging, down_goal, resolution=resolution),
        resolution=resolution,
    )
    for start_y in start_values:
        low_start = (float(low_start_x), float(start_y))
        high_start = (float(high_start_x), float(start_y))
        up_low_path = simplify_path(
            occupied,
            astar_path(occupied, low_start, low_staging, resolution=resolution),
            resolution=resolution,
        )
        up_route = []
        _append_distinct(up_route, up_low_path)
        _append_distinct(up_route, (high_staging, up_goal))
        up_routes.append(up_route)

        down_route = []
        _append_distinct(down_route, (high_start, high_staging, low_staging))
        _append_distinct(down_route, down_low_path[1:])
        down_routes.append(down_route)

    all_routes = (up_routes, down_routes)
    max_points = max(len(route) for direction in all_routes for route in direction)
    templates = np.zeros((2, int(route_bins), max_points, 2), dtype=np.float32)
    lengths = np.zeros((2, int(route_bins)), dtype=np.int64)
    for direction_index, routes in enumerate(all_routes):
        for bin_index, route in enumerate(routes):
            lengths[direction_index, bin_index] = len(route)
            templates[direction_index, bin_index, : len(route)] = np.asarray(route, dtype=np.float32)
            templates[direction_index, bin_index, len(route) :] = route[-1]
    return templates, lengths, start_values.astype(np.float32), occupied
