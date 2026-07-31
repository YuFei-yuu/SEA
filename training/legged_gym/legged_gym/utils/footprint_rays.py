"""Vectorized 2-D ray intersections for known static room footprints."""

from __future__ import annotations

import torch


def ray_aabb_distances(
    origins: torch.Tensor,
    directions: torch.Tensor,
    centers: torch.Tensor,
    half_extents: torch.Tensor,
    min_dist: float,
    max_dist: float,
) -> torch.Tensor:
    """Return ray-to-AABB distances.

    ``origins`` is ``[B, 2]``, ``directions`` is ``[B, R, 2]`` and boxes are
    supplied as ``[N, 2]`` or ``[B, N, 2]``. The returned tensor is ``[B, R, N]``.
    Rays parallel to a box face only hit when their origin lies inside that
    slab, which avoids NaNs and false hits at zero direction components.
    """
    if origins.ndim != 2 or origins.shape[-1] != 2:
        raise ValueError("origins must have shape [batch, 2]")
    if directions.ndim != 3 or directions.shape[-1] != 2:
        raise ValueError("directions must have shape [batch, rays, 2]")
    if centers.ndim not in (2, 3) or centers.shape[-1] != 2:
        raise ValueError("centers must have shape [boxes, 2] or [batch, boxes, 2]")
    if half_extents.shape != centers.shape:
        raise ValueError("half_extents must have the same shape as centers")

    if centers.ndim == 2:
        centers = centers.unsqueeze(0)
        half_extents = half_extents.unsqueeze(0)
    if centers.shape[0] not in (1, origins.shape[0]):
        raise ValueError("box batch dimension must be 1 or match origins")
    if centers.shape[0] == 1:
        centers = centers.expand(origins.shape[0], -1, -1)
        half_extents = half_extents.expand(origins.shape[0], -1, -1)

    lower = centers.unsqueeze(1) - half_extents.unsqueeze(1)
    upper = centers.unsqueeze(1) + half_extents.unsqueeze(1)
    origin = origins.unsqueeze(1).unsqueeze(2)
    direction = directions.unsqueeze(2)

    eps = torch.finfo(directions.dtype).eps * 16.0
    parallel = direction.abs() <= eps
    inside_parallel = (origin >= lower) & (origin <= upper)
    safe_direction = torch.where(
        parallel,
        torch.where(direction >= 0.0, torch.full_like(direction, eps), torch.full_like(direction, -eps)),
        direction,
    )
    t_lower = (lower - origin) / safe_direction
    t_upper = (upper - origin) / safe_direction
    t_near = torch.minimum(t_lower, t_upper)
    t_far = torch.maximum(t_lower, t_upper)
    t_enter = t_near.max(dim=-1).values
    t_exit = t_far.min(dim=-1).values
    parallel_valid = (~parallel | inside_parallel).all(dim=-1)
    hit = parallel_valid & (t_exit >= torch.maximum(t_enter, torch.zeros_like(t_enter)))

    distance = torch.clamp(t_enter, min=float(min_dist), max=float(max_dist))
    return torch.where(hit, distance, torch.full_like(distance, float(max_dist)))


def room_footprint_boxes(
    low_obstacle_boxes,
    *,
    room_size: float = 10.0,
    wall_thickness: float = 0.30,
    inflation: float = 0.15,
    device=None,
    dtype=torch.float32,
):
    """Build local-room wall and low-obstacle AABBs with uniform inflation."""
    boxes = [
        (0.5 * wall_thickness, 0.5 * room_size, wall_thickness, room_size),
        (room_size - 0.5 * wall_thickness, 0.5 * room_size, wall_thickness, room_size),
        (0.5 * room_size, 0.5 * wall_thickness, room_size, wall_thickness),
        (0.5 * room_size, room_size - 0.5 * wall_thickness, room_size, wall_thickness),
    ]
    boxes.extend((float(cx), float(cy), float(sx), float(sy)) for cx, cy, sx, sy, _ in low_obstacle_boxes)
    values = torch.tensor(boxes, device=device, dtype=dtype)
    centers = values[:, :2]
    half_extents = values[:, 2:4].mul(0.5).add(float(inflation))
    return centers, half_extents
