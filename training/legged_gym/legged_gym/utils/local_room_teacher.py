"""Local, goal-driven teacher utilities for static room navigation.

The teacher deliberately avoids binding an episode to a precomputed route.  It
tests a direct segment first and only creates lateral candidates around the
obstacles that can block that segment.  All operations are batched torch
operations so the helper can be used during large-environment rollouts.
"""

from __future__ import annotations

import torch


def _segment_intersects_aabb(starts, direction, lower, upper, eps):
    parallel = torch.abs(direction) <= eps
    safe_direction = torch.where(parallel, torch.ones_like(direction), direction)
    t1 = (lower - starts) / safe_direction
    t2 = (upper - starts) / safe_direction
    t_near = torch.minimum(t1, t2)
    t_far = torch.maximum(t1, t2)
    t_near = torch.where(parallel, torch.full_like(t_near, -float("inf")), t_near)
    t_far = torch.where(parallel, torch.full_like(t_far, float("inf")), t_far)
    parallel_outside = parallel & ((starts < lower) | (starts > upper))
    enter = t_near.amax(dim=-1)
    leave = t_far.amin(dim=-1)
    return (
        ~parallel_outside.any(dim=-1)
        & (enter <= leave)
        & (leave >= 0.0)
        & (enter <= 1.0)
    )


def segment_clearance(
    starts: torch.Tensor,
    goals: torch.Tensor,
    centers: torch.Tensor,
    half_extents: torch.Tensor,
    *,
    inflation: float = 0.42,
    samples: int = 32,
):
    """Return free mask and approximate AABB clearance for batched segments.

    Intersection is solved analytically with the slab method.  Candidate and
    obstacle dimensions are chunked so 4096/8192-environment teachers do not
    materialize ``[env, candidate, sample, obstacle, xy]`` tensors.

    Args:
        starts: ``[B, 2]`` segment starts.
        goals: ``[B, K, 2]`` segment goals or ``[B, 2]``.
        centers/half_extents: obstacle AABBs in the same local frame.
    """
    paired_starts = starts.ndim == 3
    if paired_starts:
        output_batch, output_candidates = starts.shape[:2]
        if goals.ndim != 3 or goals.shape[:2] != (output_batch, output_candidates):
            raise ValueError("paired starts/goals must both have shape [batch, candidates, 2]")
        starts = starts.reshape(output_batch * output_candidates, 2)
        goals = goals.reshape(output_batch * output_candidates, 2).unsqueeze(1)
    elif goals.ndim == 2:
        goals = goals.unsqueeze(1)
    batch, candidates = goals.shape[:2]
    if centers.numel() == 0:
        free = torch.ones(batch, candidates, dtype=torch.bool, device=starts.device)
        clearance = torch.full((batch, candidates), 10.0, device=starts.device)
        if paired_starts:
            return free.view(output_batch, output_candidates), clearance.view(output_batch, output_candidates)
        return free, clearance
    del samples  # Kept in the public signature for existing callers.
    expanded = half_extents + float(inflation)
    free = torch.ones(batch, candidates, dtype=torch.bool, device=starts.device)
    min_clearance = torch.full(
        (batch, candidates), float("inf"), dtype=starts.dtype, device=starts.device
    )
    candidate_chunk_size = 8
    obstacle_chunk_size = 16
    starts_view = starts[:, None, None, :]
    eps = torch.finfo(starts.dtype).eps * 16.0

    for candidate_start in range(0, candidates, candidate_chunk_size):
        candidate_end = min(candidate_start + candidate_chunk_size, candidates)
        goals_chunk = goals[:, candidate_start:candidate_end, None, :]
        direction = goals_chunk - starts_view
        blocked_chunk = torch.zeros(
            batch,
            candidate_end - candidate_start,
            dtype=torch.bool,
            device=starts.device,
        )
        clearance_chunk = torch.full_like(
            blocked_chunk, float("inf"), dtype=starts.dtype
        )

        for obstacle_start in range(0, centers.shape[0], obstacle_chunk_size):
            obstacle_end = min(obstacle_start + obstacle_chunk_size, centers.shape[0])
            centers_chunk = centers[obstacle_start:obstacle_end][None, None, :, :]
            extents_chunk = expanded[obstacle_start:obstacle_end][None, None, :, :]
            core_extents_chunk = half_extents[obstacle_start:obstacle_end][
                None, None, :, :
            ]
            lower = centers_chunk - extents_chunk
            upper = centers_chunk + extents_chunk
            intersects = _segment_intersects_aabb(
                starts_view, direction, lower, upper, eps
            )
            # A waypoint that lies inside an inflated obstacle is not a valid
            # bypass even when the segment starts inside the safety margin but
            # outside the physical core.  Reject the endpoint explicitly; the
            # start-inside exception below remains useful for recovery.
            goals_inside_expanded = torch.all(
                (goals_chunk >= lower) & (goals_chunk <= upper), dim=-1
            )
            starts_inside_expanded = torch.all(
                (starts_view >= lower) & (starts_view <= upper), dim=-1
            )
            if starts_inside_expanded.any():
                core_lower = centers_chunk - core_extents_chunk
                core_upper = centers_chunk + core_extents_chunk
                intersects_core = _segment_intersects_aabb(
                    starts_view, direction, core_lower, core_upper, eps
                )
                intersects = (
                    (intersects & ~starts_inside_expanded)
                    | intersects_core
                    | goals_inside_expanded
                )
            else:
                intersects |= goals_inside_expanded
            blocked_chunk |= intersects.any(dim=-1)

            start_delta = torch.clamp(
                torch.abs(starts_view - centers_chunk) - extents_chunk, min=0.0
            )
            goal_delta = torch.clamp(
                torch.abs(goals_chunk - centers_chunk) - extents_chunk, min=0.0
            )
            endpoint_clearance = torch.minimum(
                torch.linalg.vector_norm(start_delta, dim=-1),
                torch.linalg.vector_norm(goal_delta, dim=-1),
            ).amin(dim=-1)
            clearance_chunk = torch.minimum(clearance_chunk, endpoint_clearance)

        free[:, candidate_start:candidate_end] = ~blocked_chunk
        min_clearance[:, candidate_start:candidate_end] = torch.where(
            blocked_chunk, torch.zeros_like(clearance_chunk), clearance_chunk
        )
    if paired_starts:
        return free.view(output_batch, output_candidates), min_clearance.view(output_batch, output_candidates)
    return free, min_clearance


def choose_local_waypoint(
    starts: torch.Tensor,
    goals: torch.Tensor,
    centers: torch.Tensor,
    half_extents: torch.Tensor,
    *,
    inflation: float = 0.42,
    lateral_margin: float = 0.18,
    min_directed_progress: float = 0.05,
    room_size: float = 10.0,
    preferred_lateral_sign: torch.Tensor | None = None,
    min_clearance: float = 0.0,
    samples: int = 32,
):
    """Choose direct goal or the shortest feasible left/right local waypoint.

    The returned waypoint is the immediate target.  When the direct segment is
    free it is exactly ``goals``.  Otherwise, candidates are generated around
    obstacle corners and selected from the current pose.  The function is
    called again after every simulator step, so later blockers are handled by
    receding-horizon replanning instead of an episode-level fixed route.
    """
    if goals.ndim != 2 or starts.ndim != 2:
        raise ValueError("starts and goals must have shape [batch, 2]")
    if not starts.is_floating_point():
        starts = starts.float()
    if not goals.is_floating_point():
        goals = goals.float()
    batch = starts.shape[0]
    direct_free, direct_clearance = segment_clearance(
        starts, goals, centers, half_extents, inflation=inflation, samples=samples
    )
    direct_free = direct_free[:, 0]
    if centers.numel() == 0:
        return goals.clone(), direct_free, direct_clearance[:, 0], torch.zeros_like(direct_free)

    # Four corners for every obstacle.  The next step is selected from the
    # current pose only; after reaching a corner the function is called again,
    # which gives a local receding-horizon planner instead of a fixed route.
    x_offset = half_extents[:, 0] + float(inflation) + float(lateral_margin)
    y_offset = half_extents[:, 1] + float(inflation) + float(lateral_margin)
    center_batch = centers[None, :, :].expand(batch, -1, -1)
    candidates = torch.cat(
        (
            center_batch + torch.stack((x_offset, y_offset), dim=-1)[None, :, :],
            center_batch + torch.stack((x_offset, -y_offset), dim=-1)[None, :, :],
            center_batch + torch.stack((-x_offset, y_offset), dim=-1)[None, :, :],
            center_batch + torch.stack((-x_offset, -y_offset), dim=-1)[None, :, :],
        ),
        dim=1,
    )
    # Corner-only candidates can all be invalid when several low boxes form a
    # staggered column.  Add short, dynamically selected lateral gates across
    # the room so the receding-horizon teacher can choose a safe corridor
    # without binding an episode to a precomputed route template.
    goal_delta = goals - starts
    wall_lane = max(0.85, float(inflation) + float(lateral_margin) + 0.45)
    room_center = 0.5 * float(room_size)
    lane_y = torch.as_tensor(
        [wall_lane, room_center - 0.30, room_center + 0.30, float(room_size) - wall_lane],
        dtype=starts.dtype,
        device=starts.device,
    ).clamp(min=0.60, max=float(room_size) - 0.60)
    x_sign = torch.sign(goal_delta[:, 0]).clamp(min=-1.0, max=1.0)
    gate_x_ahead = (starts[:, 0] + x_sign * 1.0).clamp(min=0.75, max=9.25)
    gate_x_ahead = torch.where(
        x_sign >= 0.0,
        torch.minimum(gate_x_ahead, goals[:, 0]),
        torch.maximum(gate_x_ahead, goals[:, 0]),
    )
    near_offsets = torch.as_tensor(
        [0.10, 0.50, 1.00], dtype=starts.dtype, device=starts.device
    )
    gate_x_near = (
        starts[:, 0, None] - x_sign[:, None] * near_offsets[None, :]
    ).clamp(min=0.65, max=9.35)
    lane_candidates_ahead = torch.stack(
        (
            gate_x_ahead[:, None].expand(-1, lane_y.numel()),
            lane_y[None, :].expand(batch, -1),
        ),
        dim=-1,
    )
    lane_candidates_near = torch.stack(
        (
            gate_x_near[:, :, None].expand(-1, -1, lane_y.numel()),
            lane_y[None, None, :].expand(batch, near_offsets.numel(), -1),
        ),
        dim=-1,
    ).reshape(batch, -1, 2)
    candidates = torch.cat((candidates, lane_candidates_ahead, lane_candidates_near), dim=1)
    lane_candidate_count = (1 + near_offsets.numel()) * lane_y.numel()
    lattice_y = torch.arange(
        0.20, 9.81, 0.20, dtype=starts.dtype, device=starts.device
    )
    lattice_offsets = torch.as_tensor(
        [0.00, 0.10, 0.30, 0.60, 1.00], dtype=starts.dtype, device=starts.device
    )
    lattice_x = starts[:, 0, None] + x_sign[:, None] * lattice_offsets[None, :]
    lattice_x = lattice_x.clamp(min=0.55, max=9.45)
    lattice_x = torch.where(
        x_sign[:, None] >= 0.0,
        torch.minimum(lattice_x, goals[:, 0, None]),
        torch.maximum(lattice_x, goals[:, 0, None]),
    )
    lattice_candidates = torch.stack(
        (
            lattice_x[:, :, None].expand(-1, -1, lattice_y.numel()),
            lattice_y[None, None, :].expand(batch, lattice_offsets.numel(), -1),
        ),
        dim=-1,
    ).reshape(batch, -1, 2)
    candidates = torch.cat((candidates, lattice_candidates), dim=1)
    lattice_candidate_count = lattice_candidates.shape[1]
    candidate_free, candidate_clearance = segment_clearance(
        starts,
        candidates,
        centers,
        half_extents,
        inflation=inflation,
        samples=samples,
    )
    start_inside_expanded = torch.all(
        torch.abs(starts[:, None, :] - centers[None, :, :])
        <= half_extents[None, :, :] + float(inflation),
        dim=-1,
    ).any(dim=1)
    if float(min_clearance) > 0.0:
        # The start point's clearance can be small while the only safe
        # recovery is lateral.  Apply the endpoint margin strictly in normal
        # operation, but do not discard all lateral escapes from that state.
        candidate_free &= (candidate_clearance >= float(min_clearance)) | start_inside_expanded[:, None]
    if centers.shape[0] <= 4:
        candidate_free[:, -(lane_candidate_count + lattice_candidate_count) :] = False
    goal_delta = goals - starts
    goal_norm = torch.linalg.vector_norm(goal_delta, dim=-1).clamp(min=1.0e-6)
    goal_direction = goal_delta / goal_norm[:, None]
    candidate_delta = candidates - starts[:, None, :]
    goal_progress = torch.sum(candidate_delta * goal_direction[:, None, :], dim=-1)
    directed_x = candidate_delta[..., 0] * torch.sign(goal_delta[:, 0])[:, None]
    use_x_direction = torch.abs(goal_delta[:, 0]) > 0.20
    directed_progress = torch.where(use_x_direction[:, None], directed_x, goal_progress)
    lane_mask = torch.zeros(
        batch, candidates.shape[1], dtype=torch.bool, device=starts.device
    )
    lane_mask[:, -(lane_candidate_count + lattice_candidate_count) : -lattice_candidate_count] = True
    if centers.shape[0] <= 4:
        lane_mask.zero_()
    # If the current footprint is already inside an inflated obstacle margin,
    # a nominally forward endpoint can still cut through the physical box.
    # Require an actual lateral displacement until the robot has escaped that
    # margin; this is a local recovery rule, not a fixed episode route.
    lateral_escape = torch.abs(candidates[..., 1] - starts[:, None, 1]) >= 0.15
    forward_feasible = candidate_free & (
        directed_progress >= float(min_directed_progress)
    ) & (~start_inside_expanded[:, None] | lateral_escape)
    # A near lane is an intentional lateral escape, used only when no forward
    # candidate is currently reachable.  This preserves direct progress and
    # avoids the old fixed "move to a wall first" template.
    has_forward = forward_feasible.any(dim=1, keepdim=True)
    lane_escape = lane_mask & (
        torch.abs(candidates[..., 1] - starts[:, None, 1]) >= 0.25
    )
    forward_feasible |= lane_escape & candidate_free & ~has_forward
    forward_feasible |= lateral_escape & candidate_free & ~has_forward
    # Prefer a simultaneous forward/lateral bypass.  When the robot starts
    # close to an obstacle safety margin, allow a short escape corner instead
    # of falsely labelling an otherwise navigable state as blocked.
    backtrack_escape = directed_progress <= -0.25
    backtrack_escape |= (directed_progress < 0.0) & (
        torch.abs(candidates[..., 1] - starts[:, None, 1]) >= 0.15
    )
    has_escape = (candidate_free & (lateral_escape | backtrack_escape)).any(
        dim=1, keepdim=True
    )
    fallback = torch.where(
        has_escape,
        candidate_free & (lateral_escape | backtrack_escape),
        candidate_free,
    )
    feasible = torch.where(forward_feasible.any(dim=1, keepdim=True), forward_feasible, fallback)
    candidate_cost = (
        torch.linalg.vector_norm(candidate_delta, dim=-1)
        + 0.75 * torch.linalg.vector_norm(goals[:, None, :] - candidates, dim=-1)
        + 0.08 * torch.abs(candidates[..., 1] - starts[:, None, 1])
    )
    if preferred_lateral_sign is not None:
        sign = preferred_lateral_sign.to(dtype=starts.dtype)[:, None]
        opposite_lateral = (sign != 0.0) & (
            (candidates[..., 1] - starts[:, None, 1]) * sign < -0.05
        )
        candidate_cost += opposite_lateral.float() * (
            1.25 * torch.abs(candidates[..., 1] - starts[:, None, 1])
        )
    lane_cost = (
        0.35 * torch.linalg.vector_norm(candidate_delta, dim=-1)
        + 0.35 * torch.linalg.vector_norm(goals[:, None, :] - candidates, dim=-1)
        + 0.01 * torch.abs(candidates[..., 1] - starts[:, None, 1])
        + 0.45 * torch.abs(candidates[..., 1] - room_center)
        + 0.50 * torch.abs(candidates[..., 1] - (room_center - 0.30))
    )
    candidate_cost = torch.where(lane_mask, lane_cost, candidate_cost)
    # Lattice gates trade a little path length for an actual free corridor;
    # prefer the nearest forward gate among them when corner candidates are
    # disconnected by the staggered obstacle rows.
    lattice_mask = torch.zeros(
        batch, candidates.shape[1], dtype=torch.bool, device=starts.device
    )
    lattice_mask[:, -lattice_candidate_count:] = True
    if centers.shape[0] <= 4:
        lattice_mask.zero_()
    lattice_cost = (
        0.80 * torch.linalg.vector_norm(candidate_delta, dim=-1)
        + 0.75 * torch.linalg.vector_norm(goals[:, None, :] - candidates, dim=-1)
        + 0.02 * torch.abs(candidates[..., 1] - starts[:, None, 1])
    )
    candidate_cost = torch.where(lattice_mask, lattice_cost, candidate_cost)
    candidate_cost = torch.where(feasible, candidate_cost, torch.full_like(candidate_cost, 1.0e6))
    best_cost, best_idx = candidate_cost.min(dim=1)
    best_waypoint = candidates[torch.arange(batch, device=starts.device), best_idx]
    use_candidate = (~direct_free) & torch.isfinite(best_cost) & (best_cost < 1.0e5)
    waypoint = torch.where(use_candidate[:, None], best_waypoint, goals)
    direct_distance = torch.linalg.vector_norm(goals - starts, dim=-1).clamp(min=1.0e-6)
    selected_path_distance = (
        torch.linalg.vector_norm(best_waypoint - starts, dim=-1)
        + torch.linalg.vector_norm(goals - best_waypoint, dim=-1)
    )
    best_distance = torch.where(
        use_candidate, selected_path_distance, direct_distance
    )
    passability = torch.where(
        use_candidate,
        torch.ones(batch, dtype=torch.long, device=starts.device),
        torch.where(
            direct_free,
            torch.zeros(batch, dtype=torch.long, device=starts.device),
            torch.full((batch,), 3, dtype=torch.long, device=starts.device),
        ),
    )
    return waypoint, direct_free, best_distance, passability


def local_teacher_metrics(starts, actions, goals, dt):
    """Compute template-detection metrics for teacher diagnostics."""
    lateral_only = (torch.abs(actions[:, 0]) < 0.08) & (
        torch.abs(actions[:, 1]) > 0.08
    )
    forward_ready = actions[:, 0] > 0.08
    goal_distance = torch.linalg.vector_norm(goals - starts, dim=-1)
    return {
        "lateral_only": lateral_only,
        "forward_ready": forward_ready,
        "goal_distance": goal_distance,
        "dt": torch.full_like(goal_distance, float(dt)),
    }
