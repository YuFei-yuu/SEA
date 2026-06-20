import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ExactLSECBFLayer(nn.Module):
    def __init__(self,
                 num_rays=41,
                 fov_deg=180.0,
                 safe_radius=0.15,
                 safety_margin=0.05,
                 kappa=10.0,
                 damping_factor=1.0):
        super().__init__()
        
        self.d_safe = safe_radius + safety_margin
        self.kappa = kappa
        self.damping_factor = damping_factor
        
        # Pre-calculate unit direction vectors n_i
        start_angle = -np.deg2rad(fov_deg) / 2
        end_angle = np.deg2rad(fov_deg) / 2
        angles = torch.linspace(start_angle, end_angle, num_rays)
        self.register_buffer('ray_unit_vectors',
            torch.stack([torch.cos(angles), torch.sin(angles)], dim=1))

    def forward(self, u_bar, lidar_dists, alpha):
        """
        u_bar: [B, 3] Nominal policy (vx, vy, yaw)
        lidar_dists: [B, num_rays] Lidar distances (processed externally to 0.1~5.0)
        alpha: [B, 1] Class-K function parameter (adaptively learned)
        """
        u_2d = u_bar[:, :2]  # Corresponding to \bar{u}(x) in the paper
        yaw_rate = u_bar[:, 2:]

        # 1. Calculate independent h_i(x)
        h_i = lidar_dists - self.d_safe  # [B, num_rays]

        # 2. Calculate composite CBF: h(x) (Corresponding to Eq. 14 in the paper)
        min_h, _ = torch.min(h_i, dim=1, keepdim=True) 
        h_comp = min_h - (1.0 / self.kappa) * torch.log(
            torch.sum(torch.exp(-self.kappa * (h_i - min_h)), dim=1, keepdim=True)
        ) # [B, 1]

        # 3. Calculate \lambda_i(x)
        lambda_i = torch.exp(-self.kappa * (h_i - h_comp)).unsqueeze(-1) # [B, num_rays, 1]

        # 4. Calculate L_g h(x)
        # L_g h_i = -n_i, so L_g h = - \sum \lambda_i n_i
        n_vecs = self.ray_unit_vectors.unsqueeze(0) # [1, num_rays, 2]
        Lg_h = -torch.sum(lambda_i * n_vecs, dim=1) # [B, 2]

        # 5. Calculate \eta(x)
        # \eta = - (L_f h + L_g h * u_bar + \alpha * h) / ||L_g h||^2 # Note: L_f h = 0
        Lgh_u = torch.sum(Lg_h * u_2d, dim=1, keepdim=True) # [B, 1]
        Lgh_norm_sq = torch.sum(Lg_h**2, dim=1, keepdim=True) # [B, 1]
        
        damping_factor = self.damping_factor  # Larger means smoother, but slightly sacrifices safety
        eta = - (Lgh_u + alpha * h_comp) / (Lgh_norm_sq + damping_factor)

        # 6. Calculate safe action u_s(x)
        u_s_2d = u_2d + F.relu(eta) * Lg_h # [B, 2]

        u_s = torch.cat((u_s_2d, yaw_rate), dim=-1) # [B, 3]

        return u_s


class DynamicTokenCBFLayer(nn.Module):
    def __init__(
        self,
        default_safe_radius=0.45,
        safety_margin=0.35,
        damping_factor=1.0,
    ):
        super().__init__()
        self.default_safe_radius = default_safe_radius
        self.safety_margin = safety_margin
        self.damping_factor = damping_factor

    def forward(self, u_bar, dyn_tokens, base_vel=None, alpha=None):
        """
        u_bar: [B, 3] current safe/nominal action in robot local frame.
        dyn_tokens: [B, K * 7] or [B, K, 7] with
            [rel_x, rel_y, rel_vx, rel_vy, radius, ttc, valid].
        base_vel: [B, 2] current robot velocity in the robot local frame.
            Tokens store obstacle velocity relative to this measured velocity.
        alpha: [B, 1] class-K parameter.
        """
        if dyn_tokens is None or dyn_tokens.numel() == 0:
            return u_bar

        if dyn_tokens.dim() == 2:
            if dyn_tokens.shape[-1] % 7 != 0:
                return u_bar
            dyn_tokens = dyn_tokens.view(dyn_tokens.shape[0], -1, 7)
        if dyn_tokens.shape[1] == 0:
            return u_bar

        alpha = 1.0 if alpha is None else alpha
        if not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha, device=u_bar.device, dtype=u_bar.dtype)
        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(-1)

        pos = dyn_tokens[..., 0:2]
        rel_vel = dyn_tokens[..., 2:4]
        radius = dyn_tokens[..., 4:5].clamp(min=0.0)
        valid = dyn_tokens[..., 6:7] > 0.5

        d_safe = (radius + self.safety_margin).clamp(min=self.default_safe_radius)
        h = (pos ** 2).sum(dim=-1, keepdim=True) - d_safe ** 2
        grad_h = -2.0 * pos

        u_2d = u_bar[:, :2]
        yaw_rate = u_bar[:, 2:]
        if base_vel is None:
            base_vel = torch.zeros_like(u_2d)
        obstacle_vel = rel_vel + base_vel.unsqueeze(1)
        robot_vel = u_2d.unsqueeze(1)
        h_dot = 2.0 * (pos * (obstacle_vel - robot_vel)).sum(dim=-1, keepdim=True)
        violation = -(h_dot + alpha.unsqueeze(1) * h)
        correction_scale = F.relu(violation) / (
            (grad_h ** 2).sum(dim=-1, keepdim=True) + self.damping_factor
        )
        candidate_u = robot_vel + correction_scale * grad_h
        candidate_u = torch.where(valid, candidate_u, robot_vel.expand_as(candidate_u))
        candidate_delta = candidate_u - robot_vel
        candidate_norm = torch.norm(candidate_delta, dim=-1)
        candidate_norm = torch.where(
            valid.squeeze(-1),
            candidate_norm,
            torch.full_like(candidate_norm, -1.0),
        )
        best_idx = torch.argmax(candidate_norm, dim=1)
        batch_idx = torch.arange(u_bar.shape[0], device=u_bar.device)
        u_dyn = candidate_u[batch_idx, best_idx]
        has_valid = valid.squeeze(-1).any(dim=1, keepdim=True)
        u_dyn = torch.where(has_valid, u_dyn, u_2d)
        return torch.cat((u_dyn, yaw_rate), dim=-1)
