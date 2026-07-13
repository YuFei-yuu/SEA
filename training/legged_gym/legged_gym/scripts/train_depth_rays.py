"""Train the depth-image to navigation-ray model used by go2_pos_depth_stairs."""
from __future__ import annotations

import argparse
import glob
import os

from isaacgym import gymapi
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from legged_gym.depth_ray import DepthRayNet


class DepthRayDataset(Dataset):
    def __init__(self, data_dir):
        files = sorted(glob.glob(os.path.join(data_dir, "depth_rays_*.npz")))
        if not files:
            raise FileNotFoundError(f"No depth_rays_*.npz files found in {data_dir}.")
        depth_batches, ray_batches = [], []
        for path in files:
            data = np.load(path)
            depth_batches.append(data["depth"])
            ray_batches.append(data["rays"])
        self.depth = np.concatenate(depth_batches, axis=0)
        self.rays = np.concatenate(ray_batches, axis=0)
        if self.depth.shape[0] != self.rays.shape[0]:
            raise ValueError("Depth and ray sample counts do not match.")
        if self.rays.ndim != 2:
            raise ValueError("Ray labels must be [samples, num_rays].")

    def __len__(self):
        return self.depth.shape[0]

    def __getitem__(self, index):
        return (
            torch.from_numpy(self.depth[index]).float(),
            torch.from_numpy(self.rays[index]).float(),
        )


def _evaluate(model, loader, device, depth_min, depth_max, ray_min, ray_max):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_samples = 0
    with torch.no_grad():
        for depth, rays in loader:
            depth = depth.to(device).clamp(min=depth_min, max=depth_max)
            rays = rays.to(device).clamp(min=ray_min, max=ray_max)
            prediction_log = model(torch.log2(depth).unsqueeze(1))
            target_log = torch.log2(rays)
            batch_size = depth.shape[0]
            total_loss += torch.mean((prediction_log - target_log) ** 2).item() * batch_size
            prediction = torch.exp2(prediction_log).clamp(min=ray_min, max=ray_max)
            total_mae += torch.mean(torch.abs(prediction - rays)).item() * batch_size
            total_samples += batch_size
    return total_loss / total_samples, total_mae / total_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--depth_min", type=float, default=0.10)
    parser.add_argument("--depth_max", type=float, default=5.00)
    parser.add_argument("--ray_min", type=float, default=0.10)
    parser.add_argument("--ray_max", type=float, default=5.00)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dataset = DepthRayDataset(args.data_dir)
    num_rays = dataset.rays.shape[1]
    test_size = max(1, int(0.10 * len(dataset)))
    val_size = max(1, int(0.10 * len(dataset)))
    train_size = len(dataset) - val_size - test_size
    if train_size < 1:
        raise ValueError("Dataset is too small for an 80/10/10 split.")
    generator = torch.Generator().manual_seed(args.seed)
    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthRayNet(num_rays=num_rays).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_val_mae = float("inf")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for depth, rays in train_loader:
            depth = depth.to(device).clamp(min=args.depth_min, max=args.depth_max)
            rays = rays.to(device).clamp(min=args.ray_min, max=args.ray_max)
            prediction_log = model(torch.log2(depth).unsqueeze(1))
            loss = torch.mean((prediction_log - torch.log2(rays)) ** 2)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        val_mse, val_mae = _evaluate(
            model, val_loader, device, args.depth_min, args.depth_max, args.ray_min, args.ray_max
        )
        print(f"epoch={epoch:03d} val_log_mse={val_mse:.6f} val_ray_mae={val_mae:.4f}m")
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_rays": int(num_rays),
                    "depth_min": args.depth_min,
                    "depth_max": args.depth_max,
                    "ray_min": args.ray_min,
                    "ray_max": args.ray_max,
                    "val_ray_mae": best_val_mae,
                },
                args.output,
            )

    checkpoint = torch.load(args.output, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_mse, test_mae = _evaluate(
        model, test_loader, device, args.depth_min, args.depth_max, args.ray_min, args.ray_max
    )
    print(f"best_val_ray_mae={best_val_mae:.4f}m test_log_mse={test_mse:.6f} test_ray_mae={test_mae:.4f}m")


if __name__ == "__main__":
    main()
