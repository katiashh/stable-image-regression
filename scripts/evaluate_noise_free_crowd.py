# -*- coding: utf-8 -*-
"""
Noise-free evaluation for ShanghaiTech Part B crowd counting.

The model output and the ground-truth count are normalized to [0, 1]:

    normalized_count = clip(count / count_scale, 0, 1)

The default count_scale=100 follows the existing crowd scripts.

The script reports:
  - Clean MSE;
  - Attacked MSE;
  - empirical B(x) for clean and attacked inputs;
  - standard deviations and 95% confidence intervals over images.

For the noise-free baseline, C(x)=1 because there is no prediction
aggregator, and therefore B(x)=eps_B * L(x).
"""

import argparse
import csv
import os
import random
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    dilation_rate = 2 if dilation else 1
    layers = []

    for value in cfg:
        if value == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            conv = nn.Conv2d(
                in_channels,
                value,
                kernel_size=3,
                padding=dilation_rate,
                dilation=dilation_rate,
            )
            if batch_norm:
                layers.extend([conv, nn.BatchNorm2d(value), nn.ReLU(inplace=True)])
            else:
                layers.extend([conv, nn.ReLU(inplace=True)])
            in_channels = value

    return nn.Sequential(*layers)


class CSRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = 0
        self.frontend_feat = [
            64, 64, "M", 128, 128, "M", 256, 256, 256,
            "M", 512, 512, 512,
        ]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(
            self.backend_feat,
            in_channels=512,
            dilation=True,
        )
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        count = x.sum(dim=(1, 2, 3))

        # Same normalization as the previous crowd script.
        count = count / 100.0
        return torch.clamp(count, min=0.0, max=1.0)


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
def find_ground_truth_path(gt_dir, image_path):
    """Find ShanghaiTech ground truth in .mat or .h5 format."""
    stem = image_path.stem
    candidates = [
        gt_dir / f"{stem}.h5",
        gt_dir / f"GT_{stem}.h5",
        gt_dir / f"{stem}.mat",
        gt_dir / f"GT_{stem}.mat",
    ]
    return next((path for path in candidates if path.exists()), None)


def read_ground_truth_count(gt_path):
    """Read the crowd count from either ShanghaiTech .h5 or .mat GT."""
    if gt_path.suffix.lower() == ".h5":
        with h5py.File(gt_path, "r") as gt_file:
            if "density" not in gt_file:
                raise KeyError("density dataset is missing")
            return float(np.asarray(gt_file["density"]).sum())

    # This is the standard ShanghaiTech Part B annotation format.
    mat = sio.loadmat(str(gt_path))

    if "density" in mat:
        return float(np.asarray(mat["density"]).sum())

    if "image_info" in mat:
        # GT_IMG_XXX.mat -> image_info[0, 0][0, 0][0]
        points = mat["image_info"][0, 0][0, 0][0]
        return float(len(np.asarray(points)))

    for key in ("annPoints", "points", "locations"):
        if key in mat:
            return float(len(np.asarray(mat[key])))

    raise KeyError(
        f"No crowd annotations found in {gt_path}. "
        f"Available keys: {list(mat.keys())}"
    )


class CrowdDataset(Dataset):
    def __init__(self, image_dir, gt_dir=None, count_scale=100.0):
        self.image_dir = Path(image_dir)
        if gt_dir is not None:
            self.gt_dir = Path(gt_dir)
        else:
            # ShanghaiTech installations use both spellings:
            #   test_data/ground-truth
            #   test_data/ground_truth
            candidates = [
                self.image_dir.parent / "ground-truth",
                self.image_dir.parent / "ground_truth",
                self.image_dir.parent / "groundtruth",
            ]
            self.gt_dir = next(
                (candidate for candidate in candidates if candidate.exists()),
                candidates[0],
            )
        self.count_scale = float(count_scale)

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.gt_dir.exists():
            raise FileNotFoundError(
                f"Ground-truth directory not found: {self.gt_dir}"
            )

        self.items = []
        skipped = []

        for image_path in sorted(self.image_dir.glob("*.jpg")):
            gt_path = find_ground_truth_path(self.gt_dir, image_path)

            if gt_path is None:
                skipped.append((image_path.name, "missing ground truth"))
                continue

            try:
                with Image.open(image_path) as image:
                    image.verify()
                read_ground_truth_count(gt_path)
            except Exception as exc:
                skipped.append((image_path.name, str(exc)))
                continue

            self.items.append((image_path, gt_path))

        print(
            f"[{self.image_dir}] usable images: {len(self.items)}; "
            f"skipped: {len(skipped)}"
        )
        if skipped:
            print("First skipped files:")
            for filename, reason in skipped[:10]:
                print(f"  - {filename}: {reason}")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        image_path, gt_path = self.items[index]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        count = read_ground_truth_count(gt_path)
        target = np.clip(count / self.count_scale, 0.0, 1.0)
        target = torch.tensor(target, dtype=torch.float32)

        return image, target, image_path.name


# -----------------------------------------------------------------------------
# Checkpoint and statistics
# -----------------------------------------------------------------------------
def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        state = None
        for key in ("base_state", "model_state", "state_dict", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state = checkpoint[key]
                break
        if state is None:
            state = checkpoint
    else:
        state = checkpoint

    state = {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state.items()
    }
    model.load_state_dict(state, strict=True)
    return model


def stats(values):
    values = values.detach().cpu().float().flatten()
    n = int(values.numel())

    if n == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }

    mean = float(values.mean())
    std = float(values.std(unbiased=(n > 1))) if n > 1 else 0.0
    half_width = 1.96 * std / np.sqrt(n)

    return {
        "mean": mean,
        "std": std,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


# -----------------------------------------------------------------------------
# Attack and local stability diagnostic
# -----------------------------------------------------------------------------
def pgd_l2_attack(
    model,
    x,
    y,
    eps=20.0,
    steps=10,
    step_size=None,
):
    """Untargeted L2-PGD maximizing normalized-count MSE."""
    if step_size is None:
        step_size = eps / steps

    # ImageNet-normalized input bounds corresponding to RGB [0, 1].
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    lower = (0.0 - mean) / std
    upper = (1.0 - mean) / std

    x0 = x.detach()
    delta = torch.zeros_like(x0)

    for _ in range(steps):
        delta.requires_grad_(True)
        x_adv = torch.max(torch.min(x0 + delta, upper), lower)
        pred = model(x_adv).view(-1)
        loss = F.mse_loss(pred, y, reduction="sum")
        grad = torch.autograd.grad(loss, delta, only_inputs=True)[0]

        grad_norm = grad.flatten(1).norm(p=2, dim=1).view(-1, 1, 1, 1)
        direction = grad / (grad_norm + 1e-12)
        delta = delta.detach() + step_size * direction

        delta_flat = delta.flatten(1)
        delta_norm = delta_flat.norm(p=2, dim=1, keepdim=True)
        factor = torch.clamp(eps / (delta_norm + 1e-12), max=1.0)
        delta = (delta_flat * factor).view_as(delta)
        delta = torch.max(torch.min(x0 + delta, upper), lower) - x0

    return (torch.max(torch.min(x0 + delta, upper), lower)).detach()


def local_l2_deterministic(model, x, radius, steps=3, step_size=0.25):
    """Estimate L(x)=max ||grad_x f(x+u)||_2 in an L2 ball."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    lower = (0.0 - mean) / std
    upper = (1.0 - mean) / std

    x0 = x.detach()
    u = torch.zeros_like(x0)

    for _ in range(steps):
        z = torch.max(torch.min(x0 + u, upper), lower)
        z = z.detach().requires_grad_(True)
        output = model(z).view(-1)
        grad = torch.autograd.grad(output.sum(), z, only_inputs=True)[0]

        grad_norm = grad.flatten(1).norm(p=2, dim=1).view(-1, 1, 1, 1)
        direction = grad / (grad_norm + 1e-12)
        z_adv = torch.max(
            torch.min(z + step_size * radius * direction, upper),
            lower,
        )

        u = z_adv.detach() - x0
        u_flat = u.flatten(1)
        u_norm = u_flat.norm(p=2, dim=1, keepdim=True)
        factor = torch.clamp(radius / (u_norm + 1e-12), max=1.0)
        u = (u_flat * factor).view_as(u)

    z = torch.max(torch.min(x0 + u, upper), lower)
    z = z.detach().requires_grad_(True)
    output = model(z).view(-1)
    grad = torch.autograd.grad(output.sum(), z, only_inputs=True)[0]
    return grad.flatten(1).norm(p=2, dim=1).detach().float()


def evaluate(model, loader, device, attacked, args):
    model.eval()
    squared_errors = []
    all_B = []
    all_L = []
    total = 0

    iterator = tqdm(loader, desc="Attacked" if attacked else "Clean")
    for batch_index, (x, y, filenames) in enumerate(iterator):
        if args.max_batches is not None and batch_index >= args.max_batches:
            break

        x = x.to(device).float()
        y = y.to(device).view(-1).float()

        if attacked:
            x_eval = pgd_l2_attack(
                model,
                x,
                y,
                eps=args.attack_eps,
                steps=args.attack_steps,
            )
        else:
            x_eval = x

        with torch.no_grad():
            pred = model(x_eval).view(-1)

        squared_errors.append((pred - y).pow(2).detach().cpu())

        # Noise-free model: C(x)=1, hence B=eps_B * L.
        L = local_l2_deterministic(
            model,
            x_eval,
            radius=args.b_eps,
            steps=args.b_steps,
            step_size=args.b_step_size,
        )
        B = args.b_eps * L
        all_L.append(L.cpu())
        all_B.append(B.cpu())
        total += x.size(0)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    squared_errors = torch.cat(squared_errors)
    B = torch.cat(all_B)
    L = torch.cat(all_L)

    mse_stats = stats(squared_errors)
    B_stats = stats(B)
    L_stats = stats(L)

    return {
        "MSE": mse_stats["mean"],
        "MSE_std": mse_stats["std"],
        "MSE_ci95_low": mse_stats["ci95_low"],
        "MSE_ci95_high": mse_stats["ci95_high"],
        "B_mean": B_stats["mean"],
        "B_std": B_stats["std"],
        "B_ci95_low": B_stats["ci95_low"],
        "B_ci95_high": B_stats["ci95_high"],
        "B_p90": float(torch.quantile(B, 0.90)),
        "B_p99": float(torch.quantile(B, 0.99)),
        "L_mean": L_stats["mean"],
        "L_std": L_stats["std"],
        "L_ci95_low": L_stats["ci95_low"],
        "L_ci95_high": L_stats["ci95_high"],
        "L_p90": float(torch.quantile(L, 0.90)),
        "L_p99": float(torch.quantile(L, 0.99)),
        "num_images": total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-dir",
        default="data/ShanghaiTech/part_B/test_data/images",
    )
    parser.add_argument(
        "--gt-dir",
        default=None,
        help="Optional ground-truth directory; e.g. test_data/ground-truth",
    )
    parser.add_argument(
        "--base-checkpoint",
        default="data/partBmodel_best.pth",
    )
    parser.add_argument(
        "--noise-free-checkpoint",
        default="runs_crowd_noise_free/best.pth",
    )
    parser.add_argument("--output", default="crowd_noise_free_medium_results.csv")
    parser.add_argument("--count-scale", type=float, default=100.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--attack-eps", type=float, default=20.0)
    parser.add_argument("--attack-steps", type=int, default=10)
    parser.add_argument("--b-eps", type=float, default=8.0 / 255.0)
    parser.add_argument("--b-steps", type=int, default=3)
    parser.add_argument("--b-step-size", type=float, default=0.25)
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Noise-free crowd setting: sigma=0, N=1, no aggregator, C(x)=1")
    print(f"Count normalization: clip(count / {args.count_scale}, 0, 1)")

    if not os.path.exists(args.base_checkpoint):
        raise FileNotFoundError(f"Base checkpoint not found: {args.base_checkpoint}")
    if not os.path.exists(args.noise_free_checkpoint):
        raise FileNotFoundError(
            f"Noise-free checkpoint not found: {args.noise_free_checkpoint}"
        )

    model = CSRNet().to(device)
    load_checkpoint(model, args.base_checkpoint, device)
    load_checkpoint(model, args.noise_free_checkpoint, device)
    model.eval()

    dataset = CrowdDataset(
        image_dir=args.image_dir,
        gt_dir=args.gt_dir,
        count_scale=args.count_scale,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    print("\nEvaluation configuration:")
    print(f"  attack epsilon: {args.attack_eps}")
    print(f"  attack steps: {args.attack_steps}")
    print(f"  B radius: {args.b_eps}")
    print(f"  B ascent steps: {args.b_steps}")

    clean = evaluate(model, loader, device, attacked=False, args=args)
    attacked = evaluate(model, loader, device, attacked=True, args=args)

    rows = []
    for setting, result in (("Clean", clean), ("Attacked", attacked)):
        rows.append({
            "model": "noise_free",
            "setting": setting,
            "attack_eps": args.attack_eps,
            "attack_steps": args.attack_steps,
            "count_scale": args.count_scale,
            "b_eps": args.b_eps,
            **result,
        })

    result_df = pd.DataFrame(rows)
    print("\nNoise-free crowd results:")
    print(result_df.to_string(index=False))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    result_df.to_json(output_path.with_suffix(".json"), orient="records", indent=2)

    print(f"\nSaved CSV: {output_path}")
    print(f"Saved JSON: {output_path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
