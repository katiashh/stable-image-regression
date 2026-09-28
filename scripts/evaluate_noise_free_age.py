# -*- coding: utf-8 -*-
"""
Noise-free robustness evaluation for age estimation with SSR-Net.

The script uses the same SSR-Net implementation as train_age_noise_free.py.
Raw age predictions and targets are divided by 100 for evaluation, so the
reported regression values are in [0, 1].

The noise-free model has N=1 and no prediction aggregator. Therefore:

    C(x) = 1,       B(x) = eps_B * L(x).

The script reports Clean and Attacked MSE, standard deviations, 95% CIs,
and empirical local stability statistics.
"""

import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Reuse exactly the architecture and preprocessing used for noise-free training.
from train_age_noise_free import AgePreprocess, SSRNet, load_checkpoint


class AgeTestDataset(Dataset):
    def __init__(self, csv_path, image_dir, image_size=64):
        self.image_dir = Path(image_dir)
        df = pd.read_csv(csv_path).dropna(subset=["filename", "age"])
        self.items = []
        skipped = []

        for _, row in df.iterrows():
            filename = str(row["filename"])
            image_path = self.image_dir / filename

            if not image_path.is_file():
                skipped.append((filename, "file not found"))
                continue

            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception as exc:
                skipped.append((filename, str(exc)))
                continue

            self.items.append((image_path, float(row["age"])))

        self.image_size = int(image_size)
        print(
            f"[{self.image_dir}] usable images: {len(self.items)}; "
            f"skipped: {len(skipped)}"
        )
        if skipped:
            print("First skipped files:")
            for filename, reason in skipped[:10]:
                print(f"  - {filename}: {reason}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        image_path, raw_age = self.items[index]

        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA,
        )
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()

        # The model is trained on raw ages; metrics use age/100 in [0, 1].
        target = torch.tensor(
            np.clip(raw_age / 100.0, 0.0, 1.0),
            dtype=torch.float32,
        )
        return image, target, image_path.name


class NoiseFreeAgeModel(nn.Module):
    def __init__(self, base_model, preprocess):
        super().__init__()
        self.base = base_model
        self.pre = preprocess

    def forward(self, x):
        raw_age = self.base(self.pre(x)).view(-1)
        return torch.clamp(raw_age / 100.0, min=0.0, max=1.0)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stats(values):
    values = values.detach().cpu().float().flatten()
    n = int(values.numel())
    if n == 0:
        return {"mean": np.nan, "std": np.nan, "ci95_low": np.nan, "ci95_high": np.nan}

    mean = float(values.mean())
    std = float(values.std(unbiased=n > 1)) if n > 1 else 0.0
    half_width = 1.96 * std / np.sqrt(n)
    return {
        "mean": mean,
        "std": std,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def pgd_l2_attack(model, x, y, eps=0.5, steps=10, step_size=None):
    """Untargeted L2-PGD in the raw [0,1] image space."""
    if step_size is None:
        step_size = eps / steps

    x0 = x.detach()
    delta = torch.zeros_like(x0)

    for _ in range(steps):
        delta.requires_grad_(True)
        x_adv = (x0 + delta).clamp(0.0, 1.0)
        pred = model(x_adv)
        loss = F.mse_loss(pred, y, reduction="sum")
        grad = torch.autograd.grad(loss, delta, only_inputs=True)[0]

        grad_norm = grad.flatten(1).norm(p=2, dim=1).view(-1, 1, 1, 1)
        direction = grad / (grad_norm + 1e-12)
        delta = delta.detach() + step_size * direction

        delta_flat = delta.flatten(1)
        delta_norm = delta_flat.norm(p=2, dim=1, keepdim=True)
        factor = torch.clamp(eps / (delta_norm + 1e-12), max=1.0)
        delta = (delta_flat * factor).view_as(delta)
        delta = (x0 + delta).clamp(0.0, 1.0) - x0

    return (x0 + delta).clamp(0.0, 1.0).detach()


def local_l2_diagnostic(model, x, radius, steps=3, step_size=0.25):
    """Empirical L(x) = max ||grad_x f(x+u)||_2 in an L2 ball."""
    x0 = x.detach()
    u = torch.zeros_like(x0)

    for _ in range(steps):
        z = (x0 + u).clamp(0.0, 1.0).detach().requires_grad_(True)
        output = model(z)
        grad = torch.autograd.grad(output.sum(), z, only_inputs=True)[0]

        grad_norm = grad.flatten(1).norm(p=2, dim=1).view(-1, 1, 1, 1)
        direction = grad / (grad_norm + 1e-12)
        z_adv = (z + step_size * radius * direction).clamp(0.0, 1.0)

        u = z_adv.detach() - x0
        u_flat = u.flatten(1)
        u_norm = u_flat.norm(p=2, dim=1, keepdim=True)
        factor = torch.clamp(radius / (u_norm + 1e-12), max=1.0)
        u = (u_flat * factor).view_as(u)

    z = (x0 + u).clamp(0.0, 1.0).detach().requires_grad_(True)
    output = model(z)
    grad = torch.autograd.grad(output.sum(), z, only_inputs=True)[0]
    return grad.flatten(1).norm(p=2, dim=1).detach().float()


def evaluate(model, loader, device, attacked, args):
    model.eval()
    all_squared_errors = []
    all_B = []
    all_L = []
    total = 0

    description = "Attacked" if attacked else "Clean"
    for batch_index, (x, y, filenames) in enumerate(
        tqdm(loader, desc=description)
    ):
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
            prediction = model(x_eval)

        all_squared_errors.append((prediction - y).pow(2).cpu())

        # Noise-free model: C(x)=1, hence B(x)=eps_B * L(x).
        L = local_l2_diagnostic(
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

    if not all_squared_errors:
        raise RuntimeError("No images were evaluated.")

    squared_errors = torch.cat(all_squared_errors)
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
    parser.add_argument("--base-dir", default="data/wiki_crop/age_split")
    parser.add_argument("--checkpoint", default="runs_age_noise_free/best.pth")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--output", default="age_noise_free_results.csv")
    parser.add_argument("--attack-eps", type=float, default=0.5)
    parser.add_argument("--attack-steps", type=int, default=10)
    parser.add_argument("--b-eps", type=float, default=8.0 / 255.0)
    parser.add_argument("--b-steps", type=int, default=3)
    parser.add_argument("--b-step-size", type=float, default=0.25)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Noise-free age setting: sigma=0, N=1, no aggregator, C(x)=1")
    print("Age normalization: clip(age / 100, 0, 1)")

    base_dir = Path(args.base_dir)
    test_dir = base_dir / "test" / "images"
    test_csv = base_dir / "test" / "labels.csv"

    if not test_dir.exists():
        raise FileNotFoundError(f"Test image directory not found: {test_dir}")
    if not test_csv.exists():
        raise FileNotFoundError(f"Test labels file not found: {test_csv}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    dataset = AgeTestDataset(
        csv_path=test_csv,
        image_dir=test_dir,
        image_size=args.image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    base_model = SSRNet(image_size=args.image_size).to(device)
    load_checkpoint(base_model, args.checkpoint, device)
    preprocess = AgePreprocess(size=(args.image_size, args.image_size)).to(device)
    model = NoiseFreeAgeModel(base_model, preprocess).to(device)
    model.eval()

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
            "age_scale": 100.0,
            "b_eps": args.b_eps,
            **result,
        })

    result_df = pd.DataFrame(rows)
    print("\nNoise-free age results:")
    print(result_df.to_string(index=False))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    result_df.to_json(output_path.with_suffix(".json"), orient="records", indent=2)

    print(f"\nSaved CSV: {output_path}")
    print(f"Saved JSON: {output_path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
