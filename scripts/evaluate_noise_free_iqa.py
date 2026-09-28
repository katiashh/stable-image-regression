# -*- coding: utf-8 -*-
"""
Evaluation of the noise-free IQA baseline.

The script evaluates:
  1. MSE on clean images;
  2. MSE on adversarially attacked images;
  3. empirical local stability B(x) on clean images;
  4. empirical local stability B(x) on attacked images.

The original IQA model definitions are loaded from the previous evaluation
script without executing its old evaluation code. By default the source file
is adv_iqa_test_8(4).py.

For the noise-free baseline there is no noisy prediction set and no
aggregator. We therefore use C(x)=1 and compute

    B(x) = eps_B * L(x),

where L(x) is the maximum input-gradient norm found by a short local ascent
procedure. This is the deterministic analogue of the empirical diagnostic
used for the complete method.
"""

import argparse
import ast
import csv
import json
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
from torchvision import transforms
from tqdm import tqdm


def load_model_definitions(source_path):
    """Load imports/classes/functions from the old script only.

    The old script contains an evaluation block at module level. Executing
    only definitions prevents that old experiment from running twice.
    """
    source = Path(source_path).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))

    allowed = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
            allowed.append(node)

    namespace = {"__name__": "iqa_model_definitions"}
    module = ast.Module(body=allowed, type_ignores=[])
    compiled = compile(module, filename=str(source_path), mode="exec")
    exec(compiled, namespace)
    return namespace


class KonIQTestDataset(Dataset):
    """The same KonIQ split protocol as in the previous IQA script."""

    def __init__(self, image_dir, scores_csv, split="test"):
        self.image_dir = Path(image_dir)
        self.split = split

        all_images = sorted(
            p.name
            for p in self.image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

        if split == "train":
            split_images = all_images[:8000]
        elif split == "val":
            split_images = all_images[8000:9000]
        elif split == "test":
            split_images = all_images[9000:]
        else:
            raise ValueError("split must be train, val, or test")

        scores = pd.read_csv(scores_csv)
        score_map = {
            str(row["image_name"]): float(row["MOS_zscore"])
            for _, row in scores.iterrows()
        }

        self.items = []
        skipped = []
        for filename in split_images:
            path = self.image_dir / filename
            if filename not in score_map:
                continue

            # Filter files that cannot be read before DataLoader workers run.
            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception:
                skipped.append(filename)
                continue

            self.items.append((path, score_map[filename] / 100.0))

        print(
            f"[{split}] usable images: {len(self.items)}; "
            f"skipped unreadable images: {len(skipped)}"
        )
        if skipped:
            print("First skipped images:")
            for filename in skipped[:10]:
                print(f"  - {filename}")

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, target = self.items[index]

        with Image.open(path) as image:
            image = image.convert("RGB")
            image = np.asarray(image).astype(np.float32)

        image = cv2.resize(
            image,
            dsize=(384, 512),
            interpolation=cv2.INTER_AREA,
        )
        image = image / 255.0
        x = self.to_tensor(image)
        y = torch.tensor(target, dtype=torch.float32)
        return x, y


class NormalizeOnly(nn.Module):
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


def load_noise_free_checkpoint(model, checkpoint_path, device):
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


def deterministic_prediction(base_model, preprocess, x):
    return base_model(preprocess(x)).view(-1)


def pgd_l2_attack(
    base_model,
    preprocess,
    x,
    y,
    eps=1.0,
    steps=10,
    step_size=None,
):
    """Untargeted L2 PGD attack that maximizes prediction MSE."""
    if step_size is None:
        step_size = eps / steps

    x0 = x.detach()
    delta = torch.zeros_like(x0)

    for _ in range(steps):
        delta.requires_grad_(True)
        x_adv = (x0 + delta).clamp(0.0, 1.0)
        pred = deterministic_prediction(base_model, preprocess, x_adv)
        loss = F.mse_loss(pred, y, reduction="sum")
        grad = torch.autograd.grad(loss, delta, only_inputs=True)[0]

        grad_norm = grad.flatten(1).norm(p=2, dim=1).view(-1, 1, 1, 1)
        direction = grad / (grad_norm + 1e-12)
        delta = delta.detach() + step_size * direction

        delta_flat = delta.flatten(1)
        delta_norm = delta_flat.norm(p=2, dim=1, keepdim=True)
        factor = torch.clamp(eps / (delta_norm + 1e-12), max=1.0)
        delta = (delta_flat * factor).view_as(delta)
        delta = torch.clamp(x0 + delta, 0.0, 1.0) - x0

    return (x0 + delta).clamp(0.0, 1.0).detach()


def local_l2_deterministic(
    base_model,
    preprocess,
    x,
    radius,
    steps=3,
    step_size=0.25,
):
    """Estimate max ||grad_x f(x+u)||_2 in an L2 ball around x."""
    batch_size = x.size(0)
    L_max = torch.zeros(batch_size, device=x.device, dtype=torch.float32)
    base = x.detach()
    u = torch.zeros_like(base)

    for _ in range(steps):
        z = (base + u).clamp(0.0, 1.0).detach().requires_grad_(True)
        output = deterministic_prediction(base_model, preprocess, z)
        grad = torch.autograd.grad(output.sum(), z, only_inputs=True)[0]

        grad_norm = grad.flatten(1).norm(p=2, dim=1).view(-1, 1, 1, 1)
        direction = grad / (grad_norm + 1e-12)
        z_adv = (z + step_size * radius * direction).clamp(0.0, 1.0)

        u = z_adv.detach() - base
        u_flat = u.flatten(1)
        u_norm = u_flat.norm(p=2, dim=1, keepdim=True)
        factor = torch.clamp(radius / (u_norm + 1e-12), max=1.0)
        u = (u_flat * factor).view_as(u)

    z = (base + u).clamp(0.0, 1.0).detach().requires_grad_(True)
    output = deterministic_prediction(base_model, preprocess, z)
    grad = torch.autograd.grad(output.sum(), z, only_inputs=True)[0]
    L_max = grad.flatten(1).norm(p=2, dim=1).float()
    return L_max.detach()


def quantile(tensor, probability):
    if tensor.numel() == 0:
        return float("nan")
    return float(torch.quantile(tensor, probability).item())


def mean_std_ci95(values):
    """Statistics of per-image values for a mean and its 95% CI."""
    values = values.detach().cpu().float().flatten()
    n = int(values.numel())

    if n == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "n": 0,
        }

    mean = float(values.mean().item())
    std = float(values.std(unbiased=(n > 1)).item()) if n > 1 else 0.0
    standard_error = std / np.sqrt(n)
    half_width = 1.96 * standard_error

    return {
        "mean": mean,
        "std": std,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "n": n,
    }


def evaluate_split(
    base_model,
    preprocess,
    loader,
    device,
    attack=False,
    attack_eps=1.0,
    attack_steps=10,
    b_eps=8.0 / 255.0,
    b_steps=3,
    b_step_size=0.25,
    max_batches=None,
):
    base_model.eval()
    mse_sum = 0.0
    total = 0
    all_squared_errors = []
    all_B = []
    all_L = []

    iterator = enumerate(tqdm(loader, desc="Attacked" if attack else "Clean"))
    for batch_index, (x, y) in iterator:
        if max_batches is not None and batch_index >= max_batches:
            break

        x = x.to(device).float()
        y = y.to(device).float().view(-1)

        if attack:
            x_eval = pgd_l2_attack(
                base_model,
                preprocess,
                x,
                y,
                eps=attack_eps,
                steps=attack_steps,
            )
        else:
            x_eval = x

        with torch.no_grad():
            pred = deterministic_prediction(base_model, preprocess, x_eval)

        squared_errors = (pred - y).pow(2).detach().cpu()
        all_squared_errors.append(squared_errors)
        mse_sum += squared_errors.sum().item()
        total += x_eval.size(0)

        # There is no aggregator in the noise-free baseline, hence C(x)=1.
        L = local_l2_deterministic(
            base_model,
            preprocess,
            x_eval,
            radius=b_eps,
            steps=b_steps,
            step_size=b_step_size,
        )
        B = b_eps * L
        all_L.append(L.cpu())
        all_B.append(B.cpu())

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    L = torch.cat(all_L) if all_L else torch.empty(0)
    B = torch.cat(all_B) if all_B else torch.empty(0)
    squared_errors = (
        torch.cat(all_squared_errors)
        if all_squared_errors
        else torch.empty(0)
    )

    mse_stats = mean_std_ci95(squared_errors)
    b_stats = mean_std_ci95(B)
    l_stats = mean_std_ci95(L)

    return {
        "MSE": mse_stats["mean"],
        "MSE_std": mse_stats["std"],
        "MSE_ci95_low": mse_stats["ci95_low"],
        "MSE_ci95_high": mse_stats["ci95_high"],
        "B_mean": b_stats["mean"],
        "B_std": b_stats["std"],
        "B_ci95_low": b_stats["ci95_low"],
        "B_ci95_high": b_stats["ci95_high"],
        "B_p90": quantile(B, 0.90),
        "B_p99": quantile(B, 0.99),
        "L_mean": l_stats["mean"],
        "L_std": l_stats["std"],
        "L_ci95_low": l_stats["ci95_low"],
        "L_ci95_high": l_stats["ci95_high"],
        "L_p90": quantile(L, 0.90),
        "L_p99": quantile(L, 0.99),
        "num_images": int(total),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="scripts/evaluate_iqa.py")
    parser.add_argument("--image-dir", default="data/koniq")
    parser.add_argument(
        "--scores-csv",
        default="data/koniq10k_scores_and_distributions.csv",
    )
    parser.add_argument(
        "--noise-free-checkpoint",
        default="runs_iqa_noise_free/best.pth",
    )
    parser.add_argument("--backbone-checkpoint", default="data/inceptionresnetv2-520b38e4.pth")
    parser.add_argument("--base-checkpoint", default="data/KonCept512.pth")
    parser.add_argument("--output", default="iqa_noise_free_medium_results.csv")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--attack-eps", type=float, default=1.0)
    parser.add_argument("--attack-steps", type=int, default=10)
    parser.add_argument("--b-eps", type=float, default=8.0 / 255.0)
    parser.add_argument("--b-steps", type=int, default=3)
    parser.add_argument("--b-step-size", type=float, default=0.25)
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Noise-free evaluation: N=1, sigma=0, no aggregator, C(x)=1")

    definitions = load_model_definitions(args.source)
    MetricModel = definitions["MetricModel"]

    metric = MetricModel(
        device=device,
        model_path=args.base_checkpoint,
        backbone_path=args.backbone_checkpoint,
    )
    base_model = metric.model.to(device)
    preprocess = NormalizeOnly().to(device)

    if not os.path.exists(args.noise_free_checkpoint):
        raise FileNotFoundError(
            f"Noise-free checkpoint was not found: {args.noise_free_checkpoint}"
        )

    load_noise_free_checkpoint(
        base_model,
        args.noise_free_checkpoint,
        device,
    )
    base_model.eval()

    dataset = KonIQTestDataset(
        image_dir=args.image_dir,
        scores_csv=args.scores_csv,
        split=args.split,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    print("\nEvaluation configuration:")
    print(f"  split: {args.split}")
    print(f"  attack epsilon: {args.attack_eps}")
    print(f"  attack steps: {args.attack_steps}")
    print(f"  B radius: {args.b_eps}")
    print(f"  B ascent steps: {args.b_steps}")

    clean = evaluate_split(
        base_model,
        preprocess,
        loader,
        device,
        attack=False,
        b_eps=args.b_eps,
        b_steps=args.b_steps,
        b_step_size=args.b_step_size,
        max_batches=args.max_batches,
    )
    attacked = evaluate_split(
        base_model,
        preprocess,
        loader,
        device,
        attack=True,
        attack_eps=args.attack_eps,
        attack_steps=args.attack_steps,
        b_eps=args.b_eps,
        b_steps=args.b_steps,
        b_step_size=args.b_step_size,
        max_batches=args.max_batches,
    )

    rows = []
    for setting, result in (("Clean", clean), ("Attacked", attacked)):
        rows.append(
            {
                "model": "noise_free",
                "setting": setting,
                "split": args.split,
                "attack_eps": args.attack_eps,
                "attack_steps": args.attack_steps,
                "b_eps": args.b_eps,
                "MSE": result["MSE"],
                "MSE_std": result["MSE_std"],
                "MSE_ci95_low": result["MSE_ci95_low"],
                "MSE_ci95_high": result["MSE_ci95_high"],
                "B_mean": result["B_mean"],
                "B_std": result["B_std"],
                "B_ci95_low": result["B_ci95_low"],
                "B_ci95_high": result["B_ci95_high"],
                "B_p90": result["B_p90"],
                "B_p99": result["B_p99"],
                "L_mean": result["L_mean"],
                "L_std": result["L_std"],
                "L_ci95_low": result["L_ci95_low"],
                "L_ci95_high": result["L_ci95_high"],
                "L_p90": result["L_p90"],
                "L_p99": result["L_p99"],
                "num_images": result["num_images"],
            }
        )

    print("\nNoise-free IQA results:")
    print(pd.DataFrame(rows).to_string(index=False))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)

    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"\nSaved CSV: {output_path}")
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
