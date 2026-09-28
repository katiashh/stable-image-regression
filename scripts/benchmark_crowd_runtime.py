# -*- coding: utf-8 -*-
"""Fair ShanghaiTech runtime comparison: learned aggregation vs fixed Mean."""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


def load_crowd_definitions(source_script):
    source_script = Path(source_script)
    if not source_script.exists():
        raise FileNotFoundError(f"Source crowd script not found: {source_script}")
    source = source_script.read_text(encoding="utf-8")
    marker = "\nmodel = CSRNet()\nmodel = model.cuda()\ncheckpoint ="
    if marker not in source:
        raise RuntimeError("Could not find the crowd model-loading marker.")
    namespace = {"__file__": str(source_script)}
    prefix = source.split(marker, 1)[0]
    exec(compile(prefix, str(source_script), "exec"), namespace)
    return namespace


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_state(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint
    if isinstance(checkpoint, dict):
        for key in (
            "state_dict",
            "model_state",
            "base_state",
            "model_state_dict",
        ):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state = checkpoint[key]
                break
    state = {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state.items()
    }
    model.load_state_dict(state, strict=True)


def load_learned_checkpoint(defended, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "base_state" in checkpoint:
        defended.base.load_state_dict(checkpoint["base_state"], strict=True)
    if "agg_state" in checkpoint:
        defended.agg.load_state_dict(checkpoint["agg_state"], strict=True)


def load_test_images(image_dir, num_images, device):
    image_paths = sorted(Path(image_dir).glob("*.jpg"))[:num_images]
    if len(image_paths) < num_images:
        raise RuntimeError(
            f"Only {len(image_paths)} JPG images found in {image_dir}; "
            f"requested {num_images}."
        )
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    images = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        images.append(transform(image).unsqueeze(0).to(device))
    return images


class ChunkedMeanMetric(nn.Module):
    """Memory-safe smoothing with a fixed arithmetic mean."""

    def __init__(self, base, preprocess, n_samples, sigma, chunk_size):
        super().__init__()
        self.base = base
        self.pre = preprocess
        self.n = int(n_samples)
        self.sigma = float(sigma)
        self.chunk_size = int(chunk_size)

    def forward(self, x):
        batch_size = x.shape[0]
        predictions = []
        for start in range(0, self.n, self.chunk_size):
            m = min(self.chunk_size, self.n - start)
            noise = torch.randn(
                (m, *x.shape), device=x.device, dtype=x.dtype
            ) * self.sigma
            x_rep = (x.unsqueeze(0) + noise).clamp(0.0, 1.0)
            x_flat = self.pre(
                x_rep.reshape(m * batch_size, *x.shape[1:])
            )
            y_flat = self.base(x_flat)
            if y_flat.ndim > 1:
                y_flat = y_flat.reshape(-1)
            predictions.append(
                y_flat.reshape(m, batch_size).transpose(0, 1)
            )
        values = torch.cat(predictions, dim=1)
        return values.mean(dim=1), values


class ChunkedLearnedMetric(nn.Module):
    """Memory-safe learned smoothing with the same prediction set Y."""

    def __init__(self, base, preprocess, aggregator, n_samples, sigma, chunk_size):
        super().__init__()
        self.base = base
        self.pre = preprocess
        self.agg = aggregator
        self.n = int(n_samples)
        self.sigma = float(sigma)
        self.chunk_size = int(chunk_size)

    def forward(self, x):
        batch_size = x.shape[0]
        predictions = []
        for start in range(0, self.n, self.chunk_size):
            m = min(self.chunk_size, self.n - start)
            noise = torch.randn(
                (m, *x.shape), device=x.device, dtype=x.dtype
            ) * self.sigma
            x_rep = (x.unsqueeze(0) + noise).clamp(0.0, 1.0)
            x_flat = self.pre(
                x_rep.reshape(m * batch_size, *x.shape[1:])
            )
            y_flat = self.base(x_flat)
            if y_flat.ndim > 1:
                y_flat = y_flat.reshape(-1)
            predictions.append(
                y_flat.reshape(m, batch_size).transpose(0, 1)
            )
        values = torch.cat(predictions, dim=1)
        return self.agg(values), values


@torch.inference_mode()
def run_pass(model, images):
    for image in images:
        model(image)


def timed_pass(model, images, device):
    synchronize(device)
    start = time.perf_counter()
    run_pass(model, images)
    synchronize(device)
    return time.perf_counter() - start


def measure_pair(learned, fixed, images, device, warmup, repeats):
    learned.eval()
    fixed.eval()
    for _ in range(warmup):
        run_pass(learned, images)
        run_pass(fixed, images)
    synchronize(device)

    values = {"learned_aggregation": [], "fixed_mean": []}
    for repeat in range(repeats):
        if repeat % 2 == 0:
            order = [
                ("learned_aggregation", learned),
                ("fixed_mean", fixed),
            ]
        else:
            order = [
                ("fixed_mean", fixed),
                ("learned_aggregation", learned),
            ]
        for name, model in order:
            values[name].append(timed_pass(model, images, device))

    rows = []
    n_images = len(images)
    for name in ("learned_aggregation", "fixed_mean"):
        seconds = np.asarray(values[name], dtype=np.float64)
        total_mean = float(seconds.mean())
        total_std = float(seconds.std(ddof=1)) if len(seconds) > 1 else 0.0
        rows.append({
            "method": name,
            "time_total_mean_s": total_mean,
            "time_total_std_s": total_std,
            "time_per_image_mean_s": total_mean / n_images,
            "time_per_image_std_s": total_std / n_images,
            "fps": n_images / total_mean if total_mean > 0 else float("nan"),
            "repeats": int(repeats),
            "num_images": int(n_images),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-script", default="scripts/evaluate_crowd.py")
    parser.add_argument(
        "--image-dir",
        default="data/ShanghaiTech/part_B/test_data/images",
    )
    parser.add_argument(
        "--base-checkpoint",
        default="data/partBmodel_best.pth",
    )
    parser.add_argument(
        "--learned-checkpoint",
        default="runs_crowd_agg_8_255/best.pth",
    )
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--num-images", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", default="crowd_learned_vs_mean_runtime.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be at least 1")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Runtime comparison: learned aggregation vs fixed Mean")
    print(f"N=1/64/128/256; sigma={args.sigma}")
    print(f"Images: {args.num_images}; warm-up: {args.warmup}; repeats: {args.repeats}")
    print("Resolution: native ShanghaiTech Part B")
    print(f"Chunk size: {args.chunk_size}")

    for path in (args.base_checkpoint, args.learned_checkpoint):
        if not Path(path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

    ns = load_crowd_definitions(args.source_script)
    CSRNet = ns["CSRNet"]
    SmoothedMetric = ns["SmoothedMetric"]
    images = load_test_images(args.image_dir, args.num_images, device)
    print(f"Loaded {len(images)} crowd test images.")

    all_rows = []
    for n in (1, 64, 128, 256):
        print(f"\nN={n}: constructing one shared base model")
        base = CSRNet(load_weights=True).to(device)
        load_state(base, args.base_checkpoint, device)
        base.eval()

        preprocess = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        learned_template = SmoothedMetric(
            base_model=base,
            preprocess=preprocess,
            n_samples=1,
            sigma=args.sigma,
            aggregator_mode="deepsets",
            hidden=32,
            psi_depth=3,
            rho_depth=3,
        ).to(device)
        load_learned_checkpoint(
            learned_template,
            args.learned_checkpoint,
            device,
        )

        learned = ChunkedLearnedMetric(
            learned_template.base,
            learned_template.pre,
            learned_template.agg,
            n,
            args.sigma,
            args.chunk_size,
        ).to(device)

        fixed = ChunkedMeanMetric(
            base=base,
            preprocess=preprocess,
            n_samples=n,
            sigma=args.sigma,
            chunk_size=args.chunk_size,
        ).to(device)

        pair_rows = measure_pair(
            learned,
            fixed,
            images,
            device,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        for row in pair_rows:
            row.update({
                "task": "Crowd counting",
                "N": n,
                "sigma": args.sigma,
                "resolution": "native",
                "chunk_size": args.chunk_size,
            })
            all_rows.append(row)
            print(row)

        del fixed, learned, learned_template, base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = pd.DataFrame(all_rows)
    columns = [
        "task", "method", "N", "sigma", "resolution", "chunk_size",
        "time_total_mean_s", "time_total_std_s",
        "time_per_image_mean_s", "time_per_image_std_s", "fps",
        "repeats", "num_images",
    ]
    result = result[columns]
    print("\nCrowd learned-vs-Mean runtime results:")
    print(result.to_string(index=False))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    result.to_json(output_path.with_suffix(".json"), orient="records", indent=2)
    print(f"Saved CSV: {output_path}")
    print(f"Saved JSON: {output_path.with_suffix('.json')}")
if __name__ == "__main__":
    main()
