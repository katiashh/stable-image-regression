# -*- coding: utf-8 -*-
"""Fair age-estimation runtime comparison: learned aggregation vs fixed Mean."""

import argparse
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def load_age_definitions(source_script):
    source_script = Path(source_script)
    if not source_script.exists():
        raise FileNotFoundError(f"Source age script not found: {source_script}")
    source = source_script.read_text(encoding="utf-8")
    marker = "\nbase_model = SSRNet(image_size=input_size).to(device)"
    if marker not in source:
        raise RuntimeError("Could not find the age checkpoint-loading marker.")

    # The age model contains view(...) calls after feature fusion. Replacing
    # them in memory avoids failures on non-contiguous tensors; the source
    # file itself is not modified.
    source = source.replace(".view(", ".reshape(")
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
        for key in ("state_dict", "model_state", "model_state_dict"):
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


def load_test_images(image_dir, num_images, image_size, device):
    image_dir = Path(image_dir)
    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )[:num_images]
    if len(image_paths) < num_images:
        raise RuntimeError(
            f"Only {len(image_paths)} images found in {image_dir}; "
            f"requested {num_images}."
        )

    images = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Cannot read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(
            image,
            (image_size, image_size),
            interpolation=cv2.INTER_AREA,
        )
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        images.append(image.unsqueeze(0).to(device, non_blocking=True))
    return images


class ChunkedMeanMetric(nn.Module):
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
        return values.mean(dim=1)


class ChunkedLearnedMetric(nn.Module):
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
        return self.agg(values)


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
    parser.add_argument("--source-script", default="scripts/evaluate_age.py")
    parser.add_argument(
        "--image-dir",
        default="data/wiki_crop/age_split/test/images",
    )
    parser.add_argument(
        "--base-checkpoint",
        default="trained_models/ssrnet_best.pth",
    )
    parser.add_argument(
        "--learned-checkpoint",
        default="runs_age_agg_8/best.pth",
    )
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--num-images", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", default="age_learned_vs_mean_runtime.csv")
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
    print(f"Resolution: {args.image_size} x {args.image_size}")
    print(f"Chunk size: {args.chunk_size}")

    for path in (args.base_checkpoint, args.learned_checkpoint):
        if not Path(path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

    ns = load_age_definitions(args.source_script)
    SSRNet = ns["SSRNet"]
    AgePreprocess = ns["AgePreprocess"]
    SmoothedMetric = ns["SmoothedMetric"]

    images = load_test_images(
        args.image_dir,
        args.num_images,
        args.image_size,
        device,
    )
    print(f"Loaded {len(images)} age test images.")

    all_rows = []
    for n in (1, 64, 128, 256):
        print(f"\nN={n}: constructing one shared base model")
        base = SSRNet(image_size=args.image_size).to(device)
        load_state(base, args.base_checkpoint, device)
        base.eval()

        preprocess = AgePreprocess(
            size=(args.image_size, args.image_size),
        ).to(device)

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
            base=learned_template.base,
            preprocess=learned_template.pre,
            aggregator=learned_template.agg,
            n_samples=n,
            sigma=args.sigma,
            chunk_size=args.chunk_size,
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
                "task": "Age estimation",
                "N": n,
                "sigma": args.sigma,
                "resolution": f"{args.image_size}x{args.image_size}",
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
    print("\nAge learned-vs-Mean runtime results:")
    print(result.to_string(index=False))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    result.to_json(output_path.with_suffix(".json"), orient="records", indent=2)
    print(f"Saved CSV: {output_path}")
    print(f"Saved JSON: {output_path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
