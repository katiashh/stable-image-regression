# -*- coding: utf-8 -*-
"""Fair IQA runtime comparison: learned aggregation vs fixed Mean."""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_iqa_definitions(source_script):
    source_script = Path(source_script)
    if not source_script.exists():
        raise FileNotFoundError(f"Source IQA script not found: {source_script}")
    source = source_script.read_text(encoding="utf-8")
    marker = "\nmetric = MetricModel("
    if marker not in source:
        raise RuntimeError("Could not find the IQA model-loading marker.")
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

    # Warm up both methods before collecting measurements.
    for _ in range(warmup):
        run_pass(learned, images)
        run_pass(fixed, images)
    synchronize(device)

    values = {"learned_aggregation": [], "fixed_mean": []}
    for repeat in range(repeats):
        # Alternate order to reduce systematic bias from GPU clocking and
        # thermal/cache effects.
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


def load_learned_checkpoint(defended, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "base_state" in checkpoint:
        defended.base.load_state_dict(checkpoint["base_state"])
    if "agg_state" in checkpoint:
        defended.agg.load_state_dict(checkpoint["agg_state"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-script", default="scripts/evaluate_iqa.py")
    parser.add_argument("--data-dir", default="data/koniq")
    parser.add_argument("--model-path", default="data/KonCept512.pth")
    parser.add_argument(
        "--backbone-path",
        default="data/inceptionresnetv2-520b38e4.pth",
    )
    parser.add_argument(
        "--learned-checkpoint",
        default="best_model_8_255/last.pth",
    )
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--num-images", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", default="iqa_learned_vs_mean_runtime.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Runtime comparison: learned aggregation vs fixed Mean")
    print(f"N=1/64/128/256; sigma={args.sigma}")
    print(f"Images: {args.num_images}; warm-up: {args.warmup}; repeats: {args.repeats}")
    print("Resolution: 512 x 384")

    ns = load_iqa_definitions(args.source_script)
    MetricModel = ns["MetricModel"]
    SmoothedMetric = ns["SmoothedMetric"]
    NormalizeOnly = ns["NormalizeOnly"]
    MyCustomDataset = ns["MyCustomDataset"]

    dataset = MyCustomDataset(path_gt=args.data_dir, mode="test")
    if len(dataset) < args.num_images:
        raise RuntimeError(f"Only {len(dataset)} test images are available.")
    images = [
        dataset[i][0].unsqueeze(0).to(device)
        for i in range(args.num_images)
    ]
    print(f"Loaded {len(images)} IQA test images.")

    preprocess = NormalizeOnly(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ).to(device)

    all_rows = []
    for n in (1, 64, 128, 256):
        print(f"\nN={n}: constructing one shared base model")
        metric = MetricModel(
            device=device,
            model_path=args.model_path,
            backbone_path=args.backbone_path,
        )
        base = metric.model

        learned = SmoothedMetric(
            base_model=base,
            preprocess=preprocess,
            n_samples=n,
            sigma=args.sigma,
            aggregator_mode="deepsets",
            hidden=32,
        ).to(device)
        load_learned_checkpoint(learned, args.learned_checkpoint, device)

        fixed = SmoothedMetric(
            base_model=base,
            preprocess=preprocess,
            n_samples=n,
            sigma=args.sigma,
            aggregator_mode="mean",
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
                "task": "IQA",
                "N": n,
                "sigma": args.sigma,
                "resolution": "512x384",
            })
            all_rows.append(row)
            print(row)

        del fixed, learned, metric
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = pd.DataFrame(all_rows)
    columns = [
        "task", "method", "N", "sigma", "resolution",
        "time_total_mean_s", "time_total_std_s",
        "time_per_image_mean_s", "time_per_image_std_s", "fps",
        "repeats", "num_images",
    ]
    result = result[columns]
    print("\nIQA learned-vs-Mean runtime results:")
    print(result.to_string(index=False))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    result.to_json(output_path.with_suffix(".json"), orient="records", indent=2)
    print(f"Saved CSV: {output_path}")
    print(f"Saved JSON: {output_path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
