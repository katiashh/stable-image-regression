# -*- coding: utf-8 -*-
"""
Noise-free baseline for the age-estimation experiment.

This script trains the same SSR-Net backbone as the noisy method, but without:
  - Gaussian input noise;
  - Monte-Carlo samples N;
  - prediction aggregation;
  - gradient regularization;
  - sensitivity regularization;
  - computation of B(x).

The train/validation/test split and the standard training augmentations are
kept unchanged so that this is an ablation of the noise-based method.
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
from torch.nn import init
from torch.utils.data import DataLoader, Dataset, default_collate
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class CustomAgeDataset(Dataset):
    def __init__(self, df, image_dir, image_size=64, augment=False, split_name="dataset"):
        self.image_dir = image_dir
        self.image_size = image_size
        self.augment = augment
        self.split_name = split_name

        # Filter missing or unreadable images before creating the DataLoader.
        # Returning None from __getitem__ would break PyTorch's default collate.
        valid_rows = []
        skipped = []

        for _, row in df.reset_index(drop=True).iterrows():
            filename = str(row["filename"])
            img_path = os.path.join(self.image_dir, filename)

            if not os.path.isfile(img_path):
                skipped.append((filename, "file not found"))
                continue

            img = cv2.imread(img_path)
            if img is None:
                skipped.append((filename, "unreadable image"))
                continue

            valid_rows.append(row)

        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)

        print(
            f"[{self.split_name}] usable images: {len(self.df)}; "
            f"skipped images: {len(skipped)}"
        )

        if skipped:
            print(f"[{self.split_name}] first skipped files:")
            for filename, reason in skipped[:10]:
                print(f"  - {filename}: {reason}")

    def __len__(self):
        return len(self.df)

    def _augment(self, img):
        # img: HWC, RGB, uint8
        # These are the same augmentations as in the original training script.
        if np.random.rand() < 0.5:
            img = np.fliplr(img).copy()

        if np.random.rand() < 0.3:
            alpha = np.random.uniform(0.9, 1.1)
            beta = np.random.uniform(-10, 10)
            img = np.clip(img * alpha + beta, 0, 255).astype(np.uint8)

        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = str(row["filename"])
        age = float(row["age"])

        img_path = os.path.join(self.image_dir, filename)

        # Check existence immediately before reading. This protects against
        # files disappearing from a shared filesystem after the initial scan.
        if not os.path.isfile(img_path):
            return None

        img = cv2.imread(img_path)
        if img is None:
            # A file can disappear or become temporarily unreadable after the
            # initial dataset scan, especially on a shared filesystem. Return
            # None and remove this sample in safe_collate instead of stopping
            # the whole training process.
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))

        if self.augment:
            img = self._augment(img)

        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW

        return (
            torch.tensor(img, dtype=torch.float32),
            torch.tensor(age, dtype=torch.float32),
        )


def safe_collate(batch):
    """Drop samples that cannot be read by a DataLoader worker."""
    batch = [sample for sample in batch if sample is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)


# -----------------------------------------------------------------------------
# SSR-Net backbone
# -----------------------------------------------------------------------------
class SSRNet(nn.Module):
    def __init__(
        self,
        stage_num=[3, 3, 3],
        image_size=64,
        class_range=101,
        lambda_index=1.0,
        lambda_delta=1.0,
    ):
        super().__init__()
        self.image_size = image_size
        self.stage_num = stage_num
        self.lambda_index = lambda_index
        self.lambda_delta = lambda_delta
        self.class_range = class_range

        self.stream1_stage3 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
        )
        self.stream1_stage2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
        )
        self.stream1_stage1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.stream2_stage3 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.MaxPool2d(2, 2),
        )
        self.stream2_stage2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.MaxPool2d(2, 2),
        )
        self.stream2_stage1 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
        )

        self.funsion_block_stream1_stage_3_before_PB = nn.Sequential(
            nn.Conv2d(32, 10, 1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(8, 8),
        )
        self.funsion_block_stream1_stage_3_prediction_block = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(10 * 4 * 4, self.stage_num[2]),
            nn.ReLU(),
        )
        self.funsion_block_stream1_stage_2_before_PB = nn.Sequential(
            nn.Conv2d(32, 10, 1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(4, 4),
        )
        self.funsion_block_stream1_stage_2_prediction_block = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(10 * 4 * 4, self.stage_num[1]),
            nn.ReLU(),
        )
        self.funsion_block_stream1_stage_1_before_PB = nn.Sequential(
            nn.Conv2d(32, 10, 1, padding=0),
            nn.ReLU(),
        )
        self.funsion_block_stream1_stage_1_prediction_block = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(10 * 8 * 8, self.stage_num[0]),
            nn.ReLU(),
        )

        self.funsion_block_stream2_stage_3_before_PB = nn.Sequential(
            nn.Conv2d(16, 10, 1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(8, 8),
        )
        self.funsion_block_stream2_stage_3_prediction_block = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(10 * 4 * 4, self.stage_num[2]),
            nn.ReLU(),
        )
        self.funsion_block_stream2_stage_2_before_PB = nn.Sequential(
            nn.Conv2d(16, 10, 1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(4, 4),
        )
        self.funsion_block_stream2_stage_2_prediction_block = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(10 * 4 * 4, self.stage_num[1]),
            nn.ReLU(),
        )
        self.funsion_block_stream2_stage_1_before_PB = nn.Sequential(
            nn.Conv2d(16, 10, 1, padding=0),
            nn.ReLU(),
        )
        self.funsion_block_stream2_stage_1_prediction_block = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(10 * 8 * 8, self.stage_num[0]),
            nn.ReLU(),
        )

        self.stage3_FC_after_PB = nn.Sequential(
            nn.Linear(self.stage_num[0], 2 * self.stage_num[0]),
            nn.ReLU(),
        )
        self.stage3_prob = nn.Sequential(
            nn.Linear(2 * self.stage_num[0], self.stage_num[0]),
            nn.ReLU(),
        )
        self.stage3_index_offsets = nn.Sequential(
            nn.Linear(2 * self.stage_num[0], self.stage_num[0]),
            nn.Tanh(),
        )
        self.stage3_delta_k = nn.Sequential(
            nn.Linear(10 * 4 * 4, 1),
            nn.Tanh(),
        )

        self.stage2_FC_after_PB = nn.Sequential(
            nn.Linear(self.stage_num[0], 2 * self.stage_num[0]),
            nn.ReLU(),
        )
        self.stage2_prob = nn.Sequential(
            nn.Linear(2 * self.stage_num[0], self.stage_num[0]),
            nn.ReLU(),
        )
        self.stage2_index_offsets = nn.Sequential(
            nn.Linear(2 * self.stage_num[0], self.stage_num[0]),
            nn.Tanh(),
        )
        self.stage2_delta_k = nn.Sequential(
            nn.Linear(10 * 4 * 4, 1),
            nn.Tanh(),
        )

        self.stage1_FC_after_PB = nn.Sequential(
            nn.Linear(self.stage_num[0], 2 * self.stage_num[0]),
            nn.ReLU(),
        )
        self.stage1_prob = nn.Sequential(
            nn.Linear(2 * self.stage_num[0], self.stage_num[0]),
            nn.ReLU(),
        )
        self.stage1_index_offsets = nn.Sequential(
            nn.Linear(2 * self.stage_num[0], self.stage_num[0]),
            nn.Tanh(),
        )
        self.stage1_delta_k = nn.Sequential(
            nn.Linear(10 * 8 * 8, 1),
            nn.Tanh(),
        )

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, image_):
        s1_3 = self.stream1_stage3(image_)
        s1_2 = self.stream1_stage2(s1_3)
        s1_1 = self.stream1_stage1(s1_2)

        s2_3 = self.stream2_stage3(image_)
        s2_2 = self.stream2_stage2(s2_3)
        s2_1 = self.stream2_stage1(s2_2)

        s1_3_pb = self.funsion_block_stream1_stage_3_before_PB(s1_3)
        s1_2_pb = self.funsion_block_stream1_stage_2_before_PB(s1_2)
        s1_1_pb = self.funsion_block_stream1_stage_1_before_PB(s1_1)
        s2_3_pb = self.funsion_block_stream2_stage_3_before_PB(s2_3)
        s2_2_pb = self.funsion_block_stream2_stage_2_before_PB(s2_2)
        s2_1_pb = self.funsion_block_stream2_stage_1_before_PB(s2_1)

        e1_3 = s1_3_pb.view(s1_3_pb.size(0), -1)
        e1_2 = s1_2_pb.view(s1_2_pb.size(0), -1)
        e1_1 = s1_1_pb.view(s1_1_pb.size(0), -1)
        e2_3 = s2_3_pb.view(s2_3_pb.size(0), -1)
        e2_2 = s2_2_pb.view(s2_2_pb.size(0), -1)
        e2_1 = s2_1_pb.view(s2_1_pb.size(0), -1)

        d1 = self.stage1_delta_k(torch.mul(e1_1, e2_1))
        d2 = self.stage2_delta_k(torch.mul(e1_2, e2_2))
        d3 = self.stage3_delta_k(torch.mul(e1_3, e2_3))

        z1 = torch.mul(
            self.funsion_block_stream1_stage_1_prediction_block(e1_1),
            self.funsion_block_stream2_stage_1_prediction_block(e2_1),
        )
        z2 = torch.mul(
            self.funsion_block_stream1_stage_2_prediction_block(e1_2),
            self.funsion_block_stream2_stage_2_prediction_block(e2_2),
        )
        z3 = torch.mul(
            self.funsion_block_stream1_stage_3_prediction_block(e1_3),
            self.funsion_block_stream2_stage_3_prediction_block(e2_3),
        )

        z1 = self.stage1_FC_after_PB(z1)
        z2 = self.stage2_FC_after_PB(z2)
        z3 = self.stage3_FC_after_PB(z3)

        p1 = self.stage1_prob(z1)
        o1 = self.stage1_index_offsets(z1)
        p2 = self.stage2_prob(z2)
        o2 = self.stage2_index_offsets(z2)
        p3 = self.stage3_prob(z3)
        o3 = self.stage3_index_offsets(z3)

        r1 = p1[:, 0] * 0
        r2 = p2[:, 0] * 0
        r3 = p3[:, 0] * 0

        for index in range(self.stage_num[0]):
            r1 = r1 + (index + self.lambda_index * o1[:, index]) * p1[:, index]
        r1 = r1.unsqueeze(1)
        r1 = r1 / (self.stage_num[0] * (1 + self.lambda_delta * d1))

        for index in range(self.stage_num[1]):
            r2 = r2 + (index + self.lambda_index * o2[:, index]) * p2[:, index]
        r2 = r2.unsqueeze(1)
        r2 = r2 / (
            self.stage_num[0]
            * (1 + self.lambda_delta * d1)
            * (self.stage_num[1] * (1 + self.lambda_delta * d2))
        )

        for index in range(self.stage_num[2]):
            r3 = r3 + (index + self.lambda_index * o3[:, index]) * p3[:, index]
        r3 = r3.unsqueeze(1)
        r3 = r3 / (
            self.stage_num[0]
            * (1 + self.lambda_delta * d1)
            * (self.stage_num[1] * (1 + self.lambda_delta * d2))
            * (self.stage_num[2] * (1 + self.lambda_delta * d3))
        )

        output = (r1 + r2 + r3) * self.class_range
        return output.squeeze(1)


# -----------------------------------------------------------------------------
# Noise-free model wrapper
# -----------------------------------------------------------------------------
class AgePreprocess(nn.Module):
    def __init__(self, size=(64, 64), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        super().__init__()
        self.size = size
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        x = F.interpolate(x, size=self.size, mode="bilinear", align_corners=False)
        return (x - self.mean) / self.std


class NoiseFreeAgeModel(nn.Module):
    def __init__(self, base_model, preprocess):
        super().__init__()
        self.base = base_model
        self.pre = preprocess

    def forward(self, x):
        # No random perturbation and exactly one deterministic prediction.
        x = self.pre(x)
        return self.base(x).view(-1)


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        state = None
        for key in ("state_dict", "model_state", "base_state", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state = checkpoint[key]
                break
        if state is None:
            state = checkpoint
    else:
        state = checkpoint

    # Support checkpoints saved from DataParallel.
    state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }
    model.load_state_dict(state, strict=True)
    return model


# -----------------------------------------------------------------------------
# Metrics and training
# -----------------------------------------------------------------------------
def compute_metrics(preds, targets):
    errors = preds - targets
    mae = torch.mean(torch.abs(errors)).item()
    mse = torch.mean(errors ** 2).item()
    rmse = float(np.sqrt(mse))
    acc3 = torch.mean((torch.abs(errors) < 3).float()).item() * 100.0
    acc5 = torch.mean((torch.abs(errors) < 5).float()).item() * 100.0
    return {"mae": mae, "mse": mse, "rmse": rmse, "acc3": acc3, "acc5": acc5}


@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    n_items = 0
    all_preds = []
    all_targets = []

    for batch in loader:
        if batch is None:
            continue

        images, targets = batch
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).view(-1)

        preds = model(images)
        loss = loss_fn(preds, targets)
        total_loss += loss.item() * images.size(0)
        n_items += images.size(0)
        all_preds.append(preds.detach().cpu())
        all_targets.append(targets.detach().cpu())

    if n_items == 0:
        raise RuntimeError("No readable images remained in the evaluation dataset.")

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    metrics = compute_metrics(preds, targets)
    metrics["loss"] = total_loss / n_items
    return metrics


def train_noise_free(
    model,
    train_loader,
    val_loader,
    device,
    epochs=20,
    lr=1e-5,
    max_grad_norm=1.0,
    save_root="runs_age_noise_free",
):
    os.makedirs(save_root, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss(beta=3.0)
    best_val_mae = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_mae = 0.0
        n_seen = 0

        progress = tqdm(train_loader, desc=f"[noise-free] epoch {epoch}/{epochs}")
        for batch in progress:
            if batch is None:
                continue

            images, targets = batch
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).view(-1)

            optimizer.zero_grad(set_to_none=True)
            preds = model(images)
            loss = loss_fn(preds, targets)
            loss.backward()

            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

            batch_size = images.size(0)
            batch_mae = torch.mean(torch.abs(preds.detach() - targets)).item()
            running_loss += loss.item() * batch_size
            running_mae += batch_mae * batch_size
            n_seen += batch_size

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                mae=f"{batch_mae:.3f}",
            )

        train_loss = running_loss / max(1, n_seen)
        train_mae = running_mae / max(1, n_seen)
        val_metrics = evaluate(model, val_loader, device, loss_fn)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_loss:.4f} | train MAE {train_mae:.3f} | "
            f"val loss {val_metrics['loss']:.4f} | "
            f"val MAE {val_metrics['mae']:.3f} | "
            f"val RMSE {val_metrics['rmse']:.3f} | "
            f"val Acc@3 {val_metrics['acc3']:.2f}% | "
            f"val Acc@5 {val_metrics['acc5']:.2f}%"
        )

        checkpoint = {
            "model_state": model.base.state_dict(),
            "epoch": epoch,
            "val_metrics": val_metrics,
            "sigma": 0.0,
            "n_samples": 1,
            "aggregator": None,
            "regularization": None,
        }
        torch.save(checkpoint, os.path.join(save_root, "last.pth"))

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            torch.save(checkpoint, os.path.join(save_root, "best.pth"))
            print(f"Saved new best checkpoint: val MAE={best_val_mae:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="age_split")
    parser.add_argument("--checkpoint", default="trained_models/ssrnet_best.pth")
    parser.add_argument("--save-root", default="runs_age_noise_free")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--no-augmentation",
        action="store_true",
        help="Disable the original flip/brightness training augmentations.",
    )
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Noise-free setting: sigma=0, N=1, no aggregator, no B(x)")

    base_dir = Path(args.base_dir)
    train_dir = base_dir / "train" / "images"
    val_dir = base_dir / "val" / "images"
    test_dir = base_dir / "test" / "images"
    train_csv = base_dir / "train" / "labels.csv"
    val_csv = base_dir / "val" / "labels.csv"
    test_csv = base_dir / "test" / "labels.csv"

    required = [train_dir, val_dir, test_dir, train_csv, val_csv, test_csv]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing dataset files/directories:\n" + "\n".join(missing)
        )

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    print(f"Train images: {len(train_df)}")
    print(f"Validation images: {len(val_df)}")
    print(f"Test images: {len(test_df)}")

    train_dataset = CustomAgeDataset(
        train_df,
        str(train_dir),
        image_size=args.image_size,
        augment=not args.no_augmentation,
        split_name="train",
    )
    val_dataset = CustomAgeDataset(
        val_df,
        str(val_dir),
        image_size=args.image_size,
        augment=False,
        split_name="validation",
    )
    test_dataset = CustomAgeDataset(
        test_df,
        str(test_dir),
        image_size=args.image_size,
        augment=False,
        split_name="test",
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=safe_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=safe_collate,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=safe_collate,
    )

    base_model = SSRNet(image_size=args.image_size).to(device)
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"Initial checkpoint was not found: {args.checkpoint}\n"
            "Use --checkpoint to specify the correct path."
        )
    base_model = load_checkpoint(base_model, args.checkpoint, device)

    preprocess = AgePreprocess(
        size=(args.image_size, args.image_size)
    ).to(device)
    model = NoiseFreeAgeModel(base_model, preprocess).to(device)

    train_noise_free(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        max_grad_norm=args.max_grad_norm,
        save_root=args.save_root,
    )

    best_path = os.path.join(args.save_root, "best.pth")
    best_checkpoint = torch.load(best_path, map_location=device)
    model.base.load_state_dict(best_checkpoint["model_state"])

    test_metrics = evaluate(model, test_loader, device, nn.SmoothL1Loss(beta=3.0))
    print("\nFinal noise-free test results:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.6f}")

    torch.save(
        {
            **best_checkpoint,
            "test_metrics": test_metrics,
        },
        os.path.join(args.save_root, "best_with_test_metrics.pth"),
    )


if __name__ == "__main__":
    main()
