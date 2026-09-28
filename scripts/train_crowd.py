#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train the learned aggregation model for ShanghaiTech Part B.

Configuration:
    N_train = 8
    sigma = 8/255
    DeepSets aggregator, hidden=32, psi_depth=rho_depth=3

The script is standalone and includes the .mat-compatible dataset and CSRNet
definitions. Images are kept at native resolution. The input is converted to
[0, 1] first; ImageNet normalization is applied inside the model before
CSRNet, which is required for adding Gaussian input noise.
"""

import argparse
import os
import random
from pathlib import Path

from PIL import Image
import numpy as np
from scipy.io import loadmat
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Standalone ShanghaiTech Part B .mat dataset and CSRNet definitions
# -----------------------------------------------------------------------------
def get_ground_truth_path(image_path):
    image_path = Path(image_path)
    candidates = []
    for ground_truth_dir in ("ground-truth", "ground_truth"):
        candidates.extend([
            image_path.parent.parent / ground_truth_dir / f"GT_{image_path.stem}.mat",
            image_path.parent.parent / ground_truth_dir / f"{image_path.stem}.mat",
            image_path.parent.parent.parent / ground_truth_dir / f"GT_{image_path.stem}.mat",
            image_path.parent.parent.parent / ground_truth_dir / f"{image_path.stem}.mat",
        ])
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Ground-truth file was not found. Checked:\n  "
        + "\n  ".join(str(path) for path in candidates)
    )


def read_people_count(image_path):
    annotation = loadmat(str(get_ground_truth_path(image_path)))
    if "image_info" in annotation:
        points = annotation["image_info"][0, 0][0, 0][0]
        return float(len(np.asarray(points)))
    for key in ("annPoints", "points", "locations"):
        if key in annotation:
            return float(len(np.asarray(annotation[key])))
    raise KeyError(
        f"No point annotations found. Available keys: {list(annotation.keys())}"
    )


class CrowdDataset(Dataset):
    def __init__(self, image_paths, transform=None, train=False, repeat_train=4):
        image_paths = list(image_paths)
        if train:
            image_paths = image_paths * repeat_train
            random.shuffle(image_paths)
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        count = read_people_count(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(count, dtype=torch.float32)


def make_csr_layers(cfg, in_channels=3, dilation=False):
    dilation_rate = 2 if dilation else 1
    layers = []
    for value in cfg:
        if value == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.extend([
                nn.Conv2d(
                    in_channels,
                    value,
                    kernel_size=3,
                    padding=dilation_rate,
                    dilation=dilation_rate,
                ),
                nn.ReLU(inplace=True),
            ])
            in_channels = value
    return nn.Sequential(*layers)


class CSRNet(nn.Module):
    def __init__(self):
        super().__init__()
        frontend = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512]
        backend = [512, 512, 512, 256, 128, 64]
        self.frontend = make_csr_layers(frontend)
        self.backend = make_csr_layers(backend, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x.sum(dim=(1, 2, 3))


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        state = checkpoint.get(
            "state_dict",
            checkpoint.get("model_state", checkpoint.get("base_state", checkpoint)),
        )
    else:
        state = checkpoint
    state = {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state.items()
    }
    model.load_state_dict(state, strict=True)
    print(f"Loaded checkpoint: {checkpoint_path}")
    return model


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class NormalizeOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def forward(self, x):
        return (x - self.mean) / self.std


def make_mlp(in_dim, hidden, out_dim, depth):
    if depth == 1:
        return nn.Sequential(nn.Linear(in_dim, out_dim))

    layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
    for _ in range(depth - 2):
        layers.extend([nn.Linear(hidden, hidden), nn.ReLU()])
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class SoftQuantileAggregator(nn.Module):
    def __init__(self, hidden=32, psi_depth=3, rho_depth=3):
        super().__init__()
        self.hidden = hidden
        self.psi = make_mlp(1, hidden, hidden, psi_depth)
        self.rho = make_mlp(hidden, hidden, 2, rho_depth)

    def forward(self, Y):
        batch_size, n_samples = Y.shape
        Y_flat = Y.reshape(batch_size * n_samples, 1)
        embedding = self.psi(Y_flat).reshape(
            batch_size, n_samples, self.hidden
        )
        pooled = embedding.mean(dim=1)
        output = self.rho(pooled)

        center = output[:, 0]
        sharpness = F.softplus(output[:, 1]) + 1e-6
        logits = -sharpness[:, None] * (Y - center[:, None]).abs()
        weights = torch.softmax(logits, dim=1)
        return (weights * Y).sum(dim=1)


class ChunkedSmoothedMetric(nn.Module):
    """Memory-safe noisy prediction set and learned aggregation."""

    def __init__(
        self,
        base_model,
        preprocess,
        n_samples=8,
        sigma=8 / 255.0,
        chunk_size=1,
    ):
        super().__init__()
        self.base = base_model
        self.pre = preprocess
        self.n = int(n_samples)
        self.sigma = float(sigma)
        self.chunk_size = int(chunk_size)
        self.agg = SoftQuantileAggregator(
            hidden=32,
            psi_depth=3,
            rho_depth=3,
        )

    def forward(self, x):
        batch_size = x.shape[0]
        predictions = []

        for start in range(0, self.n, self.chunk_size):
            chunk_n = min(self.chunk_size, self.n - start)
            noise = torch.randn(
                (chunk_n, *x.shape),
                device=x.device,
                dtype=x.dtype,
            ) * self.sigma

            noisy_x = (x.unsqueeze(0) + noise).clamp(0.0, 1.0)
            flat_x = noisy_x.reshape(chunk_n * batch_size, *x.shape[1:])
            flat_x = self.pre(flat_x)
            flat_y = self.base(flat_x).reshape(chunk_n, batch_size)
            predictions.append(flat_y.transpose(0, 1))

        Y = torch.cat(predictions, dim=1)
        return self.agg(Y), Y


def gradient_penalty(output_sum, inputs):
    gradient = torch.autograd.grad(
        outputs=output_sum,
        inputs=inputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return gradient.pow(2).flatten(1).mean()


def sensitivity_penalty(aggregator, Y, threshold=1.0):
    Y_req = Y.detach().clone().requires_grad_(True)
    output = aggregator(Y_req)
    gradient = torch.autograd.grad(
        outputs=output.sum(),
        inputs=Y_req,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    sensitivity = gradient.abs().sum(dim=1)
    penalty = torch.relu(sensitivity - threshold).pow(2).mean()
    return penalty, sensitivity.mean().detach()


class AdaptiveLossBalancer(nn.Module):
    def __init__(self, n_terms=3, momentum=0.99, eps=1e-8):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("ema", torch.ones(n_terms))

    @torch.no_grad()
    def update(self, values):
        self.ema.mul_(self.momentum).add_(values * (1 - self.momentum))

    def forward(self, losses):
        values = torch.stack([loss.detach() for loss in losses])
        self.update(values)
        weights = 1.0 / (self.ema + self.eps)
        weights = weights / (weights.sum() + self.eps) * len(losses)
        total = sum(weight * loss for weight, loss in zip(weights, losses))
        return total, weights.detach()


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    loss_sum = 0.0
    mae_sum = 0.0
    count = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True).float()
        targets = targets.to(device, non_blocking=True).view(-1).float()
        predictions, _ = model(images)
        loss = criterion(predictions, targets)
        batch_size = images.size(0)
        loss_sum += loss.item() * batch_size
        mae_sum += (predictions - targets).abs().sum().item()
        count += batch_size

    model.train()
    return loss_sum / max(count, 1), mae_sum / max(count, 1)


def train(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Crowd learned aggregation training")
    print("N_train=8, sigma=8/255, DeepSets, native resolution")
    print(f"Batch size: {args.batch_size}; prediction chunk: {args.chunk_size}")

    data_root = Path(args.data_root)
    train_dir = data_root / "train_data" / "images"
    val_dir = data_root / "test_data" / "images"

    train_paths = sorted(str(path) for path in train_dir.glob("*.jpg"))
    val_paths = sorted(str(path) for path in val_dir.glob("*.jpg"))
    if not train_paths or not val_paths:
        raise FileNotFoundError(
            "Expected images were not found:\n"
            f"  train: {train_dir}\n"
            f"  val:   {val_dir}"
        )

    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    # Keep images in [0,1]. Normalization is applied after noise injection.
    image_transform = transforms.ToTensor()

    train_dataset = CrowdDataset(
        train_paths,
        transform=image_transform,
        train=True,
        repeat_train=4,
    )
    val_dataset = CrowdDataset(
        val_paths,
        transform=image_transform,
        train=False,
    )

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    base_model = CSRNet().to(device)
    load_checkpoint(base_model, args.checkpoint, device)

    model = ChunkedSmoothedMetric(
        base_model=base_model,
        preprocess=NormalizeOnly().to(device),
        n_samples=8,
        sigma=8 / 255.0,
        chunk_size=args.chunk_size,
    ).to(device)

    optimizer = torch.optim.Adam(
        list(model.base.parameters()) + list(model.agg.parameters()),
        lr=args.lr,
    )
    criterion = nn.SmoothL1Loss(beta=3.0)
    balancer = AdaptiveLossBalancer(n_terms=3).to(device)
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_mae = float("inf")

    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = {"loss": 0.0, "main": 0.0, "lip": 0.0, "sens": 0.0}
        total_items = 0

        progress = tqdm(
            train_loader,
            desc=f"[crowd_8_255] epoch {epoch}/{args.epochs}",
        )

        for images, targets in progress:
            images = images.to(device, non_blocking=True).float()
            targets = targets.to(device, non_blocking=True).view(-1).float()

            optimizer.zero_grad(set_to_none=True)
            images_gp = images.detach().clone().requires_grad_(True)

            predictions, Y = model(images_gp)
            main_loss = criterion(predictions, targets)

            clean_output = model.base(model.pre(images_gp)).view(-1)
            lip_loss = gradient_penalty(clean_output.sum(), images_gp)
            sens_loss, sens_value = sensitivity_penalty(
                model.agg,
                Y,
                threshold=args.sens_threshold,
            )

            loss, weights = balancer([main_loss, lip_loss, sens_loss])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.base.parameters()) + list(model.agg.parameters()),
                max_norm=1.0,
            )
            optimizer.step()

            batch_size = images.size(0)
            running["loss"] += loss.item() * batch_size
            running["main"] += main_loss.item() * batch_size
            running["lip"] += lip_loss.item() * batch_size
            running["sens"] += sens_loss.item() * batch_size
            total_items += batch_size

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                mse=f"{main_loss.item():.4f}",
                S=f"{sens_value.item():.3f}",
            )

        train_loss = running["loss"] / max(total_items, 1)
        train_main = running["main"] / max(total_items, 1)
        train_lip = running["lip"] / max(total_items, 1)
        train_sens = running["sens"] / max(total_items, 1)
        val_loss, val_mae = evaluate(model, val_loader, device, criterion)

        print(
            f"Epoch {epoch:03d} | loss {train_loss:.4f} | "
            f"main {train_main:.4f} | lip {train_lip:.4f} | "
            f"sens {train_sens:.4f} | val loss {val_loss:.4f} | "
            f"val MAE {val_mae:.4f}"
        )

        checkpoint = {
            "base_state": model.base.state_dict(),
            "agg_state": model.agg.state_dict(),
            "sigma": 8 / 255.0,
            "n": 8,
            "epoch": epoch,
            "val_mae": val_mae,
        }
        torch.save(checkpoint, os.path.join(args.save_dir, "last.pth"))

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(checkpoint, os.path.join(args.save_dir, "best.pth"))
            print(f"Saved best checkpoint: val MAE={best_val_mae:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/ShanghaiTech/part_B")
    parser.add_argument("--checkpoint", default="data/partBmodel_best.pth")
    parser.add_argument("--save-dir", default="runs_crowd_agg_8_255")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--sens-threshold", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
