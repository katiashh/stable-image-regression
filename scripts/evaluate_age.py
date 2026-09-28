import os
import csv
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
batch_size=1
input_size = 64
BASE_DIR = "age_split"   # <-- поправь под свой путь

TRAIN_DIR = os.path.join(BASE_DIR, "train", "images")
VAL_DIR   = os.path.join(BASE_DIR, "val", "images")
TEST_DIR  = os.path.join(BASE_DIR, "test", "images")

TRAIN_CSV = os.path.join(BASE_DIR, "train", "labels.csv")
VAL_CSV   = os.path.join(BASE_DIR, "val", "labels.csv")
TEST_CSV  = os.path.join(BASE_DIR, "test", "labels.csv")

# =========================
# Utils
# =========================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


device = 'cuda'

class CustomAgeDataset(Dataset):
    def __init__(self, df, image_dir, image_size=64, augment=False):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def _augment(self, img):
        # img: HWC, RGB, uint8
        if np.random.rand() < 0.5:
            img = np.fliplr(img).copy()

        if np.random.rand() < 0.3:
            alpha = np.random.uniform(0.9, 1.1)
            beta = np.random.uniform(-10, 10)
            img = np.clip(img * alpha + beta, 0, 255).astype(np.uint8)

        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["filename"]
        age = float(row["age"])

        img_path = os.path.join(self.image_dir, filename)
        img = cv2.imread(img_path)

        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))

        if self.augment:
            img = self._augment(img)

        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW

        img = torch.tensor(img, dtype=torch.float32)
        age = torch.tensor(age, dtype=torch.float32)
        age = age / 100
        if age < 0:
            age = 0
        if age > 1:
            age = 1
        return img, age


# =========================
# Dataset
# =========================
class AgeSplitDataset(Dataset):
    def __init__(self, csv_path, image_dir, image_size=64):
        self.df = pd.read_csv(csv_path).dropna(subset=["filename", "age"]).reset_index(drop=True)
        self.image_dir = image_dir
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = str(row["filename"])
        age = float(row["age"])

        img_path = os.path.join(self.image_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW

        x = torch.tensor(img, dtype=torch.float32)
        y = torch.tensor(age, dtype=torch.float32)
        return x, y, filename


# =========================
# Preprocess
# =========================
class AgePreprocess(nn.Module):
    def __init__(self, size=(64, 64), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        super().__init__()
        self.size = size
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        x = F.interpolate(x, size=self.size, mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        return x


# =========================
# SSRNet
# =========================
class SSRNet(nn.Module):
    def __init__(self, stage_num=[3, 3, 3], image_size=64,
                 class_range=101, lambda_index=1., lambda_delta=1.):
        super(SSRNet, self).__init__()
        self.image_size = image_size
        self.stage_num = stage_num
        self.lambda_index = lambda_index
        self.lambda_delta = lambda_delta
        self.class_range = class_range

        self.stream1_stage3 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2, 2)
        )
        self.stream1_stage2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2, 2)
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
            nn.MaxPool2d(2, 2)
        )
        self.stream2_stage2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.MaxPool2d(2, 2)
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
            nn.AvgPool2d(8, 8)
        )
        self.funsion_block_stream1_stage_3_prediction_block = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(10 * 4 * 4, self.stage_num[2]),
            nn.ReLU()
        )

        self.funsion_block_stream1_stage_2_before_PB = nn.Sequential(
            nn.Conv2d(32, 10, 1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(4, 4)
        )
        self.funsion_block_stream1_stage_2_prediction_block = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(10 * 4 * 4, self.stage_num[1]),
            nn.ReLU()
        )

        self.funsion_block_stream1_stage_1_before_PB = nn.Sequential(
            nn.Conv2d(32, 10, 1, padding=0),
            nn.ReLU(),
        )
        self.funsion_block_stream1_stage_1_prediction_block = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(10 * 8 * 8, self.stage_num[0]),
            nn.ReLU()
        )

        self.funsion_block_stream2_stage_3_before_PB = nn.Sequential(
            nn.Conv2d(16, 10, 1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(8, 8)
        )
        self.funsion_block_stream2_stage_3_prediction_block = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(10 * 4 * 4, self.stage_num[2]),
            nn.ReLU()
        )

        self.funsion_block_stream2_stage_2_before_PB = nn.Sequential(
            nn.Conv2d(16, 10, 1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(4, 4)
        )
        self.funsion_block_stream2_stage_2_prediction_block = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(10 * 4 * 4, self.stage_num[1]),
            nn.ReLU()
        )

        self.funsion_block_stream2_stage_1_before_PB = nn.Sequential(
            nn.Conv2d(16, 10, 1, padding=0),
            nn.ReLU(),
        )
        self.funsion_block_stream2_stage_1_prediction_block = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(10 * 8 * 8, self.stage_num[0]),
            nn.ReLU()
        )

        self.stage3_FC_after_PB = nn.Sequential(
            nn.Linear(self.stage_num[0], 2 * self.stage_num[0]),
            nn.ReLU()
        )
        self.stage3_prob = nn.Sequential(
            nn.Linear(2 * self.stage_num[0], self.stage_num[0]),
            nn.Softmax(dim=1)
        )
        self.stage3_index_offsets = nn.Sequential(
            nn.Linear(2 * self.stage_num[0], self.stage_num[0]),
            nn.Tanh()
        )
        self.stage3_delta_k = nn.Sequential(
            nn.Linear(10 * 4 * 4, 1),
            nn.Tanh()
        )

        self.stage2_FC_after_PB = nn.Sequential(
            nn.Linear(self.stage_num[0], 2 * self.stage_num[0]),
            nn.ReLU()
        )
        self.stage2_prob = nn.Sequential(
            nn.Linear(2 * self.stage_num[0], self.stage_num[0]),
            nn.Softmax(dim=1)
        )
        self.stage2_index_offsets = nn.Sequential(
            nn.Linear(2 * self.stage_num[0], self.stage_num[0]),
            nn.Tanh()
        )
        self.stage2_delta_k = nn.Sequential(
            nn.Linear(10 * 4 * 4, 1),
            nn.Tanh()
        )

        self.stage1_FC_after_PB = nn.Sequential(
            nn.Linear(self.stage_num[0], 2 * self.stage_num[0]),
            nn.ReLU()
        )
        self.stage1_prob = nn.Sequential(
            nn.Linear(2 * self.stage_num[0], self.stage_num[0]),
            nn.Softmax(dim=1)
        )
        self.stage1_index_offsets = nn.Sequential(
            nn.Linear(2 * self.stage_num[0], self.stage_num[0]),
            nn.Tanh()
        )
        self.stage1_delta_k = nn.Sequential(
            nn.Linear(10 * 8 * 8, 1),
            nn.Tanh()
        )

    def forward(self, image_):
        feature_stream1_stage3 = self.stream1_stage3(image_)
        feature_stream1_stage2 = self.stream1_stage2(feature_stream1_stage3)
        feature_stream1_stage1 = self.stream1_stage1(feature_stream1_stage2)

        feature_stream2_stage3 = self.stream2_stage3(image_)
        feature_stream2_stage2 = self.stream2_stage2(feature_stream2_stage3)
        feature_stream2_stage1 = self.stream2_stage1(feature_stream2_stage2)

        feature_stream1_stage3_before_PB = self.funsion_block_stream1_stage_3_before_PB(feature_stream1_stage3)
        feature_stream1_stage2_before_PB = self.funsion_block_stream1_stage_2_before_PB(feature_stream1_stage2)
        feature_stream1_stage1_before_PB = self.funsion_block_stream1_stage_1_before_PB(feature_stream1_stage1)

        feature_stream2_stage3_before_PB = self.funsion_block_stream2_stage_3_before_PB(feature_stream2_stage3)
        feature_stream2_stage2_before_PB = self.funsion_block_stream2_stage_2_before_PB(feature_stream2_stage2)
        feature_stream2_stage1_before_PB = self.funsion_block_stream2_stage_1_before_PB(feature_stream2_stage1)

        embedding_stream1_stage3_before_PB = feature_stream1_stage3_before_PB.view(feature_stream1_stage3_before_PB.size(0), -1)
        embedding_stream1_stage2_before_PB = feature_stream1_stage2_before_PB.view(feature_stream1_stage2_before_PB.size(0), -1)
        embedding_stream1_stage1_before_PB = feature_stream1_stage1_before_PB.view(feature_stream1_stage1_before_PB.size(0), -1)

        embedding_stream2_stage3_before_PB = feature_stream2_stage3_before_PB.view(feature_stream2_stage3_before_PB.size(0), -1)
        embedding_stream2_stage2_before_PB = feature_stream2_stage2_before_PB.view(feature_stream2_stage2_before_PB.size(0), -1)
        embedding_stream2_stage1_before_PB = feature_stream2_stage1_before_PB.view(feature_stream2_stage1_before_PB.size(0), -1)

        stage1_delta_k = self.stage1_delta_k(torch.mul(embedding_stream1_stage1_before_PB, embedding_stream2_stage1_before_PB))
        stage2_delta_k = self.stage2_delta_k(torch.mul(embedding_stream1_stage2_before_PB, embedding_stream2_stage2_before_PB))
        stage3_delta_k = self.stage3_delta_k(torch.mul(embedding_stream1_stage3_before_PB, embedding_stream2_stage3_before_PB))

        embedding_stage1_after_PB = torch.mul(
            self.funsion_block_stream1_stage_1_prediction_block(embedding_stream1_stage1_before_PB),
            self.funsion_block_stream2_stage_1_prediction_block(embedding_stream2_stage1_before_PB)
        )
        embedding_stage2_after_PB = torch.mul(
            self.funsion_block_stream1_stage_2_prediction_block(embedding_stream1_stage2_before_PB),
            self.funsion_block_stream2_stage_2_prediction_block(embedding_stream2_stage2_before_PB)
        )
        embedding_stage3_after_PB = torch.mul(
            self.funsion_block_stream1_stage_3_prediction_block(embedding_stream1_stage3_before_PB),
            self.funsion_block_stream2_stage_3_prediction_block(embedding_stream2_stage3_before_PB)
        )

        embedding_stage1_after_PB = self.stage1_FC_after_PB(embedding_stage1_after_PB)
        embedding_stage2_after_PB = self.stage2_FC_after_PB(embedding_stage2_after_PB)
        embedding_stage3_after_PB = self.stage3_FC_after_PB(embedding_stage3_after_PB)

        prob_stage_1 = self.stage1_prob(embedding_stage1_after_PB)
        index_offset_stage1 = self.stage1_index_offsets(embedding_stage1_after_PB)

        prob_stage_2 = self.stage2_prob(embedding_stage2_after_PB)
        index_offset_stage2 = self.stage2_index_offsets(embedding_stage2_after_PB)

        prob_stage_3 = self.stage3_prob(embedding_stage3_after_PB)
        index_offset_stage3 = self.stage3_index_offsets(embedding_stage3_after_PB)

        stage1_regress = prob_stage_1[:, 0] * 0
        stage2_regress = prob_stage_2[:, 0] * 0
        stage3_regress = prob_stage_3[:, 0] * 0

        for index in range(self.stage_num[0]):
            stage1_regress = stage1_regress + (index + self.lambda_index * index_offset_stage1[:, index]) * prob_stage_1[:, index]
        stage1_regress = torch.unsqueeze(stage1_regress, 1)
        stage1_regress = stage1_regress / (self.stage_num[0] * (1 + self.lambda_delta * stage1_delta_k))

        for index in range(self.stage_num[1]):
            stage2_regress = stage2_regress + (index + self.lambda_index * index_offset_stage2[:, index]) * prob_stage_2[:, index]
        stage2_regress = torch.unsqueeze(stage2_regress, 1)
        stage2_regress = stage2_regress / (
            self.stage_num[0] * (1 + self.lambda_delta * stage1_delta_k) *
            (self.stage_num[1] * (1 + self.lambda_delta * stage2_delta_k))
        )

        for index in range(self.stage_num[2]):
            stage3_regress = stage3_regress + (index + self.lambda_index * index_offset_stage3[:, index]) * prob_stage_3[:, index]
        stage3_regress = torch.unsqueeze(stage3_regress, 1)
        stage3_regress = stage3_regress / (
            self.stage_num[0] * (1 + self.lambda_delta * stage1_delta_k) *
            (self.stage_num[1] * (1 + self.lambda_delta * stage2_delta_k)) *
            (self.stage_num[2] * (1 + self.lambda_delta * stage3_delta_k))
        )

        regress_class = (stage1_regress + stage2_regress + stage3_regress) * self.class_range
        regress_class = torch.squeeze(regress_class, 1) / 200
        regress_class = torch.clip(regress_class, min=0, max=1)
        return regress_class


# =========================
# Aggregators
# =========================
def make_mlp(in_dim, hidden, out_dim, depth):
    layers = []
    if depth == 1:
        layers.append(nn.Linear(in_dim, out_dim))
        return nn.Sequential(*layers)

    layers.append(nn.Linear(in_dim, hidden))
    layers.append(nn.ReLU())

    for _ in range(depth - 2):
        layers.append(nn.Linear(hidden, hidden))
        layers.append(nn.ReLU())

    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class MeanAggregator(nn.Module):
    def forward(self, Y):
        return Y.mean(dim=1)


class MedianAggregator(nn.Module):
    def forward(self, Y):
        return Y.median(dim=1).values


class TrimmedMeanAggregator(nn.Module):
    def __init__(self, trim_ratio=0.25):
        super().__init__()
        self.trim_ratio = float(trim_ratio)

    def forward(self, Y):
        B, n = Y.shape
        k = int(n * self.trim_ratio)
        Y_sorted, _ = torch.sort(Y, dim=1)
        if 2 * k >= n:
            return Y.mean(dim=1)
        Y_trim = Y_sorted[:, k:n-k]
        return Y_trim.mean(dim=1)


class SoftQuantileAggregator(nn.Module):
    def __init__(self, mode="lightweight", hidden=32, psi_depth=1, rho_depth=1):
        super().__init__()
        assert mode in ["lightweight", "deepsets"]
        self.mode = mode
        self.hidden = hidden

        if mode == "lightweight":
            self.log_alpha = nn.Parameter(torch.tensor(0.0))
        else:
            self.psi = make_mlp(in_dim=1, hidden=hidden, out_dim=hidden, depth=psi_depth)
            self.rho = make_mlp(in_dim=hidden, hidden=hidden, out_dim=2, depth=rho_depth)

    def forward(self, Y):
        B, n = Y.shape

        if self.mode == "lightweight":
            t = Y.mean(dim=1)
            alpha = F.softplus(self.log_alpha) + 1e-6
            alpha = alpha.expand(B)
        else:
            Y_flat = Y.reshape(B * n, 1)
            emb = self.psi(Y_flat).reshape(B, n, self.hidden)
            s = emb.mean(dim=1)
            out = self.rho(s)
            t = out[:, 0]
            alpha = F.softplus(out[:, 1]) + 1e-6

        dist = (Y - t.unsqueeze(1)).abs()
        logits = -alpha.unsqueeze(1) * dist
        w = torch.softmax(logits, dim=1)
        g = (w * Y).sum(dim=1)
        return g


# =========================
# Smoothed model
# =========================
class SmoothedMetric(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        preprocess: nn.Module,
        n_samples: int = 16,
        sigma: float = 4.0 / 255.0,
        aggregator_mode: str = "deepsets",
        hidden: int = 32,
        psi_depth: int = 1,
        rho_depth: int = 1,
        trim_ratio: float = 0.1,
    ):
        super().__init__()
        self.base = base_model
        self.pre = preprocess
        self.n = int(n_samples)
        self.sigma = float(sigma)

        if aggregator_mode in ["lightweight", "deepsets"]:
            self.agg = SoftQuantileAggregator(
                mode=aggregator_mode,
                hidden=hidden,
                psi_depth=psi_depth,
                rho_depth=rho_depth,
            )
        elif aggregator_mode == "mean":
            self.agg = MeanAggregator()
        elif aggregator_mode == "median":
            self.agg = MedianAggregator()
        elif aggregator_mode == "trim_mean":
            self.agg = TrimmedMeanAggregator(trim_ratio=trim_ratio)
        else:
            raise ValueError(f"Unknown aggregator_mode: {aggregator_mode}")

    def forward(self, x):
        B = x.shape[0]
        noises = torch.randn(self.n, *x.shape, device=x.device, dtype=x.dtype) * self.sigma
        x_rep = (x.unsqueeze(0) + noises).clamp(0.0, 1.0)
        x_flat = x_rep.reshape(self.n * B, *x.shape[1:])
        x_flat = self.pre(x_flat)

        y_flat = self.base(x_flat)
        if y_flat.ndim > 1:
            y_flat = y_flat.view(-1)

        Y = y_flat.view(self.n, B).transpose(0, 1)
        g = self.agg(Y)
        return g, Y


# =========================
# Eval
# =========================
@torch.no_grad()
def evaluate_age_model(model, loader, device, save_predictions_path=None):
    model.eval()

    mae_sum = 0.0
    mse_sum = 0.0
    ca3_sum = 0
    ca5_sum = 0
    n = 0

    rows = []

    for x, y, filenames in tqdm(loader, desc="Testing"):
        x = x.to(device).float()
        y = y.to(device).view(-1).float()

        g, Y = model(x)

        err = torch.abs(g - y)
        mae_sum += err.sum().item()
        mse_sum += ((g - y) ** 2).sum().item()
        ca3_sum += (err < 3).sum().item()
        ca5_sum += (err < 5).sum().item()
        n += x.size(0)

        if save_predictions_path is not None:
            for i in range(x.size(0)):
                rows.append({
                    "filename": filenames[i],
                    "true_age": float(y[i].item()),
                    "pred_age": float(g[i].item()),
                    "abs_error": float(err[i].item()),
                    "mc_mean": float(Y[i].mean().item()),
                    "mc_std": float(Y[i].std(unbiased=False).item()),
                })

    metrics = {
        "MAE": mae_sum / max(1, n),
        "MSE": mse_sum / max(1, n),
        "RMSE": (mse_sum / max(1, n)) ** 0.5,
        "CA_3": ca3_sum / max(1, n),
        "CA_5": ca5_sum / max(1, n),
        "N": n,
    }

    if save_predictions_path is not None:
        pd.DataFrame(rows).to_csv(save_predictions_path, index=False)

    return metrics



import torch
import torch.nn.functional as F
from tqdm import tqdm

@torch.no_grad()
def compute_Y_streaming_defended(model, x, n_samples: int, sigma: float, chunk: int = 16,
                                 generator: torch.Generator | None = None):
    """
    model must have .base and .pre
    Returns Y_cpu [B,n] (float32, CPU). No [n,B,3,H,W] allocations.
    """
    B = x.shape[0]
    device = x.device
    dtype = x.dtype

    Y_cpu = torch.empty(B, n_samples, device="cpu", dtype=torch.float32)

    filled = 0
    while filled < n_samples:
        m = min(chunk, n_samples - filled)
        #print(sigma)
        noises = torch.randn((m, *x.shape), device=device, dtype=dtype, generator=generator) * sigma
        x_rep = (x.unsqueeze(0) + noises).clamp(0.0, 1.0)                       # [m,B,3,H,W]
        x_flat = model.pre(x_rep.reshape(m * B, *x.shape[1:]))                  # [m*B,3,H,W]
        y_flat = model.base(x_flat).view(m, B).transpose(0, 1)                  # [B,m]

        Y_cpu[:, filled:filled+m] = y_flat.detach().float().cpu()
        filled += m

        del noises, x_rep, x_flat, y_flat

    return Y_cpu


def local_C_from_Y(agg, Y):
    """
    agg: model.agg
    Y: [B, n] tensor (detached ok)
    returns: C_batch [B] where C(x)=sum_i |dPhi/dY_i|
    """
    Y_req = Y.detach().clone().requires_grad_(True)   # [B,n]
    gY = agg(Y_req)                                   # [B]
    dgdY = torch.autograd.grad(gY.sum(), Y_req, create_graph=False)[0]  # [B,n]
    C_batch = dgdY.abs().sum(dim=1)                   # [B]
    return C_batch.detach()


def forward_with_noises(defended, x, noises):
    # noises: [n,B,3,H,W]
    B = x.shape[0]
    x_rep = (x.unsqueeze(0) + noises).clamp(0.0, 1.0)
    x_flat = defended.pre(x_rep.reshape(defended.n * B, *x.shape[1:]))
    y_flat = defended.base(x_flat).view(defended.n, B).transpose(0, 1)  # [B,n]
    g = defended.agg(y_flat)
    return g, y_flat 


def local_L2_base_worst_streaming(model, x, n_samples: int, sigma: float, eps2: float,
                                  steps: int = 3, step_size: float = 0.25,
                                  generator: torch.Generator | None = None):
    """
    Memory-safe proxy for:
      max_i sup_{||u||_2<=eps2} ||∇ f(x+eta_i+u)||_2
    Processes one noise per iteration (no [n,...] tensors).
    model must have .base and .pre
    """
    B = x.shape[0]
    device = x.device
    dtype = x.dtype

    L_max = torch.zeros(B, device=device, dtype=torch.float32)

    for _ in range(n_samples):
        eta = torch.randn(x.shape, device=device, dtype=dtype, generator=generator) * sigma
        base = (x + eta).clamp(0.0, 1.0)

        u = torch.zeros_like(x)
        for _ in range(steps):
            z = (base + u).clamp(0.0, 1.0)
            z_req = z.detach().requires_grad_(True)

            f = model.base(model.pre(z_req)).view(-1)  # [B]
            grad = torch.autograd.grad(f.sum(), z_req, create_graph=False)[0]

            g_dir = grad.detach()
            g_dir = g_dir / (g_dir.flatten(1).norm(p=2, dim=1).view(-1, 1, 1, 1) + 1e-12)
            z_adv = (z_req + step_size * eps2 * g_dir).clamp(0.0, 1.0)

            u = (z_adv.detach() - base).detach()
            u_flat = u.view(B, -1)
            norm = u_flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
            factor = (eps2 / norm).clamp_max(1.0)
            u = (u_flat * factor).view_as(u)

            del z, z_req, f, grad, g_dir, z_adv

        z = (base + u).clamp(0.0, 1.0)
        z_req = z.detach().requires_grad_(True)
        f = model.base(model.pre(z_req)).view(-1)
        grad = torch.autograd.grad(f.sum(), z_req, create_graph=False)[0]
        gn = grad.flatten(1).norm(p=2, dim=1).float()  # [B]

        L_max = torch.maximum(L_max, gn)

        del eta, base, u, z, z_req, f, grad, gn

    return L_max.detach()



base_model = SSRNet(image_size=input_size).to(device)
base_ckpt = torch.load('trained_models/ssrnet_best.pth', map_location=device)
if "state_dict" in base_ckpt:
    base_model.load_state_dict(base_ckpt["state_dict"])
else:
    base_model.load_state_dict(base_ckpt)
base_model.eval()
pre = AgePreprocess(size=(input_size, input_size)).to(device)

def pgd_l2_attack_base_model(
    base_model,
    x,
    y,
    eps: float,
    steps: int = 10,
    step_size: float | None = None,
    clamp_min: float = -1.0,
    clamp_max: float = 1.0,
):
    base_model = SSRNet(image_size=input_size).to(device)
    base_ckpt = torch.load('trained_models/ssrnet_best.pth', map_location=device)
    if "state_dict" in base_ckpt:
        base_model.load_state_dict(base_ckpt["state_dict"])
    else:
        base_model.load_state_dict(base_ckpt)
    base_model.eval()

    if step_size is None:
        step_size = eps / steps

    x0 = x.detach()
    x0 = pre(x0)
    delta = torch.zeros_like(x0)

    for _ in range(steps):
        delta.requires_grad_(True)

        x_adv = torch.clamp(x0 + delta, clamp_min, clamp_max)
        pred = base_model(x_adv).view(-1)

        loss = F.mse_loss(pred, y, reduction="sum")
        grad = torch.autograd.grad(loss, delta)[0]

        grad_norm = grad.view(grad.size(0), -1).norm(p=2, dim=1).view(-1, 1, 1, 1)
        grad = grad / (grad_norm + 1e-12)

        delta = delta.detach() + step_size * grad

        delta_flat = delta.view(delta.size(0), -1)
        delta_norm = delta_flat.norm(p=2, dim=1).view(-1, 1, 1, 1)
        factor = torch.clamp(eps / (delta_norm + 1e-12), max=1.0)
        delta = delta * factor

        delta = torch.clamp(x0 + delta, clamp_min, clamp_max) - x0

    return torch.clamp(x0 + delta, clamp_min, clamp_max)*0.5+0.5

def compute_local_bounds_L2_ours_streaming(
    defended_model,
    loader,
    device,
    eps2: float,
    n_samples: int = 500,
    sigma: float = 4/255.0,
    max_batches=None,
    # memory knobs
    chunk_Y: int = 16,
    # L-worst knobs
    steps_L: int = 3,
    step_size_L: float = 0.25,
    # optionally compute L with fewer noises than Y to save time
    n_L: int | None = None,
    # reproducibility
    seed: int | None = None,
):
    """
    Memory-safe evaluation for OUR defended model:
      - Y computed in chunks (no [n,3,H,W] stored)
      - C(x) computed from aggregator on Y (tiny tensor [B,n])
      - L_max computed streaming one noise at a time (no [n,...] tensors)
      - coupled noises between Y-pass and L-pass via per-batch generator seed
    """
    defended_model.eval()

    all_L, all_C, all_B = [], [], []
    mse_sum = 0.0
    total = 0

    total_iters = len(loader) if max_batches is None else min(len(loader), max_batches)

    for bi, (x, y) in enumerate(tqdm(loader, total=total_iters, desc="Eval (ours, streaming)")):
        if max_batches is not None and bi >= max_batches:
            break

        x = x.to(device).float()
        y = y.to(device).float().view(-1)
        x = pgd_l2_attack_base_model(
            base_model=base_model,
            x=x,
            y=y,
            eps=0.5,
            steps=10,
            step_size=0.05,
        )

        # --- per-batch coupled RNG (so Y and L see the same eta_i sequence) ---
        if seed is None:
            batch_seed = None
        else:
            batch_seed = int(seed) + int(bi)

        genY = torch.Generator(device=device).manual_seed(batch_seed) if batch_seed is not None else None
        genL = torch.Generator(device=device).manual_seed(batch_seed) if batch_seed is not None else None

        # ---- Y streaming on CPU ----
        Y_cpu = compute_Y_streaming_defended(
            defended_model, x, n_samples=n_samples, sigma=sigma, chunk=chunk_Y, generator=genY
        )  # CPU [B,n]

        # ---- g and MSE ----
        with torch.no_grad():
            Y_dev = Y_cpu.to(device=device, dtype=torch.float32)   # [B,n]
            g = defended_model.agg(Y_dev)                          # [B]
         #   print(Y_dev)
         #   print(g)

        mse_sum += F.mse_loss(g.view(-1), y, reduction="sum").item()
        total += x.size(0)

        # ---- C(x) from aggregator ----
        Cx = local_C_from_Y(defended_model.agg, Y_dev).detach().cpu()  # [B]

        # ---- L_max streaming ----
        n_L_eff = n_samples if n_L is None else int(n_L)
        Lx = local_L2_base_worst_streaming(
            defended_model, x, n_samples=n_L_eff, sigma=sigma,
            eps2=eps2, steps=steps_L, step_size=step_size_L,
            generator=genL,
        ).detach().cpu()  # [B]

        Bx = eps2 * Lx * Cx

        all_L.append(Lx)
        all_C.append(Cx)
        all_B.append(Bx)

        del Y_dev, g, Cx
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    L = torch.cat(all_L) if all_L else torch.empty(0)
    C = torch.cat(all_C) if all_C else torch.empty(0)
    B = torch.cat(all_B) if all_B else torch.empty(0)

    mse = mse_sum / total if total > 0 else float("nan")

    def q(t, p):
        return float(t.quantile(p)) if t.numel() else float("nan")

    return {
        "L": L, "C": C, "B": B,
        "stats": {
            "MSE": mse,
            "L_mean": float(L.mean()) if L.numel() else float("nan"),
            "L_p90": q(L, 0.90), "L_p99": q(L, 0.99),
            "C_mean": float(C.mean()) if C.numel() else float("nan"),
            "C_p90": q(C, 0.90), "C_p99": q(C, 0.99),
            "B_mean": float(B.mean()) if B.numel() else float("nan"),
            "B_p90": q(B, 0.90), "B_p99": q(B, 0.99),
        }
    }

class NormalizeOnly(nn.Module):
    def __init__(self, mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1,3,1,1))
        self.register_buffer("std", torch.tensor(std).view(1,3,1,1))

    def forward(self, x):
        return (x - self.mean) / self.std

class MyCustomDataset(Dataset):
    def __init__(self, path_gt, mode):
        self._items = []
        self.mode = mode

        all_imgs = sorted(os.listdir(path_gt))
        if mode == 'train':
            dir_img = all_imgs[:8000]
        elif mode == 'val':
            dir_img = all_imgs[8000:9000]
        else:
            dir_img = all_imgs[9000:]

        df = pd.read_csv('data/koniq10k_scores_and_distributions.csv')
        av_names = set(df['image_name'].tolist())

        # keep only images present in csv
        for img_name in dir_img:
            if img_name in av_names:
                self._items.append(os.path.join(path_gt, img_name))

        random.shuffle(self._items)

        # map name -> MOS_zscore
        self.dic = {}
        for _, row in df.iterrows():
            name = row['image_name']
            if name in dir_img:
                self.dic[name] = float(row['MOS_zscore'])

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        img_path = self._items[index]
        image = Image.open(img_path).convert('RGB')
        image = np.array(image).astype(np.float32)

        # Your resize: (width=384, height=512)
        image = cv2.resize(image, dsize=(384, 512), interpolation=cv2.INTER_AREA)

        image = image / 255.0
        x = self.to_tensor(image)  # [3,512,384], float32 in [0,1]

        y = self.dic[os.path.basename(img_path)]
        y = torch.tensor(y, dtype=torch.float32)  # scalar tensor
        return x, y / 100


# -------------------------
#  Preprocess (tensor-safe)
# -------------------------
class TensorPreprocess(nn.Module):
    """
    Resize + Normalize for tensor images.
    Expects image in range [0,1], shape [B,3,H,W].
    """
    def __init__(self, size=(512, 384), mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)):
        super().__init__()
        self.size = size
        self.register_buffer("mean", torch.tensor(mean).view(1,3,1,1))
        self.register_buffer("std", torch.tensor(std).view(1,3,1,1))

    def forward(self, x):
        x = F.interpolate(x, size=self.size, mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        return x

eps = 8/255.
sigma = 8/255.
n=128

# ===== LOAD CSV =====
train_df = pd.read_csv(TRAIN_CSV)
val_df   = pd.read_csv(VAL_CSV)
test_df  = pd.read_csv(TEST_CSV)

print("Train:", len(train_df))
print("Val:", len(val_df))
print("Test:", len(test_df))


# ===== DATASETS =====
train_gen = CustomAgeDataset(
    train_df,
    TRAIN_DIR,
    image_size=input_size,
    augment=True
)

val_gen = CustomAgeDataset(
    val_df,
    VAL_DIR,
    image_size=input_size,
    augment=False
)

test_gen = CustomAgeDataset(
    test_df,
    TEST_DIR,
    image_size=input_size,
    augment=False
)


# ===== DATALOADERS =====
train_loader = DataLoader(
    train_gen,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_gen,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_gen,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)


dataloaders = {
    "train": train_loader,
    "val": val_loader,
    "test": test_loader,
}



# =========================
# Main
# =========================
if __name__ == "__main__":
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BASE_DIR = "age_split"
    TEST_DIR = os.path.join(BASE_DIR, "test", "images")
    TEST_CSV = os.path.join(BASE_DIR, "test", "labels.csv")

    SAVE_DIR = Path("test_results2")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # базовая age-модель
    base_ckpt_path = "trained_models/ssrnet_best.pth"

    # чекпойнт агрегатора
    agg_ckpt_path = "runs_age_agg_8/best.pth"

    # параметры smoothed model
    input_size = 64
    batch_size = 1
    num_workers = 1

    sigma = 8 / 255.0
    n_samples = 128

    methods = [
        {
            "name": "deepsets",
            "aggregator_mode": "deepsets",
            "ckpt": agg_ckpt_path,
            "hidden": 32,
            "psi_depth": 3,
            "rho_depth": 3,
        },
        {
            "name": "mean",
            "aggregator_mode": "mean",
            "ckpt": None,
        },
        {
            "name": "median",
            "aggregator_mode": "median",
            "ckpt": None,
        },
        {
            "name": "trim_mean",
            "aggregator_mode": "trim_mean",
            "ckpt": None,
            "trim_ratio": 0.1,
        },
    ]

    ds_test = AgeSplitDataset(TEST_CSV, TEST_DIR, image_size=input_size)
    test_loader = DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    results = []
    results_csv = SAVE_DIR / "age_aggregator_results_8_128_adv.csv"

    if not results_csv.exists():
        with open(results_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "method",
                "eps",
                "sigma",
                "n",
                "MSE",
                "L_mean",
                "L_p90", 
                "L_p99",
                "C_mean",
                "C_p90", 
                "C_p99",
                "B_mean",
                "B_p90", 
                "B_p99",
            ])

    for cfg in methods:
        print(f"\n=== Testing: {cfg['name']} ===")

        # base model
        base_model = SSRNet(image_size=input_size).to(device)
        base_ckpt = torch.load(base_ckpt_path, map_location=device)
        if "state_dict" in base_ckpt:
            base_model.load_state_dict(base_ckpt["state_dict"])
        else:
            base_model.load_state_dict(base_ckpt)
        base_model.eval()

        pre = AgePreprocess(size=(input_size, input_size)).to(device)

        defended = SmoothedMetric(
            base_model=base_model,
            preprocess=pre,
            n_samples=n_samples,
            sigma=sigma,
            aggregator_mode=cfg["aggregator_mode"],
            hidden=cfg.get("hidden", 32),
            psi_depth=cfg.get("psi_depth", 1),
            rho_depth=cfg.get("rho_depth", 1),
            trim_ratio=cfg.get("trim_ratio", 0.1),
        ).to(device)

        ckpt_path = cfg.get("ckpt", None)
        if ckpt_path is not None and os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)

            if "base_state" in ckpt:
                defended.base.load_state_dict(ckpt["base_state"])

            if "agg_state" in ckpt and any(p.numel() > 0 for p in defended.agg.parameters()):
                defended.agg.load_state_dict(ckpt["agg_state"])

            defended.sigma = ckpt.get("sigma", sigma) if ckpt.get("sigma", None) is not None else sigma
            #defended.n = ckpt.get("n", n_samples) if ckpt.get("n", None) is not None else n_samples

        res = compute_local_bounds_L2_ours_streaming(
            defended,
            val_loader,
            device,
            sigma=sigma,
            eps2=eps,
            max_batches=None,
            steps_L=3,
            step_size_L=0.25,
            n_samples = n,
            n_L=64,
            seed=0,
        )

        row = {
            "method": cfg["name"],
            "eps": float(eps),
            "sigma": float(defended.sigma),
            "n": int(defended.n),
            "MSE" : res["stats"]["MSE"],
            "L_mean": res["stats"]["L_mean"],
            "L_p90": res["stats"]["L_p90"], 
            "L_p99": res["stats"]["L_p99"],
            "C_mean": res["stats"]["C_mean"],
            "C_p90": res["stats"]["C_p90"], 
            "C_p99": res["stats"]["C_p99"],
            "B_mean": res["stats"]["B_mean"],
            "B_p90": res["stats"]["B_p90"], 
            "B_p99": res["stats"]["B_p99"],
        }

        print(row)

        # сохраняем в память
        results.append(row)

        # append CSV
        with open(results_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                row["method"],
                row["eps"],
                row["sigma"],
                row["n"],
                row["MSE"],
                row["L_mean"],
                row["L_p90"], 
                row["L_p99"],
                row["C_mean"],
                row["C_p90"], 
                row["C_p99"],
                row["B_mean"],
                row["B_p90"], 
                row["B_p99"],
            ])
