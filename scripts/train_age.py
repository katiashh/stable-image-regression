
from tqdm import tqdm
import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import csv
import json
from pathlib import Path 
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
device='cuda'
H=32
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
import os, random
import pandas as pd
import torch
from torchvision import transforms

import torch.nn as nn

from tqdm import tqdm
from typing import Any, Dict, List

import os
import torch
import torch.nn as nn
from torchvision import transforms



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

        return img, age

# -*- coding: utf-8 -*-

import os
import time
import copy
import random
import numpy as np
import pandas as pd

import torch
import torch.optim as optim
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.nn import init


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
            # nn.AvgPool2d(2, 2) # paper has this layer, but official codes don't.
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
            # nn.MaxPool2d(2, 2) # paper has this layer, but official codes don't.
        )
        
        # fusion block
        self.funsion_block_stream1_stage_3_before_PB = nn.Sequential(
            nn.Conv2d(32, 10, 1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(8, 8)
        )
        self.funsion_block_stream1_stage_3_prediction_block = nn.Sequential(
            nn.Dropout(0.2, ),
            nn.Linear(10 * 4 * 4, self.stage_num[2]),
            nn.ReLU()
        )
        
        self.funsion_block_stream1_stage_2_before_PB = nn.Sequential(
            nn.Conv2d(32, 10, 1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(4, 4)
        )
        self.funsion_block_stream1_stage_2_prediction_block = nn.Sequential(
            nn.Dropout(0.2, ),
            nn.Linear(10 * 4 * 4, self.stage_num[1]),
            nn.ReLU()
        )
        
        self.funsion_block_stream1_stage_1_before_PB = nn.Sequential(
            nn.Conv2d(32, 10, 1, padding=0),
            nn.ReLU(),
            # nn.AvgPool2d(2, 2) # paper has this layer, but official codes don't.
        )
        self.funsion_block_stream1_stage_1_prediction_block = nn.Sequential(
            nn.Dropout(0.2, ),
            nn.Linear(10 * 8 * 8, self.stage_num[0]),
            nn.ReLU()
        )
        
        # stream2
        self.funsion_block_stream2_stage_3_before_PB = nn.Sequential(
            nn.Conv2d(16, 10, 1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(8, 8)
        )
        self.funsion_block_stream2_stage_3_prediction_block = nn.Sequential(
            nn.Dropout(0.2, ),
            nn.Linear(10 * 4 * 4, self.stage_num[2]),
            nn.ReLU()
        )
        
        self.funsion_block_stream2_stage_2_before_PB = nn.Sequential(
            nn.Conv2d(16, 10, 1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(4, 4)
        )
        self.funsion_block_stream2_stage_2_prediction_block = nn.Sequential(
            nn.Dropout(0.2, ),
            nn.Linear(10 * 4 * 4, self.stage_num[1]),
            nn.ReLU()
        )
        
        self.funsion_block_stream2_stage_1_before_PB = nn.Sequential(
            nn.Conv2d(16, 10, 1, padding=0),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2) # paper has this layer, but official codes don't.
        )
        self.funsion_block_stream2_stage_1_prediction_block = nn.Sequential(
            nn.Dropout(0.2, ),
            nn.Linear(10 * 8 * 8, self.stage_num[0]),
            nn.ReLU()
        )
        
        self.stage3_FC_after_PB = nn.Sequential(
            nn.Linear(self.stage_num[0], 2 * self.stage_num[0]),
            nn.ReLU()
        )
        self.stage3_prob = nn.Sequential(
            nn.Linear(2 * self.stage_num[0], self.stage_num[0]),
            nn.ReLU()
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
            nn.ReLU()
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
            nn.ReLU()
        )
        self.stage1_index_offsets = nn.Sequential(
            nn.Linear(2 * self.stage_num[0], self.stage_num[0]),
            nn.Tanh()
        )
        self.stage1_delta_k = nn.Sequential(
            nn.Linear(10 * 8 * 8, 1),
            nn.Tanh()
        )
        self.init_params()
    
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
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
        
        embedding_stage1_after_PB = torch.mul(self.funsion_block_stream1_stage_1_prediction_block(embedding_stream1_stage1_before_PB),
                                              self.funsion_block_stream2_stage_1_prediction_block(embedding_stream2_stage1_before_PB))
        embedding_stage2_after_PB = torch.mul(self.funsion_block_stream1_stage_2_prediction_block(embedding_stream1_stage2_before_PB),
                                              self.funsion_block_stream2_stage_2_prediction_block(embedding_stream2_stage2_before_PB))
        embedding_stage3_after_PB = torch.mul(self.funsion_block_stream1_stage_3_prediction_block(embedding_stream1_stage3_before_PB),
                                              self.funsion_block_stream2_stage_3_prediction_block(embedding_stream2_stage3_before_PB))
        
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
        stage2_regress = stage2_regress / (self.stage_num[0] * (1 + self.lambda_delta * stage1_delta_k) *
                                           (self.stage_num[1] * (1 + self.lambda_delta * stage2_delta_k)))
        
        for index in range(self.stage_num[2]):
            stage3_regress = stage3_regress + (index + self.lambda_index * index_offset_stage3[:, index]) * prob_stage_3[:, index]
        stage3_regress = torch.unsqueeze(stage3_regress, 1)
        stage3_regress = stage3_regress / (self.stage_num[0] * (1 + self.lambda_delta * stage1_delta_k) *
                                           (self.stage_num[1] * (1 + self.lambda_delta * stage2_delta_k)) *
                                           (self.stage_num[2] * (1 + self.lambda_delta * stage3_delta_k))
                                           )
        regress_class = (stage1_regress + stage2_regress + stage3_regress) * self.class_range
        regress_class = torch.squeeze(regress_class, 1)
        return regress_class

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate_metrics(outputs, labels):
    mae = torch.sum(torch.abs(outputs - labels)).item()
    ca3 = torch.sum(torch.abs(outputs - labels) < 3).item()
    ca5 = torch.sum(torch.abs(outputs - labels) < 5).item()
    return mae, ca3, ca5


class AgeMetricModel(torch.nn.Module):
    def __init__(self, device, model_path, image_size=64):
        super().__init__()
        self.device = device

        model = SSRNet(image_size=image_size).to(device)
        checkpoint = torch.load(model_path, map_location=device)

        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.eval().to(device)
        self.model = model
        self.lower_better = False

    def forward(self, image, inference=False):
        out = self.model(image)  # [B] or [B,1]
        if out.ndim > 1:
            out = out.view(-1)

        if inference:
            return out.detach().cpu().numpy()
        return out

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

class SmoothedMetric(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        preprocess: nn.Module,
        n_samples: int = 8,
        sigma: float = 4.0 / 255.0,
        hidden=32,
        aggregator_mode: str = "lightweight",
        psi_depth=1,
        rho_depth=1,
    ):
        super().__init__()
        self.base = base_model
        self.pre = preprocess
        self.n = int(n_samples)
        self.sigma = float(sigma)
        self.agg = SoftQuantileAggregator(
            mode=aggregator_mode,
            hidden=hidden,
            psi_depth=psi_depth,
            rho_depth=rho_depth,
        )

    def forward(self, x, training=True):
        B = x.shape[0]

        noises = torch.randn(self.n, *x.shape, device=x.device, dtype=x.dtype) * self.sigma
        x_rep = (x.unsqueeze(0) + noises).clamp(0.0, 1.0)      # [n,B,3,H,W]

        x_flat = x_rep.reshape(self.n * B, *x.shape[1:])       # [n*B,3,H,W]
        x_flat = self.pre(x_flat)

        y_flat = self.base(x_flat)                              # [n*B] or [n*B,1]
        if y_flat.ndim > 1:
            y_flat = y_flat.view(-1)

        Y = y_flat.view(self.n, B).transpose(0, 1)             # [B,n]
        g = self.agg(Y)                                        # [B]
        return g, Y


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

# -------------------------
#  Learnable Aggregators
# -------------------------
class SoftQuantileAggregator(nn.Module):
    """
    Soft-quantile-like aggregator for scalar samples Y: [B, n].
    Two modes:
      - lightweight: t=mean(Y), alpha=softplus(log_alpha) (global scalar)
      - deepsets: predict t and alpha per-example from the set {Y_i}
    """
    def __init__(self, mode="lightweight", hidden=32, psi_depth=1, rho_depth=1):
        super().__init__()
        assert mode in ["lightweight", "deepsets"]
        self.mode = mode
        self.hidden = hidden

        if mode == "lightweight":
            self.log_alpha = nn.Parameter(torch.tensor(0.0))
        else:
            self.psi = make_mlp(
                in_dim=1,
                hidden=hidden,
                out_dim=hidden,
                depth=psi_depth,
            )

            self.rho = make_mlp(
                in_dim=hidden,
                hidden=hidden,
                out_dim=2,
                depth=rho_depth,
            )

    def forward(self, Y):
        """
        Y: [B, n] float
        returns: g [B]
        """
        B, n = Y.shape

        if self.mode == "lightweight":
            t = Y.mean(dim=1)  # [B]
            alpha = F.softplus(self.log_alpha) + 1e-6  # scalar > 0
            alpha = alpha.expand(B)  # [B]
        else:
            # DeepSets over scalar samples
            Y_flat = Y.reshape(B * n, 1)
            emb = self.psi(Y_flat).reshape(B, n, self.hidden)  # [B,n,h]
            s = emb.mean(dim=1)  # [B,h]
            out = self.rho(s)    # [B,2]
            t = out[:, 0]
            alpha = F.softplus(out[:, 1]) + 1e-6

        # soft weights around t
        dist = (Y - t.unsqueeze(1)).abs()       # [B,n]
        logits = -alpha.unsqueeze(1) * dist     # [B,n]
        w = torch.softmax(logits, dim=1)        # [B,n]
        g = (w * Y).sum(dim=1)                  # [B]
        return g


# -------------------------
#  Smoothed defended model
# -------------------------
class SmoothedMetric(nn.Module):
    """
    Wrap a base scalar regressor f(x)->[B,1] with randomized smoothing:
      Y_i = f(x + noise_i)
      g = Phi(Y_1..Y_n)
    Includes optional gradient penalty for Lipschitz control.
    """
    def __init__(
        self,
        base_model: nn.Module,
        preprocess: nn.Module,
        n_samples: int = 16,
        sigma: float = 4.0/255.0,
        hidden=32,
        aggregator_mode: str = "lightweight",  # or "deepsets"
        psi_depth=1,
        rho_depth=1
    ):
        super().__init__()
        self.base = base_model
        self.pre = preprocess
        self.n = int(n_samples)
        self.sigma = float(sigma)
        self.agg = SoftQuantileAggregator(mode=aggregator_mode, hidden=hidden, psi_depth=psi_depth, rho_depth=rho_depth)

    def forward(self, x, training=True):
        """
        x: [B,3,H,W] in [0,1]
        returns: g [B] and optionally Y [B,n]
        """
        B = x.shape[0]

        # sample Gaussian noise for each Monte Carlo draw
        # noise is applied in input space; you can also apply after resize if you prefer
        noises = torch.randn(self.n, *x.shape, device=x.device, dtype=x.dtype) * self.sigma
        x_rep = x.unsqueeze(0) + noises                  # [n,B,3,H,W]
        x_rep = x_rep.clamp(0.0, 1.0)                    # keep valid range

        # flatten and preprocess
        x_flat = x_rep.reshape(self.n * B, *x.shape[1:]) # [n*B,3,H,W]
        x_flat = self.pre(x_flat)

        y_flat = self.base(x_flat)                       # [n*B,1] (or [n*B])
        y_flat = y_flat.view(self.n, B).transpose(0, 1)  # [B,n]
        g = self.agg(y_flat)                             # [B]
        return g, y_flat


# -------------------------
#  Losses / Regularizers
# -------------------------
def gradient_penalty_on_output(output_sum, inputs):
    """
    Computes || d output_sum / d inputs ||_2^2 averaged over batch.
    output_sum: scalar (e.g., g.sum())
    inputs: [B,3,H,W] requires_grad=True
    """
    grad = torch.autograd.grad(
        outputs=output_sum, inputs=inputs,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_sq = grad.pow(2).flatten(1).mean(dim=1).mean()
    return grad_sq.mean()

def sensitivity_penalty(agg_module, Y, C=1.5):
    """
    Y: [B, n] noisy predictions (requires_grad=True)
    Returns: hinge penalty on S = sum_i |dPhi/dY_i|.
    """
    Y_req = Y.detach().clone().requires_grad_(True)
    g = agg_module(Y_req)  # [B]
    # sum outputs to get scalar for autograd
    grad = torch.autograd.grad(
        outputs=g.sum(), inputs=Y_req,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]  # [B,n]
    S = grad.abs().sum(dim=1)  # [B]
    # hinge: max(0, S - C)^2
    return torch.relu(S - C).pow(2).mean(), S.mean().detach()


class AdaptiveLossBalancer(nn.Module):
    """
    Делает вклады loss-компонент примерно равными:
      w_i ~ 1 / EMA(loss_i)
      total = sum_i w_i * loss_i

    Плюсы:
      - не нужно подбирать lambda_lip/lambda_sens
      - очень простой, стабильный
    """
    def __init__(self, n_terms: int, momentum: float = 0.99, eps: float = 1e-8):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("ema", torch.ones(n_terms))  # старт с 1 чтобы не взрывалось

    @torch.no_grad()
    def update(self, losses: torch.Tensor):
        # losses: [n_terms] (detach уже ок)
        self.ema.mul_(self.momentum).add_(losses * (1.0 - self.momentum))

    def forward(self, losses: list[torch.Tensor]):
        """
        losses: list of scalars (тензоры-скаляры)
        returns: (total_loss, weights_tensor)
        """
        vals = torch.stack([l.detach() for l in losses])  # [n_terms]
        self.update(vals)

        # веса обратно пропорциональны среднему масштабу
        w = 1.0 / (self.ema + self.eps)
        # чтобы суммарный масштаб оставался примерно постоянным
        w = w / (w.sum() + self.eps) * len(losses)

        total = 0.0
        for wi, li in zip(w, losses):
            total = total + wi * li
        return total, w.detach()


balancer = AdaptiveLossBalancer(n_terms=3, momentum=0.99).to(device)


def finetune_smoothed_age(
    model,
    train_loader,
    val_loader,
    device,
    *,
    epochs=10,
    lr=1e-5,
    lambda_lip=1.0,
    lip_on="f",
    max_grad_norm=1.0,
    lambda_sens=1.0,
    C_sens=1.5,
    name="",
    save_root="runs_age",
    seed=None,
):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    os.makedirs(save_root, exist_ok=True)

    model.to(device)
    model.train()

    for p in model.base.parameters():
        p.requires_grad = True

    optimizer = torch.optim.Adam(
        list(model.base.parameters()) + list(model.agg.parameters()),
        lr=lr
    )

    reg_loss = nn.SmoothL1Loss(beta=3.0)
    balancer = AdaptiveLossBalancer(n_terms=3, momentum=0.99).to(device)

    def eval_epoch():
        if val_loader is None:
            return None, None

        model.eval()
        total_loss, total_mae, n_items = 0.0, 0.0, 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device).float()
                y = y.to(device).view(-1).float()

                g, _ = model(x, training=False)

                loss = reg_loss(g, y)
                mae = torch.mean(torch.abs(g - y))

                total_loss += loss.item() * x.size(0)
                total_mae += mae.item() * x.size(0)
                n_items += x.size(0)

        model.train()
        return total_loss / max(1, n_items), total_mae / max(1, n_items)

    best_val_mae = float("inf")

    for ep in range(1, epochs + 1):
        running = {
            "loss": 0.0,
            "main": 0.0,
            "lip": 0.0,
            "sens": 0.0,
            "mae": 0.0,
            "S": 0.0,
            "w_main": 0.0,
            "w_lip": 0.0,
            "w_sens": 0.0,
        }
        n_seen = 0

        for x, y in tqdm(train_loader, desc=f"[{name}] ep {ep}/{epochs}"):
            x = x.to(device).float()
            y = y.to(device).view(-1).float()

            optimizer.zero_grad(set_to_none=True)

            x_gp = x.detach().clone()
            x_gp.requires_grad_(lambda_lip > 0)

            g, Y = model(x_gp, training=True)

            main_loss = reg_loss(g, y)
            mae = torch.mean(torch.abs(g - y))

            lip_pen = torch.tensor(0.0, device=device)
            if lambda_lip > 0:
                if lip_on == "g":
                    lip_pen = gradient_penalty_on_output(g.sum(), x_gp)
                else:
                    x_clean = model.pre(x_gp)
                    fx = model.base(x_clean).view(-1)
                    lip_pen = gradient_penalty_on_output(fx.sum(), x_gp)

            sens_pen = torch.tensor(0.0, device=device)
            S_mean = 0.0
            if lambda_sens > 0:
                sens_pen, S_mean = sensitivity_penalty(model.agg, Y, C=C_sens)

            loss, w = balancer([main_loss, lip_pen, sens_pen])

            loss.backward()

            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    list(model.base.parameters()) + list(model.agg.parameters()),
                    max_grad_norm
                )

            optimizer.step()

            bs = x.size(0)
            running["loss"] += float(loss.item()) * bs
            running["main"] += float(main_loss.item()) * bs
            running["lip"] += float(lip_pen.item()) * bs
            running["sens"] += float(sens_pen.item()) * bs
            running["mae"] += float(mae.item()) * bs
            running["S"] += float(S_mean) * bs
            running["w_main"] += float(w[0]) * bs
            running["w_lip"] += float(w[1]) * bs
            running["w_sens"] += float(w[2]) * bs
            n_seen += bs

        train_loss = running["loss"] / max(1, n_seen)
        train_main = running["main"] / max(1, n_seen)
        train_lip = running["lip"] / max(1, n_seen)
        train_sens = running["sens"] / max(1, n_seen)
        train_mae = running["mae"] / max(1, n_seen)

        val_loss, val_mae = eval_epoch()

        print(
            f"Epoch {ep:03d} | "
            f"train loss {train_loss:.4f} | main {train_main:.4f} | mae {train_mae:.4f} | "
            f"lip {train_lip:.4f} | sens {train_sens:.4f} | "
            f"val loss {val_loss:.4f} | val mae {val_mae:.4f}"
        )

        ckpt = {
            "base_state": model.base.state_dict(),
            "agg_state": model.agg.state_dict(),
            "sigma": model.sigma,
            "n": model.n,
            "epoch": ep,
            "val_mae": val_mae,
        }

        torch.save(ckpt, os.path.join(save_root, "last.pth"))

        if val_mae is not None and val_mae < best_val_mae:
            best_val_mae = float(val_mae)
            torch.save(ckpt, os.path.join(save_root, "best.pth"))

    return model

batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 64
BASE_DIR = "data/wiki_crop/age_split"

TRAIN_DIR = os.path.join(BASE_DIR, "train", "images")
VAL_DIR   = os.path.join(BASE_DIR, "val", "images")
TEST_DIR  = os.path.join(BASE_DIR, "test", "images")

def choose_labels_csv(split_dir):
    """Use the rebuilt list when it exists; otherwise use labels.csv."""
    rebuilt = os.path.join(split_dir, "labels_rebuilt.csv")
    original = os.path.join(split_dir, "labels.csv")
    return rebuilt if os.path.exists(rebuilt) else original


TRAIN_CSV = choose_labels_csv(os.path.join(BASE_DIR, "train"))
VAL_CSV   = choose_labels_csv(os.path.join(BASE_DIR, "val"))
TEST_CSV  = choose_labels_csv(os.path.join(BASE_DIR, "test"))


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

age_metric = AgeMetricModel(
    device=device,
    model_path="trained_models/ssrnet_best.pth",
    image_size=64,
)

base_model = age_metric.model
pre = AgePreprocess(size=(64, 64)).to(device)

defended = SmoothedMetric(
    base_model=base_model,
    preprocess=pre,
    n_samples=8,
    sigma=8 / 255.0,
    aggregator_mode="deepsets",   # сначала можно "lightweight"
    psi_depth=3,
    rho_depth=3,
    hidden=32,
).to(device)

model = finetune_smoothed_age(
    model=defended,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    epochs=20,
    lr=1e-5,
    lambda_lip=1.0,
    lip_on="f",
    max_grad_norm=1.0,
    lambda_sens=1.0,
    C_sens=1.0,
    name="age_agg",
    save_root="runs_age_agg_8_255",
    seed=0,
)
