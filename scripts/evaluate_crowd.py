from torchvision import models
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

import random
import os
from PIL import Image,ImageFilter,ImageDraw
import numpy as np
import h5py
from PIL import ImageStat
import cv2

def load_data(img_path,train = True):
    gt_path = img_path.replace('.jpg','.h5').replace('images','ground_truth')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])
    if False:
        crop_size = (img.size[0]/2,img.size[1]/2)
        if random.randint(0,9)<= -1:
            
            
            dx = int(random.randint(0,1)*img.size[0]*1./2)
            dy = int(random.randint(0,1)*img.size[1]*1./2)
        else:
            dx = int(random.random()*img.size[0]*1./2)
            dy = int(random.random()*img.size[1]*1./2)
        
        
        
        img = img.crop((dx,dy,crop_size[0]+dx,crop_size[1]+dy))
        target = target[dy:crop_size[1]+dy,dx:crop_size[0]+dx]
        
        
        
        
        if random.random()>0.8:
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    
    
    
    h, w = target.shape
    target = cv2.resize(target,(int(w / 8), int(h / 8)),interpolation=cv2.INTER_CUBIC) * 64
    
    
    return img,target

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
        return 5 # len(self.df)

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


class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512, 512, 512,256,128,64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat,in_channels = 512,dilation = True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        if not load_weights:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()
            frontend_items = list(self.frontend.state_dict().items())
            pretrained_items = list(mod.state_dict().items())

            for i in range(len(frontend_items)):
                frontend_items[i][1].data[:] = pretrained_items[i][1].data[:]
    def forward(self,x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        x = x.sum(dim=(1, 2, 3)) / 100
        x = torch.clip(x, min=0, max=1)
        return x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
                
def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)         

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
        #x_flat = self.pre(x_flat)

        y_flat = self.base(x_flat)                       # [n*B,1] (or [n*B])
        #print('y_flat.shape', y_flat.shape)
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
        x_flat = x_rep.reshape(m * B, *x.shape[1:])                  # [m*B,3,H,W]
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

            f = model.base(z_req).view(-1)  # [B]
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
        f = model.base(z_req).view(-1)
        grad = torch.autograd.grad(f.sum(), z_req, create_graph=False)[0]
        gn = grad.flatten(1).norm(p=2, dim=1).float()  # [B]

        L_max = torch.maximum(L_max, gn)

        del eta, base, u, z, z_req, f, grad, gn

    return L_max.detach()


model = CSRNet()
model = model.cuda()
checkpoint = torch.load('data/partBmodel_best.pth', weights_only=False)
model.load_state_dict(checkpoint['state_dict'])

model.eval()
base_model = model
        
base_model.eval()

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
    model = CSRNet()
    model = model.cuda()
    checkpoint = torch.load('data/partBmodel_best.pth', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    base_model = model
    base_model.eval()

    if step_size is None:
        step_size = eps / steps

    x0 = x.detach()
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

    return torch.clamp(x0 + delta.detach(), clamp_min, clamp_max)

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
        y = y.to(device).float().view(-1) / 100
        y = torch.clip(y, min=0, max=1)

        x = pgd_l2_attack_base_model(
            base_model=base_model,
            x=x,
            y=y,
            eps=20.0,
            steps=10,
            step_size=2.0,
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

class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None,  train=False, seen=0, batch_size=1, num_workers=4):
        if train:
            root = root *4
        random.shuffle(root)
        
        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        
    def __len__(self):
        return self.nSamples
    def __getitem__(self, index):
        assert index <= len(self), 'index range error' 
        
        img_path = self.lines[index]
        
        img,target = load_data(img_path,self.train)
        
        #img = 255.0 * F.to_tensor(img)
        
        #img[0,:,:]=img[0,:,:]-92.8207477031
        #img[1,:,:]=img[1,:,:]-95.2757037428
        #img[2,:,:]=img[2,:,:]-104.877445883


        
        
        if self.transform is not None:
            img = self.transform(img)
        return img,target.sum()

batch_size = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CSRNet()

methods = [
    {"name": "deepsets", "aggregator_mode": "deepsets", "ckpt": "age_crowd_8_255/best.pth"},
    {"name": "mean", "aggregator_mode": "mean", "ckpt": None},
    {"name": "median", "aggregator_mode": "median", "ckpt": None},
    {"name": "trim_mean", "aggregator_mode": "trim_mean", "trim_ratio": 0.1, "ckpt": None},
]
    
model = model.cuda()
checkpoint = torch.load('data/partBmodel_best.pth', weights_only=False)
model.load_state_dict(checkpoint['state_dict'])

model.train()

train_root = 'data/ShanghaiTech_Crowd_Counting_Dataset/part_B_final/train_data/images'
val_root   = 'data/ShanghaiTech_Crowd_Counting_Dataset/part_B_final/test_data/images'

train_list = [os.path.join(train_root, img) for img in os.listdir(train_root)]
val_list   = [os.path.join(val_root, img) for img in os.listdir(val_root)]

train_list = sorted([os.path.join(train_root, img) for img in os.listdir(train_root)])
val_list   = sorted([os.path.join(val_root, img) for img in os.listdir(val_root)])


train_loader = torch.utils.data.DataLoader(
        listDataset(train_list,
                       shuffle=True,
                       transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]), 
                       train=True, 
                       seen=model.seen,
                       batch_size=batch_size),
        batch_size=batch_size)

val_loader = torch.utils.data.DataLoader(
    listDataset(val_list,
                   shuffle=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]),  train=False),
    batch_size=batch_size)

# =========================
# Main
# =========================
SAVE_DIR = 'crowd_res'
sigma = 8/255.
eps = sigma
n = 128
if __name__ == "__main__":
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    batch_size = 1
    num_workers = 1

    sigma = 8 / 255.0
    n_samples = 128

    methods = [
        #{
        #    "name": "deepsets",
        #    "aggregator_mode": "deepsets",
        #    "ckpt": "runs_crowd_agg_8/best.pth",
        #    "hidden": 32,
        #    "psi_depth": 3,
        #    "rho_depth": 3,
        #},
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


    results = []
    results_csv = Path("crowd_res/adv_crowd_aggregator_results_8_256.csv")

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
        model = CSRNet()
    
        model = model.cuda()
        checkpoint = torch.load('data/partBmodel_best.pth', weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])

        model.eval()
        base_model = model
        
        base_model.eval()


        defended = SmoothedMetric(
            base_model=base_model,
            preprocess=None,
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
