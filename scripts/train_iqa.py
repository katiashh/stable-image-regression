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


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False) # verify bias false
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001, # value found in tensorflow
                                 momentum=0.1, # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_5b(nn.Module):

    def __init__(self):
        super(Mixed_5b, self).__init__()

        self.branch0 = BasicConv2d(192, 96, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(192, 48, kernel_size=1, stride=1),
            BasicConv2d(48, 64, kernel_size=5, stride=1, padding=2)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(192, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1),
            BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(192, 64, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block35(nn.Module):

    def __init__(self, scale=1.0):
        super(Block35, self).__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(320, 32, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(320, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(320, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 48, kernel_size=3, stride=1, padding=1),
            BasicConv2d(48, 64, kernel_size=3, stride=1, padding=1)
        )

        self.conv2d = nn.Conv2d(128, 320, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_6a(nn.Module):

    def __init__(self):
        super(Mixed_6a, self).__init__()

        self.branch0 = BasicConv2d(320, 384, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv2d(320, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Block17(nn.Module):

    def __init__(self, scale=1.0):
        super(Block17, self).__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(1088, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(1088, 128, kernel_size=1, stride=1),
            BasicConv2d(128, 160, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(160, 192, kernel_size=(7,1), stride=1, padding=(3,0))
        )

        self.conv2d = nn.Conv2d(384, 1088, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_7a(nn.Module):

    def __init__(self):
        super(Mixed_7a, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(1088, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(1088, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 288, kernel_size=3, stride=2)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(1088, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 288, kernel_size=3, stride=1, padding=1),
            BasicConv2d(288, 320, kernel_size=3, stride=2)
        )

        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block8(nn.Module):

    def __init__(self, scale=1.0, noReLU=False):
        super(Block8, self).__init__()

        self.scale = scale
        self.noReLU = noReLU

        self.branch0 = BasicConv2d(2080, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(2080, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 224, kernel_size=(1,3), stride=1, padding=(0,1)),
            BasicConv2d(224, 256, kernel_size=(3,1), stride=1, padding=(1,0))
        )

        self.conv2d = nn.Conv2d(448, 2080, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out


class InceptionResNetV2(nn.Module):

    def __init__(self, num_classes=1001):
        super(InceptionResNetV2, self).__init__()
        # Special attributs
        self.input_space = None
        self.input_size = (299, 299, 3)
        self.mean = None
        self.std = None
        # Modules
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.maxpool_5a = nn.MaxPool2d(3, stride=2)
        self.mixed_5b = Mixed_5b()
        self.repeat = nn.Sequential(
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17)
        )
        self.mixed_6a = Mixed_6a()
        self.repeat_1 = nn.Sequential(
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10)
        )
        self.mixed_7a = Mixed_7a()
        self.repeat_2 = nn.Sequential(
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20)
        )
        self.block8 = Block8(noReLU=True)
        self.conv2d_7b = BasicConv2d(2080, 1536, kernel_size=1, stride=1)
        self.avgpool_1a = nn.AvgPool2d(8, count_include_pad=False)
        self.last_linear = nn.Linear(1536, num_classes)

    def features(self, input):
        x = self.conv2d_1a(input)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.maxpool_5a(x)
        x = self.mixed_5b(x)
        x = self.repeat(x)
        x = self.mixed_6a(x)
        x = self.repeat_1(x)
        x = self.mixed_7a(x)
        x = self.repeat_2(x)
        x = self.block8(x)
        x = self.conv2d_7b(x)
        return x

    def logits(self, features):
        x = self.avgpool_1a(features)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

def inceptionresnetv2(weights_path, num_classes=1000, pretrained='imagenet'):
    r"""InceptionResNetV2 model architecture from the
    `"InceptionV4, Inception-ResNet..." <https://arxiv.org/abs/1602.07261>`_ paper.
    """
    if pretrained:

        # both 'imagenet'&'imagenet+background' are loaded from same parameters
        model = InceptionResNetV2(num_classes=1001)
        model.load_state_dict(torch.load(weights_path, map_location=lambda storage, loc: storage))

        if pretrained == 'imagenet':
            new_last_linear = nn.Linear(1536, 1000)
            new_last_linear.weight.data = model.last_linear.weight.data[1:]
            new_last_linear.bias.data = model.last_linear.bias.data[1:]
            model.last_linear = new_last_linear

        model.input_space = 'RGB'
        model.input_size = [3, 299, 299]
        model.input_range = [0, 1]

        model.mean = [0.5, 0.5, 0.5]
        model.std = [0.5, 0.5, 0.5]
    else:
        model = InceptionResNetV2(num_classes=num_classes)
    return model
    
class model_qa(nn.Module):
    def __init__(self, weights_path, num_classes,**kwargs):
        super(model_qa,self).__init__()
        base_model = inceptionresnetv2(weights_path, num_classes=1000, pretrained='imagenet')
        self.base= nn.Sequential(*list(base_model.children())[:-1])
        self.fc = nn.Sequential(
            nn.Linear(1536, 2048),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(2048),
            nn.Dropout(p=0.25),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=0.25),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),         
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self,x):
        x = self.base(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x 




class MetricModel(torch.nn.Module):
    def __init__(self, device, model_path, backbone_path=None):
        super().__init__()
        self.device = device

        model = model_qa(backbone_path, num_classes=1).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval().to(device)
        self.model = model
        self.lower_better = False
    
    def forward(self, image, inference=False):
        # transforms.Compose doesn't accept torch tensors
        out = self.model(
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(transforms.Resize([512, 384])(image))
        )
        if inference:
            return out.detach().cpu().numpy()[0][0].item()
        else:
            return out


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
        return x, y 


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


def finetune_smoothed(
    model,  # SmoothedMetric
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    device: torch.device,
    *,
    epochs: int = 10,
    lr: float = 1e-5,
    # lmbdas здесь используются как on/off
    lambda_lip: float = 1.0,
    lip_on: str = "f",  # "f" or "g"
    max_grad_norm: float | None = 1.0,
    lambda_sens: float = 1.0,
    C_sens: float = 1.5,
    name: str = "",
    save_root: str = "runs2",
    seed: int | None = None,
):
    """
    Fine-tune base weights + aggregator without changing architecture.
    Логирует метрики в CSV/JSONL, сохраняет last/best чекпоинты.
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    model.to(device)
    model.train()

    # base trainable
    for p in model.base.parameters():
        p.requires_grad = True

    optim = torch.optim.Adam(list(model.base.parameters()) + list(model.agg.parameters()), lr=lr)
    mse_loss = nn.MSELoss()

    # balancer: mse, lip, sens
    balancer = AdaptiveLossBalancer(n_terms=3, momentum=0.99).to(device)

    run_dir = save_root


    def eval_epoch():
        if val_loader is None:
            return None
        model.eval()
        total_mse, n_items = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device).view(-1).float()
                g, _ = model(x, training=False)
                loss = mse_loss(g / 100, y / 100)
                total_mse += loss.item() * x.size(0)
                n_items += x.size(0)
        model.train()
        return total_mse / max(1, n_items)

    best_val = float("inf")
    best_path = None
    last_path = None

    for ep in range(1, epochs + 1):
        running = {
            "loss": 0.0, "mse": 0.0, "lip": 0.0, "sens": 0.0, "S": 0.0,
            "w_mse": 0.0, "w_lip": 0.0, "w_sens": 0.0
        }
        n_seen = 0

        for x, y in tqdm(train_loader, desc=f"[{name}] ep {ep}/{epochs}"):
            x = x.to(device).float()
            y = y.to(device).view(-1).float()

            optim.zero_grad(set_to_none=True)

            # Need gradients w.r.t. x for gradient penalty
            x_gp = x.detach().clone()
            x_gp.requires_grad_(lambda_lip > 0)

            g, Y = model(x_gp, training=True)

            mse = mse_loss(g / 100, y / 100)

            # lip penalty
            lip_pen = torch.tensor(0.0, device=device)
            if lambda_lip > 0:
                if lip_on == "g":
                    lip_pen = gradient_penalty_on_output(g.sum(), x_gp)
                else:
                    x_clean = model.pre(x_gp)
                    fx = model.base(x_clean).view(-1)
                    lip_pen = gradient_penalty_on_output(fx.sum(), x_gp)

            # sens penalty
            sens_pen = torch.tensor(0.0, device=device)
            S_mean = 0.0
            if lambda_sens > 0:
                sens_pen, S_mean = sensitivity_penalty(model.agg, Y, C=C_sens)

            # balanced loss
            loss, w = balancer([mse, lip_pen, sens_pen])

            loss.backward()

            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    list(model.base.parameters()) + list(model.agg.parameters()),
                    max_grad_norm
                )

            optim.step()

            bs = x.size(0)
            running["loss"] += float(loss.item()) * bs
            running["mse"] += float(mse.item()) * bs
            running["lip"] += float(lip_pen.item()) * bs
            running["sens"] += float(sens_pen.item()) * bs
            running["S"] += float(S_mean) * bs
            running["w_mse"] += float(w[0]) * bs
            running["w_lip"] += float(w[1]) * bs
            running["w_sens"] += float(w[2]) * bs
            n_seen += bs

        train_loss = running["loss"] / max(1, n_seen)
        train_mse = running["mse"] / max(1, n_seen)
        train_lip = running["lip"] / max(1, n_seen)
        train_sens = running["sens"] / max(1, n_seen)
        train_S = running["S"] / max(1, n_seen)
        w_mse = running["w_mse"] / max(1, n_seen)
        w_lip = running["w_lip"] / max(1, n_seen)
        w_sens = running["w_sens"] / max(1, n_seen)

        val_mse = eval_epoch()

        if val_mse is None:
            print(
                f"Epoch {ep:03d} | train loss {train_loss:.6f} | mse {train_mse:.6f} "
                f"| lip {train_lip:.6f} | sens {train_sens:.6f} | S {train_S:.6f} "
                f"| w=[{w_mse:.2f}, {w_lip:.2f}, {w_sens:.2f}]"
            )
        else:
            print(
                f"Epoch {ep:03d} | train loss {train_loss:.6f} | mse {train_mse:.6f} "
                f"| lip {train_lip:.6f} | sens {train_sens:.6f} | S {train_S:.6f} "
                f"| val mse {val_mse:.6f} | w=[{w_mse:.2f}, {w_lip:.2f}, {w_sens:.2f}]"
            )

   

        # save ckpt (FIX: сохраняем model, не defended)
        ckpt = {
            "base_state": model.base.state_dict(),
            "agg_state": model.agg.state_dict(),
            "sigma": getattr(model, "sigma", None),
            "n": getattr(model, "n", None),
            "epoch": ep,

         }

        last_path = os.path.join(run_dir, "last.pth")
        torch.save(ckpt, last_path)

    
        if val_mse is not None and val_mse < best_val:
            best_val = float(val_mse)
            best_path = os.path.join(run_dir, "best.pth")
            torch.save(ckpt, best_path)

    
    return model, _, str(run_dir)




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

metric = MetricModel(device=device, model_path="data/KonCept512.pth", backbone_path="data/inceptionresnetv2-520b38e4.pth")
base_model = metric.model  # model_qa, output [B,1]

pre = NormalizeOnly(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)).to(device)

defended = SmoothedMetric(
    base_model=base_model,
    preprocess=pre,
    n_samples=8,            # для fine-tune можно 8, на тесте 16-32
    sigma=8/255.0,
    aggregator_mode="deepsets",  # стартуй с lightweight
).to(device)

from torch.utils.data import DataLoader


path_train = 'data/koniq'

ds_train = MyCustomDataset(path_gt=path_train, mode='train')
ds_val   = MyCustomDataset(path_gt=path_train, mode='val')

train_loader = DataLoader(ds_train, batch_size=2, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)
val_loader   = DataLoader(ds_val, batch_size=2, shuffle=False,
                          num_workers=4, pin_memory=True)

# Do not run a diagnostic forward pass here. At this resolution, retaining
# its tensors before training wastes several GB of GPU memory.
del defended
if torch.cuda.is_available():
    torch.cuda.empty_cache()





# configs = [...]
# train_loader, val_loader, device уже есть

def make_model_fn(cfg):
    # ВАЖНО: тут ты создаешь SmoothedMetric с нужной архитектурой агрегатора
    # Примерно так (подставь свои аргументы):
    defended = SmoothedMetric(
        base_model=base_model,
        preprocess=pre,
        n_samples=8,
        sigma=8/255.0,
        aggregator_mode="deepsets",
        psi_depth=cfg["psi_depth"],
        rho_depth=cfg["rho_depth"],
        hidden=cfg["hidden"],
    ).to(device)
    return defended

dmodel = SmoothedMetric(
        base_model=base_model,
        preprocess=pre,
        n_samples=8,
        sigma=8/255.0,
        aggregator_mode="deepsets",
        psi_depth=3,
        rho_depth=3,
        hidden=32,
    ).to(device)

sigmas = [8/255]

for s in sigmas:

    dmodel = SmoothedMetric(
            base_model=base_model,
            preprocess=pre,
            n_samples=8,
            sigma=s,
            aggregator_mode="deepsets",
            psi_depth=3,
            rho_depth=3,
            hidden=32,
        ).to(device)

    model, history, run_dir = finetune_smoothed(
                            model=dmodel,
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
                            name="best_model_"+str(int(s*255))+"_255",
                            save_root="best_model_"+str(int(s*255))+"_255",
                            seed=0,
                        )
