import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

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

        return x / 100 




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
            dir_img = all_imgs[9000:9010]

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
        # Y: [B, n]
        return Y.mean(dim=1)


class MedianAggregator(nn.Module):
    def forward(self, Y):
        # Y: [B, n]
        return Y.median(dim=1).values


class TrimmedMeanAggregator(nn.Module):
    def __init__(self, trim_ratio=0.25):
        super().__init__()
        self.trim_ratio = float(trim_ratio)

    def forward(self, Y):
        # Y: [B, n]
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
    def __init__(self, mode="lightweight", hidden=32):
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
                depth=3,
            )

            self.rho = make_mlp(
                in_dim=hidden,
                hidden=hidden,
                out_dim=2,
                depth=3,
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


class SmoothedMetric(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        preprocess: nn.Module,
        n_samples: int = 16,
        sigma: float = 4.0/255.0,
        aggregator_mode: str = "deepsets",
        aggregator: nn.Module | None = None,
        trim_ratio: float = 0.25,
        hidden: int = 32,
    ):
        super().__init__()
        self.base = base_model
        self.pre = preprocess
        self.n = int(n_samples)
        self.sigma = float(sigma)

        if aggregator is not None:
            self.agg = aggregator
        else:
            if aggregator_mode in ["lightweight", "deepsets"]:
                self.agg = SoftQuantileAggregator(mode=aggregator_mode, hidden=hidden)
            elif aggregator_mode == "mean":
                self.agg = MeanAggregator()
            elif aggregator_mode == "median":
                self.agg = MedianAggregator()
            elif aggregator_mode == "trim_mean":
                self.agg = TrimmedMeanAggregator(trim_ratio=trim_ratio)
            else:
                raise ValueError(f"Unknown aggregator_mode: {aggregator_mode}")

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



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

metric = MetricModel(device=device, model_path="data/KonCept512.pth", backbone_path="data/inceptionresnetv2-520b38e4.pth")
base_model = metric.model  # model_qa, output [B,1]


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
    metric = MetricModel(device=device, model_path="data/KonCept512.pth", backbone_path="data/inceptionresnetv2-520b38e4.pth")
    base_model = metric.model
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
            eps=1.0,
            steps=10,
            step_size=0.1,
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




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

metric = MetricModel(device=device, model_path="data/KonCept512.pth", backbone_path="data/inceptionresnetv2-520b38e4.pth")
base_model = metric.model  # model_qa, output [B,1]

pre = NormalizeOnly(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)).to(device)




#lrs = [1e-5, 1e-4, 1e-3]
#ns = [64, 128, 256, 512, 1024, 2048]

epsilons = [8/255]
sigma = 8/255
n = 128

methods = [
    {"name": "deepsets", "aggregator_mode": "deepsets", "ckpt": "best_model_8_255/last.pth"},
    {"name": "mean", "aggregator_mode": "mean", "ckpt": None},
    {"name": "median", "aggregator_mode": "median", "ckpt": None},
    {"name": "trim_mean", "aggregator_mode": "trim_mean", "trim_ratio": 0.1, "ckpt": None},
]

results = []

from torch.utils.data import DataLoader


path_train = 'data/koniq'

ds_train = MyCustomDataset(path_gt=path_train, mode='train')
ds_val   = MyCustomDataset(path_gt=path_train, mode='test')

train_loader = DataLoader(ds_train, batch_size=16, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)
val_loader   = DataLoader(ds_val, batch_size=1, shuffle=False,
                          num_workers=4, pin_memory=True)


import csv
from pathlib import Path

results_path = Path("aggregator_results/aggregator_results_8_255_n=128_adv.csv")

# если файл ещё не существует — создаём и пишем header
if not results_path.exists():
    with open(results_path, "w", newline="") as f:
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

for method_cfg in methods:
    for eps in epsilons:
        metric = MetricModel(device=device, model_path="data/KonCept512.pth", backbone_path="data/inceptionresnetv2-520b38e4.pth")
        base_model = metric.model  # model_qa, output [B,1]
        defended = SmoothedMetric(
            base_model=base_model,
            preprocess=pre,
            n_samples=n,
            sigma=sigma,
            aggregator_mode=method_cfg["aggregator_mode"],
            trim_ratio=method_cfg.get("trim_ratio", 0.1),
        ).to(device)

        ckpt_path = method_cfg.get("ckpt", None)
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location=device)
            defended.base.load_state_dict(ckpt["base_state"])

            if "agg_state" in ckpt and any(p.numel() > 0 for p in defended.agg.parameters()):
                defended.agg.load_state_dict(ckpt["agg_state"])

            #defended.sigma = ckpt.get("sigma", sigma)
            defended.n = n

        defended.eval()

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
            "method": method_cfg["name"],
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
        with open(results_path, "a", newline="") as f:
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
