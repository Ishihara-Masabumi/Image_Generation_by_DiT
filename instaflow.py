import argparse
import json
import math
import os
from functools import partial
from inspect import isfunction
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from PIL import Image
from torch import einsum, nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    # No More Strided Convolutions or Pooling
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim, scale=1000):
        super().__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = self.scale * time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, emb=None):  # 時間とクラスの統合埋め込みを受け取る
        scale_shift = None
        if exists(self.mlp) and exists(emb):
            emb = self.mlp(emb)
            emb = rearrange(emb, "b c -> b c 1 1")
            scale_shift = emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1),
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        self_condition=False,
        resnet_block_groups=4,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 1, padding=0)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # class embeddings（時間埋め込みと同じ方法で追加）
        self.cond_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.out_dim = default(out_dim, channels)
        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, cond, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        # 時間埋め込み
        t = self.time_mlp(time)
        # クラス埋め込み
        c = self.cond_mlp(cond)
        # 時間とクラスの埋め込みを統合（加算）
        emb = t + c

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, emb)
            h.append(x)

            x = block2(x, emb)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, emb)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, emb)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, emb)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, emb)
        return self.final_conv(x)

# Rectified Flow Class
class InstaFlow(nn.Module):  # nn.Moduleを継承
    def __init__(self, model, timesteps, ln=True):
        super().__init__()  # 最初に呼び出し
        self.model = model
        self.ln = ln
        self.timesteps = timesteps
        if self.timesteps < 2:
            print("timesteps is wrong.")
            return

    def forward(self, x, cond):
        b = x.size(0)
        device = x.device

        # 0〜1の時刻グリッドを生成
        t_grid = torch.linspace(0, 1, steps=self.timesteps, device=device)
        # ランダムに隣接する2つの時刻のインデックスを選択（n in [0, timesteps-1]）
        n = torch.randint(1, self.timesteps, (b,), device=device)
        t = t_grid[n]
        # 画像テンソルとブロードキャスト可能な形状に拡張
        texp = t.view(b, 1, 1, 1)

        z1 = torch.randn_like(x)
        z1 = torch.clamp(z1, min=-3.0, max=3.0)  # 値を[-3, 3]に制限
        zt = (1 - texp) * x + texp * z1
        vtheta = self.model(zt, t, cond)  # U-Net takes zt, t, and cond as input
        vtheta = torch.clamp(vtheta, min=-10.0, max=10.0)  # 出力制限
        batchwise_mse = ((z1 - x - vtheta) ** 2).mean(dim=list(range(1, len(x.shape))))
        tlist = batchwise_mse.detach().cpu().reshape(-1).tolist()
        ttloss = [(tv, tloss) for tv, tloss in zip(t, tlist)]
        return batchwise_mse.mean(), ttloss

    @torch.no_grad()
    def sample(self, z, cond, null_cond, cfg=2.0):
        b = z.size(0)
        dt = 1.0 / (self.timesteps - 1)
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        images = [z]
        for i in tqdm(reversed(range(1, self.timesteps)), desc='sampling loop time step', total=self.timesteps-1):
            t = i / (self.timesteps - 1)
            t = torch.tensor([t] * b).to(z.device)

            vc = self.model(z, t, cond)

            # Classifier Free Guidance
            if null_cond is not None:
                vu = self.model(z, t, null_cond)
                vc = vu + cfg * (vc - vu)

            z = z - dt * vc
            images.append(z)
        return images

# Main Function
def main():
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Choose dataset among: mnist, cifar, fashion_mnist, huggan")
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "cifar", "fashion_mnist", "huggan"],
                        help="Dataset to use (default: mnist)")
    args = parser.parse_args()

    # Load configuration based on dataset
    if args.dataset == "mnist":
        config_path = "./configs/mnist.json"
    elif args.dataset == "cifar":
        config_path = "./configs/cifar.json"
    elif args.dataset == "fashion_mnist":
        config_path = "./configs/fashion_mnist.json"
    elif args.dataset == "huggan":
        config_path = "./configs/huggan_AFHQv2.json"
    else:
        raise ValueError("Unknown dataset")

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configuration parameters
    batch_size = config["batch_size"]
    timesteps = config["timesteps"]
    cfg = config["cfg"]
    channels = config["model"]["in_channels"]
    image_size = config["model"]["input_size"]
    dim = config["model"]["dim"]

    # Dataset and preprocessing
    if config["dataset"] in ["cifar"]:
        dataset_name = "cifar"
        fdataset = datasets.CIFAR10
        transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x * 2) - 1)
        ])
        dataset = fdataset(root="./data", train=True, download=True, transform=transform)
    elif config["dataset"] in ["fashion_mnist"]:
        dataset_name = "fashion_mnist"
        fdataset = datasets.FashionMNIST
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x * 2) - 1)
        ])
        dataset = fdataset(root="./data", train=True, download=True, transform=transform)
    elif config["dataset"] in ["huggan", "huggan/AFHQv2"]:
        from datasets import load_dataset

        fdataset = load_dataset("huggan/afhqv2")
        dataset = fdataset["train"]
        dataset = dataset.filter(lambda x: x["label"] == 0)
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        def apply_transform(example):
            example["image"] = transform(example["image"])
            return example
        dataset = dataset.map(apply_transform)
        dataset.set_format("torch", columns=["image", "label"])
    else:  # Default to MNIST
        dataset_name = "mnist"
        fdataset = datasets.MNIST
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x * 2) - 1)
        ])
        dataset = fdataset(root="./data", train=True, download=True, transform=transform)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Initialize U-Net model
    model = Unet(dim=image_size, channels=3).to(device)

    # Training parameters
    training_config = config["training"]
    epochs = training_config["epochs"]
    lr = training_config["learning_rate"]

    # Output directory
    output_dir = "outputs"
    img_dir = Path(output_dir, f"{config['dataset']}")
    img_dir.mkdir(exist_ok=True, parents=True)

    # Initialize InstaFlow and optimizer
    inflow =InstaFlow(model, timesteps=timesteps)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

    # Training loop
    for epoch in range(1, epochs + 1):
        losses = []
        bar = tqdm(dataloader, desc=f"Epoch {epoch}", total=len(dataloader))
        model.train()

        for batch in bar:
            if config["dataset"] in ["huggan/AFHQv2", "huggan"]:
                x = batch['image'].type(torch.float32).to(device)
                if isinstance(batch['label'], torch.Tensor):
                    c = batch['label'].to(device)
                else:
                    c = torch.tensor(batch['label']).to(device)
            else:
                x, c = batch
                x = x.to(device)
                if isinstance(c, torch.Tensor):
                    c = c.to(device)

            optimizer.zero_grad()
            loss, blsct = inflow.forward(x, c)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(inflow.parameters(), max_norm=1.0)  # 勾配制限
            optimizer.step()
            scheduler.step()  # スケジューラ更新
            losses.append(loss.item())
            bar.set_postfix({"Average Loss": f"{torch.mean(torch.tensor(losses)):.4f}"})

        # Sampling
        inflow.model.eval()
        with torch.no_grad():
            cond = torch.arange(0, 16).to(device) % config["model"]["num_classes"]
            uncond = torch.ones_like(cond) * config["model"]["num_classes"]

            init_noise = torch.randn(16, channels, image_size, image_size).to(device)
            images = inflow.sample(init_noise, cond, uncond, cfg)

            final_image = images[-1]
            final_image = final_image * 0.5 + 0.5  # Unnormalize from [-1, 1] to [0, 1]
            final_image = final_image.clamp(0, 1)
            grid = make_grid(final_image.float(), nrow=4)
            save_image(grid, f"{img_dir}/sample_{epoch}.png")

        # Save model
        torch.save(model.state_dict(), img_dir / "unet_insta_flow.pth")

    print("Training complete.")

if __name__ == "__main__":
    main()
