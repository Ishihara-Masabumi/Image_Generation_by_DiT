import math
import os
import sys
import urllib.request
from functools import partial
from inspect import isfunction

sys.path.insert(0, "./edm")  # EDM のパスを先頭に追加
sys.path.insert(0, "./edm/torch_utils")  # torch_utils のパスを先頭に追加
import dnnlib
import legacy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_utils.misc
import torchvision
from diffusers import StableDiffusionPipeline
from einops import einsum, rearrange, reduce
from einops.layers.torch import Rearrange
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm


# ユーティリティ関数
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
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()
        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

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
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) if exists(time_emb_dim) else None
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, emb=None):
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
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv)
        q = q * self.scale
        sim = einsum(q, k, "b h d i, b h d j -> b h i j")
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = einsum(attn, v, "b h i j, b h d j -> b h i d")
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv)
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        context = einsum(k, v, "b h d n, b h e n -> b h d e")
        out = einsum(context, q, "b h d e, b h d n -> b h e n")
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
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)
        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 1, padding=0)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)
        time_dim = dim * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.cond_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

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
                        Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1),
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
                        Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1),
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

        t = self.time_mlp(time)
        c = self.cond_mlp(cond)
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

class InstaFlow(nn.Module):
    def __init__(self, model, timesteps=5):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        if self.timesteps < 2:
            print("timesteps is wrong.")
            return

    def forward(self, x, cond):
        b = x.size(0)
        device = x.device
        t_grid = torch.linspace(0, 1, steps=self.timesteps, device=device)
        n = torch.randint(1, self.timesteps, (b,), device=device)
        t = t_grid[n]
        texp = t.view(b, 1, 1, 1)
        z1 = torch.randn_like(x)
        z1 = torch.clamp(z1, min=-3.0, max=3.0)
        zt = (1 - texp) * x + texp * z1
        vtheta = self.model(zt, t, cond)
        batchwise_mse = ((z1 - x - vtheta) ** 2).mean(dim=list(range(1, len(x.shape))))
        return batchwise_mse.mean(), None

    @torch.no_grad()
    def sample(self, z, cond, null_cond=None, cfg=2.0, steps=None):
        b = z.size(0)
        device = z.device
        steps = steps if steps is not None else self.timesteps
        dt = 1.0 / (steps - 1)
        images = [z]
        for i in reversed(range(1, steps)):
            t = torch.full((b,), i / (steps - 1), device=device)
            vc = self.model(z, t, cond)
            if null_cond is not None:
                vu = self.model(z, t, null_cond)
                vc = vu + cfg * (vc - vu)
            z = z - dt * vc
            images.append(z)
        return images[-1]

    @torch.no_grad()
    def generate_pairs(self, x, cond):
        b = x.size(0)
        device = x.device
        z1 = torch.randn_like(x)
        z1 = torch.clamp(z1, min=-3.0, max=3.0)
        x0 = self.sample(z1, cond)
        return x0, z1

    def reflow(self, x0, z1, cond):
        b = x0.size(0)
        device = x0.device
        t_grid = torch.linspace(0, 1, steps=self.timesteps, device=device)
        n = torch.randint(1, self.timesteps, (b,), device=device)
        t = t_grid[n]
        texp = t.view(b, 1, 1, 1)
        zt = (1 - texp) * x0 + texp * z1
        vtheta = self.model(zt, t, cond)
        batchwise_mse = ((z1 - (x0 + vtheta)) ** 2).mean(dim=list(range(1, len(x0.shape))))
        return batchwise_mse.mean(), None

    def distill(self, x, z, cond):
        b = x.size(0)
        device = x.device
        t = torch.rand(b, device=device) * 0.2 + 0.8  # 0.8〜1.0 の範囲
        vtheta = self.model(z, t, cond)  # モデルの予測
        batchwise_mse = ((x - z + vtheta) ** 2).mean(dim=list(range(1, len(x.shape))))  # 修正
        return batchwise_mse.mean(), None


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    epochs = 40
    num_triplets = 50000
    batch_size = 32

    # モデルの初期化
    unet1 = Unet(
        dim=32
    ).to(device)
    unet2 = Unet(
        dim=64,
        channels=3,
        dim_mults=(1, 2, 4, 8),
        resnet_block_groups=8
    ).to(device)
    unet3 = Unet(
        dim=64,
        channels=3,
        dim_mults=(1, 2, 4, 8),
        resnet_block_groups=8
    ).to(device)
    instaf1 = InstaFlow(unet1, timesteps=50).to(device)
    instaf2 = InstaFlow(unet2, timesteps=50).to(device)
    instaf3 = InstaFlow(unet3, timesteps=50).to(device)

    # samples/unet_insta_flow.pthをUnetにロード
    checkpoint_path = "samples/unet_insta_flow_883.pth"
    if os.path.exists(checkpoint_path):
        print(f"Loading pre-trained model from {checkpoint_path}...")
        instaf1.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print(f"Checkpoint file {checkpoint_path} not found. Using newly initialized model.")

    optimizer2 = optim.Adam(instaf2.parameters(), lr=5e-4)
    optimizer3 = optim.Adam(instaf3.parameters(), lr=5e-4)

    instaf1.model.to(device)
    instaf1.model.eval()

    # Step 1: 既存のモデルからトリプレットを生成
    print("Step 1: Generating (text, noise, image) triplets from Stable Diffusion...")
    # 教師データのトリプレット (labels, teacher_time, teacher_images) をバッチ単位で生成
    triplets = []
    for _ in tqdm(range(num_triplets // batch_size), desc="Generating teacher triplets"):
        # 0～9の整数ラベルをランダムに生成
        labels = torch.randint(0, 10, (batch_size,), device=device)
        z = torch.randn(batch_size, 3, 32, 32).to(device)
        # EDMモデルから教師画像を生成（ラベルを条件として渡す）
        teacher_images = instaf1.sample(z, labels)
        # トリプレットとして (labels, teacher_time, teacher_images) を保存
        triplets.append((labels, teacher_images))

    # Step 2: 2-Rectified Flowのトレーニング (修正箇所)
    print("Step 2: Training 2-Rectified Flow with text-conditioned reflow...")

    for epoch in range(epochs):
        total_loss = 0
        # triplets全体を回して学習
        for labels, images in tqdm(triplets, desc=f"Epoch {epoch+1} (2-Rectified)"):
            images, labels = images.to(device), labels.to(device)
            noises = torch.randn(batch_size, 3, 32, 32).to(device)

            # reflow学習
            loss, _ = instaf2.reflow(images, noises, labels)
            total_loss += loss.item()

            optimizer2.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(instaf2.parameters(), max_norm=1.0)
            optimizer2.step()

        avg_loss = total_loss / len(triplets)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.6f}")

        # サンプル生成
        z_sample = torch.randn(16, 3, 32, 32).to(device)
        cond_sample = torch.arange(0, 16).to(device) % 10
        generated = instaf2.sample(z_sample, cond_sample)
        save_image(generated, f"samples/stage2_epoch{epoch+1}_samples.png", nrow=4, normalize=True)

    torch.save(instaf2.state_dict(), "checkpoints/2_rectified.pth")

    # Step 3: One-Step InstaFlowへの蒸留
    print("Step 3: Distilling to One-Step InstaFlow...")
    for epoch in range(epochs):
        total_loss = 0
        for labels, images in tqdm(triplets, desc=f"Epoch {epoch+1} (Distillation)"):
            images, labels = images.to(device), labels.to(device)
            noises = torch.randn(batch_size, 3, 32, 32).to(device)

            # 毎バッチでペアを生成
            z0, z1 = instaf2.generate_pairs(images, labels)

            loss, _ = instaf3.distill(z0, noises, labels)
            total_loss += loss.item()

            optimizer3.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(instaf3.parameters(), max_norm=1.0)
            optimizer3.step()

        avg_loss = total_loss / len(triplets)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

        z_sample = torch.randn(16, 3, 32, 32).to(device)
        cond_sample = torch.arange(0, 16).to(device) % 10
        generated = instaf3.sample(z_sample, cond_sample, steps=2)
        save_image(generated, f"samples/stage3_epoch{epoch+1}_samples.png", nrow=4, normalize=True)

    torch.save(instaf3.state_dict(), "checkpoints/instaflow.pth")

    print("Generating final samples...")
    z = torch.randn(16, 3, 32, 32).to(device)
    cond_final = torch.arange(0, 16).to(device) % 10
    generated = instaf3.sample(z, cond_final, steps=2)
    save_image(generated, f"samples/final_samples.png", nrow=4, normalize=True)
    print(f"Generated shape: {generated.shape}")

if __name__ == "__main__":
    main()

