import argparse

import torch
import torch.nn.functional as F
from tqdm import tqdm


class EDM:
    def __init__(self, model, timesteps, sigma_min=0.002, sigma_max=80.0):
        self.model = model
        self.timesteps = timesteps
        self.sigma_min = sigma_min  # 最小ノイズスケール
        self.sigma_max = sigma_max  # 最大ノイズスケール

    def get_sigma(self, t):
        """時間tに基づくノイズスケジュールσ(t)を計算"""
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def forward(self, x, cond):
        b = x.size(0)
        # 時間tをランダムにサンプリング (0, 1] の範囲
        t = torch.rand((b,), device=x.device)
        sigma_t = self.get_sigma(t)
        sigma_t_exp = sigma_t.view([b, *([1] * len(x.shape[1:]))])

        # ノイズ付き画像 z_t = x + σ(t) * ε
        epsilon = torch.randn_like(x)
        z_t = x + sigma_t_exp * epsilon

        # モデルでノイズ予測 ε_θ を取得
        epsilon_theta = self.model(z_t, t, cond)

        # 損失関数：予測ノイズと実際のノイズのMSE
        batchwise_mse = ((epsilon - epsilon_theta) ** 2).mean(dim=list(range(1, len(x.shape))))
        tlist = batchwise_mse.detach().cpu().reshape(-1).tolist()
        ttloss = [(tv.item(), tloss) for tv, tloss in zip(t, tlist)]
        return batchwise_mse.mean(), ttloss

    @torch.no_grad()
    def sample(self, z, cond, null_cond, cfg=2.0):
        b = z.size(0)
        images = []
        z_t = z  # 初期ノイズ

        # 時間ステップを線形に分割
        t_steps = torch.linspace(1.0, 0.0, self.timesteps + 1, device=z.device)
        for i in tqdm(range(self.timesteps), desc='sampling loop time step', total=self.timesteps):
            t = t_steps[i]
            t_next = t_steps[i + 1]
            sigma_t = self.get_sigma(t)
            sigma_t_next = self.get_sigma(t_next)

            # モデルでノイズ予測
            epsilon_theta = self.model(z_t, t, cond)

            # Classifier-Free Guidance
            if null_cond is not None:
                epsilon_uncond = self.model(z_t, t, null_cond)
                epsilon_theta = epsilon_uncond + cfg * (epsilon_theta - epsilon_uncond)

            # EDMの更新ステップ（簡略化されたEuler法）
            dt = t_next - t
            denoised = z_t - sigma_t * epsilon_theta  # D_θ(z_t, t)
            z_t = z_t + (denoised - z_t) * (sigma_t_next - sigma_t) / sigma_t

            images.append(z_t)

        return images


# -------------------------------
# Main function: 設定ファイルの読み込みと学習処理
# -------------------------------
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from dit import DiT_Llama  # 仮定：既存のモデルクラス


def main():
    parser = argparse.ArgumentParser(description="Choose dataset among: mnist, cifar, fashion_mnist, huggan")
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "cifar", "fashion_mnist", "huggan"],
                        help="Dataset to use (default: mnist)")
    args = parser.parse_args()

    # 設定ファイルの読み込み
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

    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # configからパラメータ取得
    batch_size = config["batch_size"]
    timesteps = config["timesteps"]
    cfg = config["cfg"]
    channels = config["model"]["in_channels"]
    image_size = config["model"]["input_size"]

    # データセットと前処理の設定
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
        fdataset = load_dataset("huggan/afhqv2")["train"]
        dataset = fdataset.filter(lambda x: x["label"] == 0)
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x * 2) - 1)
        ])
        def apply_transform(example):
            example["image"] = transform(example["image"])
            return example
        dataset = dataset.map(apply_transform)
        dataset.set_format("torch", columns=["image", "label"])
    else:
        dataset_name = "mnist"
        fdataset = datasets.MNIST
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x * 2) - 1)
        ])
        dataset = fdataset(root="./data", train=True, download=True, transform=transform)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # モデル初期化
    model_config = config["model"]
    model = DiT_Llama(
        in_channels=model_config["in_channels"],
        input_size=model_config["input_size"],
        patch_size=model_config["patch_size"],
        dim=model_config["dim"],
        n_layers=model_config["n_layers"],
        n_heads=model_config["n_heads"],
        multiple_of=model_config["multiple_of"],
        ffn_dim_multiplier=model_config["ffn_dim_multiplier"],
        norm_eps=model_config["norm_eps"],
        class_dropout_prob=model_config["class_dropout_prob"],
        num_classes=model_config["num_classes"] + 1
    ).to(device)

    # 学習パラメータ
    training_config = config["training"]
    epochs = training_config["epochs"]
    lr = training_config["learning_rate"]

    # 出力ディレクトリ
    output_dir = "outputs"
    img_dir = Path(output_dir, f"{config['dataset']}")
    img_dir.mkdir(exist_ok=True, parents=True)

    # EDMクラスの初期化
    edm = EDM(model, timesteps=timesteps)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 学習ループ
    for epoch in range(1, epochs + 1):
        losses = []
        bar = tqdm(dataloader, desc=f"Epoch {epoch}", total=len(dataloader))
        model.train()

        for batch in bar:
            if config["dataset"] in ["huggan/AFHQv2", "huggan"]:
                x = batch['image'].type(torch.float32).to(device)
                c = torch.tensor(batch['label']).to(device) if not isinstance(batch['label'], torch.Tensor) else batch['label'].to(device)
            else:
                x, c = batch
                x = x.to(device)
                c = c.to(device)

            optimizer.zero_grad()
            loss, blsct = edm.forward(x, c)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            bar.set_postfix({"Average Loss": f"{torch.mean(torch.tensor(losses)):.4f}"})

        # サンプル生成
        model.eval()
        with torch.no_grad():
            cond = torch.arange(0, 16).cuda() % model_config["num_classes"]
            uncond = torch.ones_like(cond) * model_config["num_classes"]
            init_noise = torch.randn(16, channels, image_size, image_size).cuda()
            images = edm.sample(init_noise, cond, uncond, cfg)

            final_image = images[-1]
            final_image = final_image * 0.5 + 0.5  # [-1, 1] -> [0, 1]
            final_image = final_image.clamp(0, 1)
            grid = make_grid(final_image.float(), nrow=4)
            save_image(grid, f"{img_dir}/sample_{epoch}_last.png")

        torch.save(model.state_dict(), img_dir / "edm.pth")

    print("Training complete.")


if __name__ == "__main__":
    main()