from dit import DiT_Llama
import argparse
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from torchvision import transforms as T, datasets

import numpy as np
from pathlib import Path
from tqdm import tqdm

########################################
# DDPM クラス（元の処理内容と同一）
########################################
class DDPM:
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        """
        model: 拡散モデル（例：DiT_Llama のインスタンス）
        timesteps: 拡散のタイムステップ数（デフォルトは1000）
        beta_start, beta_end: 線形に増加する beta スケジュールの開始・終了値
        """
        self.model = model
        self.timesteps = timesteps

        # --- オリジナルと同じ beta スケジュールの計算 ---
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # q(x_t|x_0) 用
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # q(x_{t-1}|x_t, x_0) 用
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def extract(self, a, t, x_shape):
        """
        a は1次元テンソル（長さ T）のスケジュール、
        t は各サンプルのタイムステップ (B,)
        → a から t に対応する値を取り出し、x_shape にブロードキャスト可能な形 (B, 1, 1, …) に変形する。
        """
        out = a.gather(-1, t.cpu())
        return out.reshape(len(t), *([1] * (len(x_shape) - 1))).to(t.device)

    def q_sample(self, x_start, t, noise=None):
        """
        前向き拡散の閉形式:
          x_t = sqrt_alphas_cumprod_t * x_0 + sqrt(1-alphas_cumprod_t) * noise
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    @torch.no_grad()
    def p_sample(self, x, t, c, t_index, nc, cfg=2.0):
        """
        逆拡散の1ステップ:
          x_{t-1} = 1/sqrt(alpha_t) * [ x_t - (1-alpha_t)*model(x_t, t, c) / sqrt(1-alphas_cumprod_t) ] + ...
        ※ 元のコードでは、(1-alpha_t) の代わりに betas_t を用いています
        """
        betas_t = self.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)

        vc = self.model(x, t.float(), c)

        # Classifier Free Guidance
        if nc is not None:
            vu = self.model(x, t.float(), nc)
            vc = vu + cfg * (vc - vu)
        model_out = vc

        model_mean = sqrt_recip_alphas_t * (x - betas_t * model_out / sqrt_one_minus_alphas_cumprod_t)

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, c, nc, cfg):
        """
        純粋ノイズ x_T から始めて、t = T-1 ... 0 まで逆拡散
         shape: (B, C, H, W)
         c: (B,) のラベル（各設定に応じたラベル）
        """
        device = next(self.model.parameters()).device
        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t, c, i, nc, cfg)

        return img.clamp(-1, 1)

    @torch.no_grad()
    def sample(self, image_size, c, nc, batch_size=16, channels=1, cfg=2.0):
        """
        指定したラベル c (shape = [B]) で画像をサンプルする
         - image_size: 画像の高さ・幅（正方形）
         - c: (B,) 各サンプルのクラスラベル
        """
        shape = (batch_size, channels, image_size, image_size)
        return self.p_sample_loop(shape, c, nc, cfg)

    def p_losses(self, x_start, t, c, noise=None, loss_type="l2"):
        """
        学習用ロス計算:
          1) q_sample で x_t を生成
          2) モデルが予測するノイズ model(x_t, t, c) を取得
          3) 予測ノイズと実際の noise との差（MSE または L1）を計算
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = self.model(x_noisy, t.float(), c)

        if loss_type == 'l2':
            loss = F.mse_loss(predicted_noise, noise)
        elif loss_type == 'l1':
            loss = F.l1_loss(predicted_noise, noise)
        else:
            raise NotImplementedError(f"Unknown loss_type: {loss_type}")

        return loss

########################################
# メイン処理：configファイルからパラメータを読み込み、各データセットを切り替え
########################################
def main():
    import sys
    #sys.argv = ['script.py', '--dataset', 'huggan']

    parser = argparse.ArgumentParser(description="Choose dataset among: mnist, cifar, fashion_mnist, huggan")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar", "fashion_mnist", "huggan"],
                        help="Dataset to use (default: mnist)")
    args = parser.parse_args()

    # configファイルのパスを、引数に応じて決定
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # configに基づくデータセットと変換の設定
    image_size = config["image_size"]
    batch_size = config["batch_size"]
    timesteps=config["timesteps"]
    cfg=config["cfg"]

    # モデルの in_channels は config["model"]["in_channels"] とする
    channels = config["model"]["in_channels"]

    if config["dataset"] in ["cifar"]:
        transform = T.Compose([
            T.RandomCrop(image_size, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Lambda(lambda x: (x * 2) - 1)
        ])
        fdataset = datasets.CIFAR10
        dataset = fdataset(root="./data", train=True, download=True, transform=transform)
    elif config["dataset"] in ["fashion_mnist"]:
        transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Lambda(lambda x: (x * 2) - 1)
        ])
        fdataset = datasets.FashionMNIST
        dataset = fdataset(root="./data", train=True, download=True, transform=transform)
    elif config["dataset"] in ["huggan/AFHQv2", "huggan"]:
        from datasets import load_dataset
        from torchvision import transforms
        # データセットのロード
        fdataset = load_dataset("huggan/afhqv2")
        dataset = fdataset["train"]
        # ラベルが 0 のサンプルのみ抽出する処理（例：0が「猫」を表す場合）
        dataset = dataset.filter(lambda x: x["label"] == 0)
        # torchvision の transform を定義
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # 各サンプルに対して transform を適用する関数
        def apply_transform(example):
            example["image"] = transform(example["image"])
            return example
        # map() を使って transform を適用
        dataset = dataset.map(apply_transform)
        # PyTorch 用にフォーマットを設定
        dataset.set_format("torch", columns=["image", "label"])
    else:
        # デフォルトは MNIST
        transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Lambda(lambda x: (x * 2) - 1)
        ])
        fdataset = datasets.MNIST
        dataset = fdataset(root="./data", train=True, download=True, transform=transform)

    #dataset = fdataset(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # configに基づくモデルのパラメータ設定
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

    # configに基づく学習パラメータ
    training_config = config["training"]
    epochs = training_config["epochs"]
    lr = training_config["learning_rate"]

    # 出力先ディレクトリの作成（config の output_dir を使用）
    output_dir = "outputs"
    img_dir = Path(output_dir, f"{config['dataset']}")
    img_dir.mkdir(exist_ok=True, parents=True)

    # DDPMクラスの初期化およびオプティマイザ設定
    ddpm = DDPM(model, timesteps=timesteps, beta_start=1e-4, beta_end=0.02)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        losses = []
        bar = tqdm(dataloader, desc=f"Epoch {epoch}", total=len(dataloader))
        model.train()

        for batch in bar: # この行を変更

            if config["dataset"] in ["huggan/AFHQv2", "huggan"]:
                x = batch['image'].type(torch.float32).to(device)  # (B, C, H, W)
                # ラベル c はデータセットによっては異なる形式の場合があるので、Tensor であればデバイスに移動
                if isinstance(batch['label'], torch.Tensor):
                    c = batch['label'].to(device)
                else:
                    c = torch.tensor(batch['label']).to(device) # ラベルがテンソルでない場合はテンソルに変換
                    # また、ここでの bar を batch に変更して、一貫性を保ちました
            else:
                x, c = batch # この行を変更してタプルをアンパックするようにしました
                x = x.to(device)  # (B, C, H, W)
                # ラベル c はデータセットによっては異なる形式の場合があるので、Tensor であればデバイスに移動
                if isinstance(c, torch.Tensor):
                    c = c.to(device)

            B = x.size(0)
            t = torch.randint(0, ddpm.timesteps, (B,), device=device).long()
            loss = ddpm.p_losses(x, t, c, noise=None, loss_type="l2")
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
            bar.set_postfix({"loss": f"{torch.mean(torch.tensor(losses)):.4f}"})

        model.eval()
        with torch.no_grad():
            cond = torch.arange(0, 16).cuda() % 10
            uncond = torch.ones_like(cond) * 10
            samples = ddpm.sample(image_size, cond, uncond, batch_size=16, channels=channels, cfg=cfg)
            samples = (samples + 1) / 2  # [-1, 1] -> [0, 1]
            grid = make_grid(samples, nrow=4)
            save_image(grid, img_dir / f"sample_{epoch}.png")

    print("Training complete.")

if __name__ == "__main__":
    main()

