# implementation of Rectified Flow for simple minded people like me.
import argparse

import torch
import torch.nn.functional as F
from tqdm import tqdm


class ReFlow:
    def __init__(self, model, timesteps, ln=True):
        self.model = model
        self.ln = ln
        self.timesteps = timesteps

    def adaptive_scaling(self, t):
        """
        時間 t に依存する再正規化係数 α(t) を定義
        例: sin 関数を用いたスケーリング
        """
        return 1.0 + 0.1 * torch.sin(2 * torch.pi * t)

    def forward(self, x, cond):
        b = x.size(0)
        # 時間 t のサンプリング
        if self.ln:
            nt = torch.randn((b,)).to(x.device)
            t = torch.sigmoid(nt)
        else:
            t = torch.rand((b,)).to(x.device)

        texp = t.view([b, *([1] * len(x.shape[1:]))])
        z1 = torch.randn_like(x)
        zt = (1 - texp) * x + texp * z1

        # モデルから予測された速度場
        vtheta = self.model(zt, t, cond)

        # 再正規化係数 α(t) を適用
        alpha_t = self.adaptive_scaling(t).view(b, 1, 1, 1)
        v_true = (z1 - x) * alpha_t  # 真の速度に補正を適用

        # 損失関数 (MSE)
        batchwise_mse = ((v_true - vtheta) ** 2).mean(dim=list(range(1, len(x.shape))))
        tlist = batchwise_mse.detach().cpu().reshape(-1).tolist()
        ttloss = [(tv, tloss) for tv, tloss in zip(t, tlist)]

        return batchwise_mse.mean(), ttloss

    @torch.no_grad()
    def sample(self, z, cond, null_cond=None, cfg=2.0):
        b = z.size(0)
        dt = 1.0 / self.timesteps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        images = [z]

        for i in tqdm(reversed(range(self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            t = i / self.timesteps
            t = torch.tensor([t] * b).to(z.device)

            # モデルから速度場を取得
            vc = self.model(z, t, cond)

            # 再正規化係数 α(t) を適用
            alpha_t = self.adaptive_scaling(t).view(b, 1, 1, 1)
            vc = vc * alpha_t  # 速度場に動的な補正を適用

            # Classifier-Free Guidance の適用
            if null_cond is not None:
                vu = self.model(z, t, null_cond)
                vu = vu * alpha_t  # null_cond にも補正を適用
                vc = vu + cfg * (vc - vu)

            # Euler 法による更新
            z = z - dt * vc
            images.append(z)

        return images


# -------------------------------
# Main function: 設定ファイルの読み込みと学習処理
# -------------------------------
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from dit import DiT_Llama


def main():
    import sys
    sys.argv = ['script.py', '--dataset', 'cifar']

    # コマンドライン引数でデータセットを選択
    parser = argparse.ArgumentParser(description="Choose dataset among: mnist, cifar, fashion_mnist, huggan")
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "cifar", "fashion_mnist", "huggan"],
                        help="Dataset to use (default: mnist)")
    args = parser.parse_args()

    # 選択された dataset に対応する config ファイルのパスを設定
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

    # 設定ファイルの読み込み
    with open(config_path, 'r') as f:
        config = json.load(f)

    # device の設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # config から画像サイズ、バッチサイズなどを取得
    batch_size = config["batch_size"]
    timesteps=config["timesteps"]
    cfg=config["cfg"]

    # モデルの in_channels（チャネル数）は config["model"]["in_channels"] を使用
    channels = config["model"]["in_channels"]
    # モデルの image_size は config["model"]["input_size"] とする
    image_size = config["model"]["input_size"]
    
    # データセットと前処理の設定（config["dataset"] も参考に）
    if config["dataset"] in ["cifar"]:
        dataset_name = "cifar"
        fdataset = datasets.CIFAR10
        transform = T.Compose([
            T.RandomCrop(image_size, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Lambda(lambda x: (x * 2) - 1)
        ])
        dataset = fdataset(root="./data", train=True, download=True, transform=transform)
    elif config["dataset"] in ["fashion_mnist"]:
        dataset_name = "fashion_mnist"
        fdataset = datasets.FashionMNIST
        transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Lambda(lambda x: (x * 2) - 1)
        ])
        dataset = fdataset(root="./data", train=True, download=True, transform=transform)
    elif config["dataset"] in ["huggan", "huggan/AFHQv2"]:
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
    else:  # デフォルトは mnist
        dataset_name = "mnist"
        fdataset = datasets.MNIST
        transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Lambda(lambda x: (x * 2) - 1)
        ])
        dataset = fdataset(root="./data", train=True, download=True, transform=transform)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # モデルの構成：config["model"] 内のパラメータを使用して DiT_Llama を初期化
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

    # 学習パラメータ：config["training"]
    training_config = config["training"]
    epochs = training_config["epochs"]
    lr = training_config["learning_rate"]

    # 出力先ディレクトリの作成（config の output_dir などを利用）
    output_dir = "outputs"
    img_dir = Path(output_dir, f"{config['dataset']}")
    img_dir.mkdir(exist_ok=True, parents=True)

    # ここまでが設定ファイルの読み込み、データセットの前処理、データローダーからの読み出し、
    # モデルへの入力まで、DDPMコードと同一の処理です。

    ############################################
    # 以下、Rectified Flow (ReFlow) の学習処理部分に修正
    ############################################

    # ReFlow クラスの初期化（ReFlowは model をラップするクラスとする）
    reflow = ReFlow(model, timesteps=timesteps)  # ReFlow クラスの実装に依存します
    optimizer = optim.Adam(model.parameters(), lr=lr)
    #criterion = torch.nn.MSELoss()

    # 学習ループ
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

            optimizer.zero_grad()
            loss, blsct = reflow.forward(x, c)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            bar.set_postfix({"Average Loss": f"{torch.mean(torch.tensor(losses)):.4f}"})

        # サンプル生成
        reflow.model.eval()
        with torch.no_grad():
            cond = torch.arange(0, 16).cuda() % model_config["num_classes"]
            uncond = torch.ones_like(cond) * model_config["num_classes"]

            init_noise = torch.randn(16, channels, 32, 32).cuda()
            images = reflow.sample(init_noise, cond, uncond, cfg)

            # 生成された画像列のうち、最終ステップの画像を使用
            final_image = images[-1]
            # 画像の値を [-1, 1] から [0, 1] に戻す（unnormalize）
            final_image = final_image * 0.5 + 0.5
            final_image = final_image.clamp(0, 1)
            # 複数画像の場合は、グリッド状にまとめる（ここでは 4 枚ずつのグリッド）
            grid = make_grid(final_image.float(), nrow=4)
            # 画像を保存
            save_image(grid, f"{img_dir}/sample_{epoch}_last.png")

        reflow.model.train()

    print("Training complete.")

if __name__ == "__main__":
    main()
