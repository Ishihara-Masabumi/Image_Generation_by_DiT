# implementation of Rectified Flow for simple minded people like me.
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class ConsistencyModel:
    def __init__(self, model_e, model_t, timesteps, ln=True):
        """
        Consistency Training (CT) の初期化

        Args:
            model_e: エンコーダーネットワーク（学習対象：パラメータ更新される）
            model_t: ターゲットネットワーク（EMAで更新するため直接勾配更新は行わない）
            timesteps: サンプリングに用いるステップ数
            ln: True の場合、標準正規分布サンプルにシグモイドを適用して [0,1] の範囲に収める
        """
        self.model_e = model_e
        self.model_t = model_t
        self.timesteps = timesteps
        self.ln = ln

    def forward(self, x, cond):
        """
        学習時のフォワードパス（Consistency Training, Algorithm 3 CT）

        1. バッチサイズ b に対して、時刻 t をサンプリング（ln=Trueならシグモイド適用）
        2. t の形状を入力 x と合わせるために展開し、ノイズ z1 を生成
        3. 線形補間により x_t = (1-t)x + t * z1 を計算
        4. エンコーダーネットワーク model_e に x_t, t, cond を与え、出力 y_pred を得る
        5. ターゲットネットワーク model_t に、t=0（=純粋な x）を与え y_target を得る
        6. Consistency Loss: L = MSE(y_pred, y_target) を計算し、返す

        Returns:
            loss: バッチ全体の平均損失
            info: ログ用情報（ここでは None として返す）
        """
        b = x.size(0)
        # ① 時刻 t のサンプリング
        if self.ln:
            nt = torch.randn((b,)).to(x.device)
            t = torch.sigmoid(nt)
        else:
            t = torch.rand((b,)).to(x.device)
        # 入力画像 x の次元に合わせて t を拡張（例：[b, 1, 1, 1]）
        t_expanded = t.view([b, *([1] * (len(x.shape) - 1))])
        
        # ② ノイズ z1 の生成と ③ 線形補間による x_t の計算
        z1 = torch.randn_like(x)
        x_t = (1 - t_expanded) * x + t_expanded * z1

        # ④ エンコーダーネットワークの出力（時刻 t の入力）
        y_pred = self.model_e(x_t, t, cond)
        
        # ⑤ ターゲットネットワークの出力（時刻 0 を入力）
        t0 = torch.zeros_like(t)
        y_target = self.model_t(x, t0, cond)

        # ⑥ Consistency Loss の計算（各サンプルごとの MSE を画像全体の次元で平均）
        ldata = ((z1 - x - y_pred) ** 2).mean(dim=list(range(1, len(x.shape))))
        lconsis = ((y_pred - y_target) ** 2).mean(dim=list(range(1, len(x.shape))))
        loss = ldata + lconsis
        return loss.mean(), None

    @torch.no_grad()
    def sample(self, z, cond, null_cond=None, cfg=2.0):
        """
        サンプリング（推論）時の処理

        1. 初期ノイズ z から開始
        2. 指定した timesteps 回のループで、時刻 t を徐々に下げながら model_e を適用し出力を更新
        3. オプションで Classifier-Free Guidance を適用（null_cond が与えられている場合）

        Args:
            z: 初期ノイズテンソル
            cond: 条件入力
            null_cond: Classifier-Free Guidance 用の条件（通常は None）
            cfg: Guidance の強さ

        Returns:
            images: サンプリング過程で得られた画像のリスト
        """
        b = z.size(0)
        dt = 1.0 / self.timesteps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * (len(z.shape) - 1))])
        images = [z]
        # timesteps を逆順（t=1 から t=0）にループ
        for i in tqdm(reversed(range(self.timesteps)), desc='sampling loop', total=self.timesteps):
            t_val = i / self.timesteps
            t_tensor = torch.tensor([t_val] * b).to(z.device)
            # model_e により出力を得る
            y = self.model_e(z, t_tensor, cond)
            # Classifier-Free Guidance（null_cond が指定されている場合）
            if null_cond is not None:
                y_null = self.model_e(z, t_tensor, null_cond)
                y = y_null + cfg * (y - y_null)
            # 更新（Consistency Model では通常、出力が直接 x0 の予測となる）
            z = y
            images.append(z)
        return images

# ---------------------------------------------------------------------
# 例：学習ループ内での使用例（EMA更新などは学習ループ側で実施）
# ---------------------------------------------------------------------

def update_target_network(model_e, model_t, decay):
    """
    model_t のパラメータを、model_e のパラメータのEMAで更新する

    Args:
        model_e: エンコーダーネットワーク（最新パラメータ）
        model_t: ターゲットネットワーク（EMA更新対象）
        decay: EMAの減衰係数（例: 0.999）
    """
    with torch.no_grad():
        for param_t, param_e in zip(model_t.parameters(), model_e.parameters()):
            param_t.data.mul_(decay).add_(param_e.data, alpha=1 - decay)

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
    #import sys

    #sys.argv = ['script.py', '--dataset', 'cifar']
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

    model_e = DiT_Llama(
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

    model_t = DiT_Llama(
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
    ema_decay=0.999

    # 出力先ディレクトリの作成（config の output_dir などを利用）
    output_dir = "outputs"
    img_dir = Path(output_dir, f"{config['dataset']}")
    img_dir.mkdir(exist_ok=True, parents=True)

    # ここまでが設定ファイルの読み込み、データセットの前処理、データローダーからの読み出し、
    # モデルへの入力まで、DDPMコードと同一の処理です。

    ############################################
    # ConsistencyModel (CM) の学習処理部分に修正
    ############################################

    # ConsistencyModel クラスの初期化（cmは model をラップするクラスとする）
    cm = ConsistencyModel(model_e, model_t, timesteps=timesteps)  # cm クラスの実装に依存します
    optimizer = optim.Adam(model_e.parameters(), lr=lr)
    #criterion = torch.nn.MSELoss()

    # 学習ループ
    for epoch in range(1, epochs + 1):
        losses = []
        bar = tqdm(dataloader, desc=f"Epoch {epoch}", total=len(dataloader))
        cm.model_e.train()

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
            loss, blsct = cm.forward(x, c)
            loss.backward()
            optimizer.step()
            # EMAによるターゲットネットワークの更新
            update_target_network(model_e, model_t, ema_decay)
            losses.append(loss.item())
            bar.set_postfix({"Average Loss": f"{torch.mean(torch.tensor(losses)):.4f}"})

        # サンプル生成
        cm.model_e.eval()
        with torch.no_grad():
            cond = torch.arange(0, 16).cuda() % model_config["num_classes"]
            uncond = torch.ones_like(cond) * model_config["num_classes"]

            init_noise = torch.randn(16, channels, 32, 32).cuda()
            images = cm.sample(init_noise, cond, uncond, cfg)

            # 生成された画像列のうち、最終ステップの画像を使用
            final_image = images[-1]
            # 画像の値を [-1, 1] から [0, 1] に戻す（unnormalize）
            final_image = final_image * 0.5 + 0.5
            final_image = final_image.clamp(0, 1)
            # 複数画像の場合は、グリッド状にまとめる（ここでは 4 枚ずつのグリッド）
            grid = make_grid(final_image.float(), nrow=4)
            # 画像を保存
            save_image(grid, f"{img_dir}/sample_{epoch}_last.png")

        cm.model_e.train()

    print("Training complete.")

if __name__ == "__main__":
    main()
