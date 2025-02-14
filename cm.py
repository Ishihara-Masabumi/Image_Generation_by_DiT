# implementation of Rectified Flow for simple minded people like me.
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class ConsistencyModel:
    def __init__(self, model_e, model_t, timesteps, ln=True):
        """
        Consistency Model の初期化

        Args:
            model_e: エンコーダーネットワーク（学習対象）
            model_t: ターゲットネットワーク（アンカー、t=0 のときの出力）
            timesteps: サンプリングに用いるステップ数
            ln: 時刻サンプリングに対して標準正規分布ではなく、シグモイドを適用するか否か
        """
        self.model_e = model_e
        self.model_t = model_t
        self.timesteps = timesteps
        self.ln = ln

    def forward(self, x, cond):
        """
        学習時のフォワードパス
        ・ランダムな時刻 t をサンプル
        ・z1 はノイズ、zt = (1-t) * x + t * z1 として中間状態を生成
        ・model_e(zt, t, cond) から推定 x0 (xt_pred) を得る
        ・model_t(x, 0, cond) をターゲットとして xs_pred を得る
        ・L_consistency = MSE(xt_pred, xs_pred) と L_data = MSE(xt_pred, x) を計算し、両者の和を損失とする
        """
        b = x.size(0)
        # 時刻 t のサンプル（ln=True の場合、正規分布にシグモイドを適用）
        if self.ln:
            nt = torch.randn((b,)).to(x.device)
            t = torch.sigmoid(nt)
        else:
            t = torch.rand((b,)).to(x.device)
        # バッチサイズに合わせた形状に拡張
        texp = t.view([b, *([1] * (len(x.shape) - 1))])
        # 入力 x に対してノイズ z1 を生成し、中間状態 z_t を計算
        z1 = torch.randn_like(x)
        zt = (1 - texp) * x + texp * z1

        # エンコーダーネットワークによる推定 (f_theta(x_t, t))
        xt_pred = self.model_e(zt, t, cond)
        # ターゲットネットワーク: t=0 を与えて、アンカーとしての x0 を取得
        t0 = torch.zeros_like(t)
        xs_pred = self.model_t(x, t0, cond)

        # 一貫性損失: model_e の出力同士が近くなるように
        L_consistency = ((xt_pred - xs_pred) ** 2).mean(dim=list(range(1, len(x.shape))))
        # データ損失: 推定結果が元の x に近づくように
        L_data = ((z1 - x - xt_pred) ** 2).mean(dim=list(range(1, len(x.shape))))
        loss = L_consistency + L_data

        # （ログ用）各サンプルごとの t と一貫性損失をリスト化
        t_list = t.detach().cpu().reshape(-1).tolist()
        loss_list = L_consistency.detach().cpu().reshape(-1).tolist()
        ttloss = [(tv, tloss) for tv, tloss in zip(t_list, loss_list)]

        return loss.mean(), ttloss

    @torch.no_grad()
    def sample(self, z, cond, null_cond=None, cfg=2.0):
        """
        サンプリング（推論）時の処理
        ・初期ノイズ z から開始
        ・timesteps 回の反復処理で、各ステップで時刻 t を与えて model_e により直接 x0 の推定を行う
        ・(オプション) Classifier-Free Guidance を用いて、null_cond を使った補正を行う

        Returns:
            生成途中の画像リスト images
        """
        b = z.size(0)
        dt = 1.0 / self.timesteps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * (len(z.shape) - 1))])
        images = [z]
        # 逆順（t=1 から t=0）に timesteps 分ループ
        for i in tqdm(reversed(range(self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            t_val = i / self.timesteps
            t_tensor = torch.tensor([t_val] * b).to(z.device)
            # 現在の z に対して model_e を適用し、x0 の推定を得る
            x_pred = self.model_e(z, t_tensor, cond)
            # Classifier-Free Guidance の場合
            if null_cond is not None:
                x_pred_null = self.model_e(z, t_tensor, null_cond)
                x_pred = x_pred_null + cfg * (x_pred - x_pred_null)
            # Consistency Model では、単一または少数ステップで直接 x0 を得るため z を更新
            z = x_pred
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
