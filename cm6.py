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
            model_e: オンラインネットワーク（学習対象、f_θ）
            model_t: ターゲットネットワーク（EMAで更新される, f_θ'）
            timesteps: 離散化ステップ数 N（例: 20）
            ln: True の場合、時刻サンプリングにシグモイドを適用して [0,1] に収める
        """
        self.model_e = model_e
        self.model_t = model_t
        self.timesteps = timesteps
        self.ln = ln

    def forward(self, x, cond):
        """
        CT損失の計算（Algorithm 3に基づく）

        1. 0〜1の区間を timesteps 個に線形分割した時刻グリッドを作成。
        2. ランダムに隣接する2つの時刻 t_n と t_{n+1} を選ぶ。
        3. それぞれの時刻に対して、ノイズ z を用い
           x_t = (1-t)*x + t*z という形でノイズ付与画像を生成。
        4. オンラインネットワーク (model_e) は x_{t_{n+1}} と t_{n+1} から予測を行い、
           ターゲットネットワーク (model_t) は x_{t_n} と t_n から予測を行う。
        5. 2つの出力の平方誤差 (MSE) を損失とする。

        Returns:
            loss: CT損失（スカラー）
            info: ログ情報（ここでは None）
        """
        b = x.size(0)
        device = x.device

        # 0〜1の時刻グリッドを生成
        t_grid = torch.linspace(0, 1, steps=self.timesteps, device=device)
        # ランダムに隣接する2つの時刻のインデックスを選択（n in [0, timesteps-2]）
        n = torch.randint(0, self.timesteps - 1, (1,)).item()
        t_n = t_grid[n]       # 早い時刻
        t_np1 = t_grid[n+1]   # 次の時刻

        # バッチ用の時刻テンソルを作成
        t_n_tensor = t_n * torch.ones(b, device=device)
        t_np1_tensor = t_np1 * torch.ones(b, device=device)

        # xの次元に合わせて時刻テンソルを拡張（例: [B, 1, 1, 1]）
        t_n_expanded = t_n_tensor.view(b, *([1] * (x.dim()-1)))
        t_np1_expanded = t_np1_tensor.view(b, *([1] * (x.dim()-1)))

        # ノイズを生成して、各時刻の画像を作成
        noise_n = torch.randn_like(x)
        #noise_np1 = torch.randn_like(x)
        x_tn = (1 - t_n_expanded) * x + t_n_expanded * noise_n
        x_tnp1 = (1 - t_np1_expanded) * x + t_np1_expanded * noise_n

        # オンラインネットワーク: t_{n+1} における出力
        y_pred = self.model_e(x_tnp1, t_np1_tensor, cond)
        # ターゲットネットワーク: t_n における出力
        y_target = self.model_t(x_tn, t_n_tensor, cond)

        # MSEを計算（バッチ以外の次元で平均した後、バッチ平均）
        #dims = tuple(range(1, y_pred.dim()))
        #loss = ((y_pred - y_target) ** 2).mean(dim=dims).mean()+((x - y_pred) ** 2).mean(dim=list(range(1, len(x.shape)))).mean()
        # 2つの出力間のMSE損失と再構成誤差の計算
        dims = tuple(range(1, y_pred.dim()))
        loss_pred = ((y_pred - y_target) ** 2).mean(dim=dims).mean()
        loss_recon = ((x - y_pred) ** 2).mean(dim=list(range(1, x.dim()))).mean() * 0.5
        loss = loss_pred + loss_recon
        return loss, None

    @torch.no_grad()
    def sample(self, z, cond, null_cond=None, cfg=2.0):
        """
        推論時のサンプリング

        初期ノイズ z から開始し、timesteps 回のループでオンラインネットワーク (model_e)
        を適用して最終的な x0 の推定を行う。オプションで Classifier-Free Guidance を適用可能。

        Returns:
            images: 各ステップの出力を含むリスト
        """
        sample_timesteps = 10
        b = z.size(0)
        device = z.device
        dt = 1.0 / sample_timesteps
        dt = torch.tensor([dt] * b, device=device).view(b, *([1] * (z.dim()-1)))
        images = [z]
        for i in tqdm(reversed(range(sample_timesteps + 1)), desc='sampling loop time step', total=(sample_timesteps + 1)):
            t_val = i / sample_timesteps
            t_tensor = torch.tensor([t_val] * b, device=device)
            # オンラインネットワークの出力を取得
            y = self.model_e(z, t_tensor, cond)
            # Classifier-Free Guidance（null_condが指定されている場合）
            if null_cond is not None:
                y_null = self.model_e(z, t_tensor, null_cond)
                y = y_null + cfg * (y - y_null)
            # CTでは、出力が直接 x0 の推定とみなすので、z を更新
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
    ema_decay=0.99

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
