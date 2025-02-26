# train.py
import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from dit import DiT_Llama

# InstaFlowベースのモデル（Rectified Flowを単純化）
class InstaFlowModel(nn.Module):
    def __init__(self, in_channels=3, input_size=32, dim=64, n_layers=6, n_heads=8):
        super(InstaFlowModel, self).__init__()
        # 簡易的なRectified Flowモデル（DiTベースを模倣）
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=3, padding=1),
            nn.ReLU(),
            *[nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                nn.ReLU()
            ) for _ in range(n_layers)],
            nn.Conv2d(dim, in_channels, kernel_size=3, padding=1)
        )
        self.n_heads = n_heads

    def forward(self, x, t=None, cond=None):
        # InstaFlowの単一ステップ用フォワードパス
        # tはRectified Flowで必要最小限（1ステップ用に簡略化）
        t_embedding = self._time_embedding(t if t is not None else torch.zeros_like(x[:, :1, :, :]))
        return self.model(x + t_embedding)

    def _time_embedding(self, t, n_heads):
        # 簡易的な時間埋め込み（InstaFlowのRectified Flow用）
        return t.repeat(1, 3, 1, 1)  # チャネル3に拡張（仮定）

# InstaFlow用のConsistencyModel（1ステップ最適化）
class InstaFlowConsistencyModel:
    def __init__(self, model_e, model_t, ln=True):
        """
        InstaFlow用のConsistencyModel初期化

        Args:
            model_e: オンラインネットワーク（学習対象）
            model_t: ターゲットネットワーク（EMAで更新）
            ln: Trueの場合、時刻サンプリングにシグモイドを適用
        """
        self.model_e = model_e
        self.model_t = model_t
        self.ln = ln

    def forward(self, x, cond):
        """
        InstaFlowの損失計算（Rectified Flowベース、1ステップ）

        1. 単一ステップ（t=0からt=1へのトランジション）を仮定。
        2. ノイズを加えたx_tを生成。
        3. model_eとmodel_tの予測差とデータとの差を計算。

        Returns:
            loss: 損失値（スカラー）
            info: ログ情報（None）
        """
        b = x.size(0)
        device = x.device

        # 単一ステップ用にt=0.5を仮定（InstaFlowのRectified Flowを単純化）
        t = torch.full((b,), 0.5, device=device).view(b, *[1] * (x.dim() - 1))

        # ノイズを生成
        noise = torch.randn_like(x)
        x_t = (1 - t) * x + t * noise  # Rectified Flowの単一ステップトランジション

        # オンラインネットワークとターゲットネットワークの予測
        pred_e = self.model_e(x_t, t, cond)
        pred_t = self.model_t(x_t, t, cond)

        # InstaFlowの損失：予測差（CD Loss）とデータ差（Data Loss）の組み合わせ
        cd_loss = F.mse_loss(pred_e, pred_t, reduction='mean')
        data_loss = F.mse_loss(pred_e, x, reduction='mean')
        loss = cd_loss + 3.5 * data_loss  # 重み付き損失

        return loss, {"cd_loss": cd_loss.item(), "data_loss": data_loss.item()}

    @torch.no_grad()
    def sample(self, z, cond, null_cond=None, cfg=2.5):
        """
        InstaFlowの1ステップサンプリング

        初期ノイズzから1ステップでx0を推定。

        Returns:
            images: 最終的な生成画像
        """
        device = z.device
        t = torch.zeros([z.shape[0], 1, 1, 1], device=device)  # 1ステップ用t=0

        # オンラインネットワークの出力を取得
        y = self.model_e(z, t, cond)
        if null_cond is not None:
            y_null = self.model_e(z, t, null_cond)
            y = y_null + cfg * (y - y_null)  # Classifier-Free Guidance

        # 1ステップで直接x0を推定
        final_image = y.clamp(-1, 1)  # [-1, 1]の範囲にクランプ
        return [final_image]  # 1ステップのみのリスト

# EMA更新関数
def update_target_network(model_e, model_t, decay):
    with torch.no_grad():
        for param_t, param_e in zip(model_t.parameters(), model_e.parameters()):
            param_t.data.mul_(decay).add_(param_e.data, alpha=1 - decay)

def main():
    parser = argparse.ArgumentParser(description="Choose dataset among: mnist, cifar, fashion_mnist, huggan")
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "cifar", "fashion_mnist", "huggan"],
                        help="Dataset to use (default: mnist)")
    args = parser.parse_args()

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

    batch_size = config["batch_size"]
    timesteps = config["timesteps"]
    cfg = config["cfg"]
    channels = config["model"]["in_channels"]
    image_size = config["model"]["input_size"]
    epochs = config["training"]["epochs"]
    lr = config["training"]["learning_rate"]
    ema_decay = 0.999

    output_dir = "outputs"
    img_dir = Path(output_dir, f"{config['dataset']}")
    img_dir.mkdir(exist_ok=True, parents=True)

    transform = transforms.Compose([
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 2) - 1)
    ]) if config["dataset"] == "cifar" else transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 2) - 1)
    ])

    if config["dataset"] == "cifar":
        dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    elif config["dataset"] == "fashion_mnist":
        dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    elif config["dataset"] in ["huggan", "huggan/AFHQv2"]:
        from datasets import load_dataset
        fdataset = load_dataset("huggan/afhqv2")
        dataset = fdataset["train"].filter(lambda x: x["label"] == 0)
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
    else:
        dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    model_e = DiT_Llama(
        in_channels=config["model"]["in_channels"],
        input_size=config["model"]["input_size"],
        patch_size=config["model"]["patch_size"],
        dim=config["model"]["dim"],
        n_layers=config["model"]["n_layers"],
        n_heads=config["model"]["n_heads"],
        multiple_of=config["model"]["multiple_of"],
        ffn_dim_multiplier=config["model"]["ffn_dim_multiplier"],
        norm_eps=config["model"]["norm_eps"],
        class_dropout_prob=config["model"]["class_dropout_prob"],
        num_classes=config["model"]["num_classes"] + 1
    ).to(device)

    model_t = DiT_Llama(
        in_channels=config["model"]["in_channels"],
        input_size=config["model"]["input_size"],
        patch_size=config["model"]["patch_size"],
        dim=config["model"]["dim"],
        n_layers=config["model"]["n_layers"],
        n_heads=config["model"]["n_heads"],
        multiple_of=config["model"]["multiple_of"],
        ffn_dim_multiplier=config["model"]["ffn_dim_multiplier"],
        norm_eps=config["model"]["norm_eps"],
        class_dropout_prob=config["model"]["class_dropout_prob"],
        num_classes=config["model"]["num_classes"] + 1
    ).to(device)

    cm = InstaFlowConsistencyModel(model_e, model_t)
    optimizer = optim.Adam(model_e.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        losses = []
        bar = tqdm(dataloader, desc=f"Epoch {epoch}", total=len(dataloader), leave=True)
        cm.model_e.train()

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
            loss, info = cm.forward(x, c)
            loss.backward()
            optimizer.step()
            update_target_network(model_e, model_t, ema_decay)
            losses.append(loss.item())
            bar.set_postfix({"Avg Loss": f"{torch.mean(torch.tensor(losses)):.4f}",
                           "CD Loss": f"{info['cd_loss']:.4f}",
                           "Data Loss": f"{info['data_loss']:.4f}"})

        # サンプル生成（4x4グリッド、1ステップ）
        cm.model_e.eval()
        with torch.no_grad():
            cond = torch.arange(0, 16).to(device) % config["model"]["num_classes"]
            uncond = torch.ones_like(cond) * config["model"]["num_classes"]

            init_noise = torch.randn(16, channels, image_size, image_size).to(device)
            images = cm.sample(init_noise, cond, uncond, cfg)

            # 最終画像を取得し、[0, 1]に正規化
            final_image = images[-1].clamp(-1, 1) * 0.5 + 0.5
            grid = make_grid(final_image.float(), nrow=4)
            save_image(grid, f"{img_dir}/sample_epoch_{epoch}.png")

        # モデルの保存
        torch.save(model_e.state_dict(), f"{img_dir}/model_epoch_{epoch}.pth")

    print("Training complete!")

if __name__ == "__main__":
    main()