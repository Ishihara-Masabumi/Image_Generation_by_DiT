import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from dit import DiT_Llama_600M_patch2


class ConsistencyModel:
    def __init__(self, model_e, model_t, timesteps, ln=True):
        self.model_e = model_e
        self.model_t = model_t
        self.timesteps = timesteps
        self.ln = ln
        self.sigma_min = 0.002  # 最小ノイズスケール
        self.sigma_max = 0.1    # 最大ノイズスケール

    def get_sigma(self, t):
        """ノイズスケジュールを計算（指数スケジュール）"""
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def train(self, mode=True):
        """訓練モードを設定"""
        self.model_e.train(mode)
        self.model_t.eval()  # ターゲットモデルは評価モード
        return self

    def eval(self):
        """評価モードを設定"""
        return self.train(False)

    def forward(self, x, cond):
        b = x.size(0)
        device = x.device

        # 0〜1の時刻グリッドを生成
        t_grid = torch.linspace(0, 1, steps=self.timesteps, device=device)
        if self.ln:
            t_grid = torch.sigmoid(t_grid)

        # ランダムに隣接する2つの時刻のインデックスを選択
        n = torch.randint(0, self.timesteps - 1, (1,)).item()
        t_n = t_grid[n]
        t_np1 = t_grid[n + 1]

        # バッチ用の時刻テンソルを作成
        t_n_tensor = t_n * torch.ones(b, device=device)
        t_np1_tensor = t_np1 * torch.ones(b, device=device)

        # xの次元に合わせて時刻テンソルを拡張
        #t_n_expanded = t_n_tensor.view(b, *([1] * (x.dim() - 1)))
        #t_np1_expanded = t_np1_tensor.view(b, *([1] * (x.dim() - 1)))

        # ノイズスケジュールを計算
        sigma_n = self.get_sigma(t_n)
        sigma_np1 = self.get_sigma(t_np1)

        # ノイズスケールをクリッピング
        sigma_n = torch.clamp(sigma_n, min=1e-5, max=1e1)
        sigma_np1 = torch.clamp(sigma_np1, min=1e-5, max=1e1)

        # ノイズを生成して、各時刻の画像を作成
        noise_n = torch.randn_like(x)
        noise_np1 = torch.randn_like(x)
        x_tn = x + sigma_n * noise_n
        x_tnp1 = x + sigma_np1 * noise_np1

        # モデル出力のログ
        #print(f"Input x shape: {x.shape}, mean: {x.mean()}, std: {x.std()}")
        #print(f"x_tn shape: {x_tn.shape}, mean: {x_tn.mean()}, std: {x_tn.std()}")
        #print(f"x_tnp1 shape: {x_tnp1.shape}, mean: {x_tnp1.mean()}, std: {x_tnp1.std()}")

        # オンラインネットワーク: t_{n+1} における出力
        y_pred = self.model_e(x_tnp1, t_np1_tensor, cond)
        # ターゲットネットワーク: t_n における出力
        y_target = self.model_t(x_tn, t_n_tensor, cond)

        # モデル出力をクリッピング
        y_pred = torch.clamp(y_pred, min=-1.0, max=1.0)
        y_target = torch.clamp(y_target, min=-1.0, max=1.0)

        # モデル出力のログ
        #print(f"y_pred shape: {y_pred.shape}, mean: {y_pred.mean()}, std: {y_pred.std()}")
        #print(f"y_target shape: {y_target.shape}, mean: {y_target.mean()}, std: {y_target.std()}")

        # 修正: 単純なMSEを使用
        diff = (y_pred - y_target) ** 2
        loss_raw = diff.mean()  # クリッピング前の損失
        loss = torch.clamp(loss_raw, min=0.0, max=10.0)

        # デバッグ用ログ
        #print(f"t_n: {t_n}, t_np1: {t_np1}")
        #print(f"sigma_n: {sigma_n}, sigma_np1: {sigma_np1}")
        #print(f"y_pred mean: {y_pred.mean()}, y_pred std: {y_pred.std()}")
        #print(f"y_target mean: {y_target.mean()}, y_target std: {y_target.std()}")
        #print(f"(y_pred - y_target) mean: {(y_pred - y_target).mean()}")
        #print(f"loss before clamp: {loss_raw.item()}")
        #print(f"loss after clamp: {loss.item()}")

        return loss, None

    @torch.no_grad()
    def sample(self, z, cond, null_cond=None, cfg=2.0):
        b = z.size(0)
        device = z.device

        images = [z]
        for i in tqdm(reversed(range(self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            t_val = i / (self.timesteps - 1)
            if self.ln:
                t_val = torch.sigmoid(torch.tensor(t_val, device=device))
            t_tensor = torch.tensor([t_val] * b, device=device).view(b, *([1] * (z.dim() - 1)))

            # オンラインネットワークの出力を取得
            y = self.model_e(z, t_tensor, cond)
            if null_cond is not None:
                y_null = self.model_e(z, t_tensor, null_cond)
                y = y_null + cfg * (y - y_null)

            # ノイズスケジュールを計算
            sigma_t = self.get_sigma(t_val).view(b, *([1] * (z.dim() - 1)))
            sigma_t = torch.clamp(sigma_t, min=1e-5, max=1e1)
            noise = torch.randn_like(z) * sigma_t
            z = y + noise

            images.append(z)
        return images


def main():
    import os
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # モデルの初期化
    model_e = DiT_Llama_600M_patch2(in_channels=3, input_size=32, num_classes=10).to(device)
    model_t = DiT_Llama_600M_patch2(in_channels=3, input_size=32, num_classes=10).to(device)
    model_t.load_state_dict(model_e.state_dict())  # 初期状態を同期

    # ConsistencyModel の初期化
    cm = ConsistencyModel(model_e, model_t, timesteps=20, ln=True)

    # CIFAR-10 データのロード
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)

    # オプティマイザを学習率を下げて定義
    optimizer = torch.optim.Adam(model_e.parameters(), lr=1e-4)

    output_dir = "outputs_cifar10_ct_ema"
    os.makedirs(output_dir, exist_ok=True)

    # 学習ループ
    epochs = 1000
    for epoch in range(1, epochs + 1):
        model_e.train()
        model_t.eval()

        total_loss = 0
        first_batch = True  # 最初のバッチのみ Input Labels を表示

        with tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}") as pbar:
            for x, y in pbar:
                x, y = x.to(device), y.to(device)
                
                # 最初のバッチのみ Input Labels を表示
                if first_batch:
                    #print(f"\n[Epoch {epoch}] Input labels (cond) shape: {y.shape}, values: {y[:5].tolist()}")
                    first_batch = False
                
                loss, _ = cm.forward(x, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # EMA 更新
                for param_e, param_t in zip(model_e.parameters(), model_t.parameters()):
                    param_t.data.mul_(0.999).add_(param_e.data, alpha=1 - 0.999)

                total_loss += loss.item()

                # tqdm の進行状況バーに現在の損失を表示
                pbar.set_postfix({"Loss": loss.item()})

        # 各エポックの平均損失を計算
        avg_loss = total_loss / len(dataloader)
        print(f"\n[Epoch {epoch}] Average Loss: {avg_loss:.4f}\n")

        # 評価モードに設定
        model_e.eval()
        model_t.eval()

        # モデルの保存
        torch.save(model_e.state_dict(), f"outputs_cifar10_ct_ema/model_e_epoch_{epoch}.pth")
        torch.save(model_t.state_dict(), f"outputs_cifar10_ct_ema/model_t_epoch_{epoch}.pth")

        # サンプル画像の生成と保存
        z = torch.randn(16, 3, 32, 32, device=device) * cm.sigma_max
        samples = cm.sample(z, y[:16], null_cond=torch.full((16,), 10, device=device), cfg=2.0)
        samples = (samples[-1] * 0.5 + 0.5).clamp(0, 1)  # [-1, 1] -> [0, 1]
        import os

        from torchvision.utils import make_grid, save_image
        os.makedirs("outputs_cifar10_ct_ema", exist_ok=True)
        grid = make_grid(samples, nrow=4)
        save_image(grid, f"outputs_cifar10_ct_ema/samples_epoch_{epoch}.png")

    print("Training completed.")


if __name__ == "__main__":
    main()