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
        self.sigma_min = 0.002  # 最小ノイズスケール（維持）
        self.sigma_max = 0.1    # 最大ノイズスケールを小さく調整

    def get_sigma(self, t):
        """
        ノイズスケジュールを計算（指数スケジュール）
        """
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def train(self, mode=True):
        """訓練モードを設定"""
        self.model_e.train(mode)
        self.model_t.eval()  # ターゲットモデルは通常評価モード
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
            t_grid = torch.sigmoid(t_grid)  # ln=True の場合、シグモイドを適用

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

        # ノイズスケジュールを計算
        sigma_n = self.get_sigma(t_n)
        sigma_np1 = self.get_sigma(t_np1)

        # ノイズスケールをクリッピング（ゼロや極端な値を防ぐ）
        sigma_n = torch.clamp(sigma_n, min=1e-5, max=1e1)  # 最大値を1e1に制限
        sigma_np1 = torch.clamp(sigma_np1, min=1e-5, max=1e1)

        # ノイズを生成して、各時刻の画像を作成
        noise_n = torch.randn_like(x)
        noise_np1 = torch.randn_like(x)
        x_tn = x + sigma_n * noise_n  # 指数スケジュールを使用
        x_tnp1 = x + sigma_np1 * noise_np1

        # モデル出力のログ
        print(f"Input x shape: {x.shape}, mean: {x.mean()}, std: {x.std()}")
        print(f"x_tn shape: {x_tn.shape}, mean: {x_tn.mean()}, std: {x_tn.std()}")
        print(f"x_tnp1 shape: {x_tnp1.shape}, mean: {x_tnp1.mean()}, std: {x_tnp1.std()}")

        # オンラインネットワーク: t_{n+1} における出力
        # cond は DiT_Llama のラベル（y）として渡す
        y_pred = self.model_e(x_tnp1, t_np1_tensor, cond)
        # ターゲットネットワーク: t_n における出力
        y_target = self.model_t(x_tn, t_n_tensor, cond)

        # モデル出力をクリッピング（[-1, 1] に厳密に制限）
        y_pred = torch.clamp(y_pred, min=-1.0, max=1.0)  # [-1, 1] の範囲に厳密にクリップ
        y_target = torch.clamp(y_target, min=-1.0, max=1.0)

        # モデル出力のログ
        print(f"y_pred shape: {y_pred.shape}, mean: {y_pred.mean()}, std: {y_pred.std()}")
        print(f"y_target shape: {y_target.shape}, mean: {y_target.mean()}, std: {y_target.std()}")

        # 形状確認と安全な dims 計算
        if y_pred.dim() <= 1:
            raise ValueError(f"y_pred dimension is too low: {y_pred.dim()}, shape: {y_pred.shape}")
        dims = tuple(range(1, y_pred.dim())) if y_pred.dim() > 1 else tuple()

        # MSEを計算（sigmaスケーリングを適用、負の損失を防ぐ）
        diff = (y_pred - y_target) / (sigma_np1 ** 2)
        # 負の損失を防ぐために、絶対値を使用するか、損失を正の値に変換
        loss = torch.abs(diff.mean(dim=dims).mean()) if dims else torch.abs(diff.mean())
        # または、論文のAlgorithm 3に基づき、以下のように修正
        loss = ((y_pred - y_target) ** 2 / (sigma_np1 ** 2)).mean()

        # 損失が発散しないようにクリッピング（範囲を広げる）
        loss = torch.clamp(loss, min=0.0, max=10.0)  # 負の値を防ぐため最小値を0.0に

        # デバッグ用ログ
        print(f"t_n: {t_n}, t_np1: {t_np1}")
        print(f"sigma_n: {sigma_n.mean()}, sigma_np1: {sigma_np1.mean()}")
        print(f"y_pred mean: {y_pred.mean()}, y_pred std: {y_pred.std()}")
        print(f"y_target mean: {y_target.mean()}, y_target std: {y_target.std()}")
        print(f"(y_pred - y_target) mean: {(y_pred - y_target).mean()}")
        print(f"sigma_np1 ** 2: {sigma_np1 ** 2}")
        print(f"loss before clamp: {loss.item()}")
        print(f"loss after clamp: {loss.item()}")

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
        b = z.size(0)
        device = z.device

        images = [z]
        for i in tqdm(reversed(range(self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            t_val = i / (self.timesteps - 1)  # 0から1の範囲を正確にカバー
            if self.ln:
                t_val = torch.sigmoid(torch.tensor(t_val, device=device))
            t_tensor = torch.tensor([t_val] * b, device=device).view(b, *([1] * (z.dim()-1)))

            # オンラインネットワークの出力を取得
            y = self.model_e(z, t_tensor, cond)
            # Classifier-Free Guidance（null_condが指定されている場合）
            if null_cond is not None:
                # null_cond は無条件ラベル（例: num_classes）
                y_null = self.model_e(z, t_tensor, null_cond)
                y = y_null + cfg * (y - y_null)

            # ノイズスケジュールを計算
            sigma_t = self.get_sigma(t_val).view(b, *([1] * (z.dim()-1)))
            sigma_t = torch.clamp(sigma_t, min=1e-5, max=1e1)  # ノイズスケールをクリッピング
            noise = torch.randn_like(z) * sigma_t  # ノイズをスケーリング
            z = y + noise  # モデル予測にノイズを加算

            # デバッグ用ログ（必要に応じてコメントアウト）
            print(f"t_val: {t_val}, sigma_t: {sigma_t.mean()}")
            print(f"y mean: {y.mean()}, y std: {y.std()}")

            images.append(z)
        return images

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # DiT_Llama_600M_patch2 モデルの初期化（プリセットを使用）
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
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)  # バッチサイズを16に設定

    # 学習ループ
    epochs = 1000
    for epoch in range(1, epochs + 1):
        # 訓練モードに設定
        model_e.train()  # model_e を訓練モードに
        model_t.eval()   # model_t は通常評価モード

        total_loss = 0
        for x, y in tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}"):
            x, y = x.to(device), y.to(device)  # y は [batch_size] の形状（0～9）
            print(f"Input labels (cond) shape: {y.shape}, values: {y[:5]}")  # デバッグ用
            loss, _ = cm.forward(x, y)  # y を cond として使用
            optimizer = torch.optim.Adam(model_e.parameters(), lr=2e-4)  # 仮のオプティマイザ
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # EMA更新（ターゲットモデルの更新）
            for param_e, param_t in zip(model_e.parameters(), model_t.parameters()):
                param_t.data.mul_(0.999).add_(param_e.data, alpha=1 - 0.999)
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}/{epochs}, Average Loss: {avg_loss:.4f}")

        # 評価モードに設定
        model_e.eval()  # 評価モードに
        model_t.eval()  # 評価モードに

        # エポック終了時の処理
        torch.save(model_e.state_dict(), f"outputs_cifar10_ct_ema/model_e_epoch_{epoch}.pth")
        torch.save(model_t.state_dict(), f"outputs_cifar10_ct_ema/model_t_epoch_{epoch}.pth")

        # サンプル画像の生成と保存
        z = torch.randn(16, 3, 32, 32, device=device) * cm.sigma_max  # 初期ノイズ
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