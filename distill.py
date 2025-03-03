import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from diffusers import UNet2DConditionModel
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm


class InstaFlow(nn.Module):
    def __init__(self, unet, timesteps=5):
        super().__init__()
        self.model = unet
        self.timesteps = timesteps

    def forward(self, x, cond):
        """1-Rectified Flowのトレーニング"""
        b = x.size(0)
        device = x.device
        t_grid = torch.linspace(0, 1, steps=self.timesteps, device=device)
        n = torch.randint(1, self.timesteps, (b,), device=device)
        t = t_grid[n]
        texp = t.view(b, 1, 1, 1)
        
        z1 = torch.randn_like(x)
        z1 = torch.clamp(z1, min=-3.0, max=3.0)
        zt = (1 - texp) * x + texp * z1
        
        # condを[batch_size, 1, cross_attention_dim]に調整
        cond = cond.unsqueeze(1)  # [b, 1, 10]
        vtheta = self.model(zt, t, encoder_hidden_states=cond)
        batchwise_mse = ((z1 - x - vtheta.sample) ** 2).mean(dim=list(range(1, len(x.shape))))
        return batchwise_mse.mean(), None

    @torch.no_grad()
    def sample(self, z, cond, null_cond=None, cfg=2.0):
        """1ステップ生成"""
        b = z.size(0)
        device = z.device
        t = torch.ones(b, device=device)
        
        cond = cond.unsqueeze(1)  # [b, 1, 10]
        vc = self.model(z, t, encoder_hidden_states=cond).sample
        if null_cond is not None:
            null_cond = null_cond.unsqueeze(1)
            vu = self.model(z, t, encoder_hidden_states=null_cond).sample
            vc = vu + cfg * (vc - vu)
        x = z - vc
        return x

    @torch.no_grad()
    def generate_pairs(self, x, cond):
        """サンプルペア生成 (Z_0^k, Z_1^k)"""
        b = x.size(0)
        device = x.device
        z1 = torch.randn_like(x)
        z1 = torch.clamp(z1, min=-3.0, max=3.0)
        x0 = self.sample(z1, cond)
        return x0, z1

    def reflow(self, x0, z1, cond):
        """2-Rectified Flow以降のトレーニング"""
        b = x0.size(0)
        device = x0.device
        t_grid = torch.linspace(0, 1, steps=self.timesteps, device=device)
        n = torch.randint(1, self.timesteps, (b,), device=device)
        t = t_grid[n]
        texp = t.view(b, 1, 1, 1)
        zt = (1 - texp) * x0 + texp * z1
        
        cond = cond.unsqueeze(1)  # [b, 1, 10]
        vtheta = self.model(zt, t, encoder_hidden_states=cond)
        batchwise_mse = ((z1 - x0 - vtheta.sample) ** 2).mean(dim=list(range(1, len(x0.shape))))
        return batchwise_mse.mean(), None

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
    print(f"Dataset loaded: {len(trainset)} samples, {len(trainloader)} batches")

    unet = UNet2DConditionModel(
        in_channels=3,
        out_channels=3,
        sample_size=32,
        block_out_channels=(64, 128, 256, 512),
        layers_per_block=2,
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
        cross_attention_dim=10
    ).to(device)

    instaf = InstaFlow(unet, timesteps=5).to(device)
    optimizer = optim.Adam(instaf.parameters(), lr=1e-4)
    epochs = 10

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("samples", exist_ok=True)

    print("Training 1-Rectified Flow...")
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (x, labels) in enumerate(tqdm(trainloader, desc=f"Epoch {epoch+1} (1-Rectified)")):
            x, labels = x.to(device), labels.to(device)
            cond = torch.nn.functional.one_hot(labels, num_classes=10).float().to(device)
            loss, _ = instaf.forward(x, cond)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(instaf.parameters(), max_norm=1.0)
            optimizer.step()

        avg_loss = total_loss / len(trainloader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

        z_sample = torch.randn(16, 3, 32, 32).to(device)
        cond_sample = torch.nn.functional.one_hot(torch.randint(0, 10, (16,), device=device), num_classes=10).float()
        generated = instaf.sample(z_sample, cond_sample)
        save_image(generated, f"samples/stage1_batch{batch_idx}_samples.png", nrow=4, normalize=True)

    torch.save(instaf.state_dict(), "checkpoints/1_rectified.pth")

    print("Generating pairs for 2-Rectified Flow...")
    sample_batch = next(iter(trainloader))
    x_sample, labels_sample = sample_batch
    x_sample, labels_sample = x_sample.to(device), labels_sample.to(device)
    cond_sample = torch.nn.functional.one_hot(labels_sample, num_classes=10).float()
    z0_1, z1_1 = instaf.generate_pairs(x_sample, cond_sample)

    print("Training 2-Rectified Flow...")
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx in tqdm(range(len(trainloader)), desc=f"Epoch {epoch+1} (2-Rectified)"):
            loss, _ = instaf.reflow(z0_1, z1_1, cond_sample)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(instaf.parameters(), max_norm=1.0)
            optimizer.step()

        avg_loss = total_loss / len(trainloader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

        z_sample = torch.randn(16, 3, 32, 32).to(device)
        cond_sample_gen = torch.nn.functional.one_hot(torch.randint(0, 10, (16,), device=device), num_classes=10).float()
        generated = instaf.sample(z_sample, cond_sample_gen)
        save_image(generated, f"samples/stage2_batch{batch_idx}_samples.png", nrow=4, normalize=True)

    torch.save(instaf.state_dict(), "checkpoints/2_rectified.pth")

    print("Generating pairs for 3-Rectified Flow...")
    z0_2, z1_2 = instaf.generate_pairs(z0_1, cond_sample)

    print("Training 3-Rectified Flow...")
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx in tqdm(range(len(trainloader)), desc=f"Epoch {epoch+1} (3-Rectified)"):
            loss, _ = instaf.reflow(z0_2, z1_2, cond_sample)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(instaf.parameters(), max_norm=1.0)
            optimizer.step()

        avg_loss = total_loss / len(trainloader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

        z_sample = torch.randn(16, 3, 32, 32).to(device)
        cond_sample_gen = torch.nn.functional.one_hot(torch.randint(0, 10, (16,), device=device), num_classes=10).float()
        generated = instaf.sample(z_sample, cond_sample_gen)
        save_image(generated, f"samples/stage3_batch{batch_idx}_samples.png", nrow=4, normalize=True)

    torch.save(instaf.state_dict(), "checkpoints/3_rectified.pth")

    print("Generating final samples...")
    z = torch.randn(16, 3, 32, 32).to(device)
    cond_final = torch.nn.functional.one_hot(torch.randint(0, 10, (16,), device=device), num_classes=10).float()
    generated = instaf.sample(z, cond_final)
    final_batch_idx = len(trainloader) - 1
    save_image(generated, f"samples/stage3_batch{final_batch_idx}_samples.png", nrow=4, normalize=True)
    print(f"Generated shape: {generated.shape}")

if __name__ == "__main__":
    main()