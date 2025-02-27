import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

# U-Net Architecture Definition
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512], time_emb_dim=32, cond_emb_dim=16):
        super(UNet, self).__init__()
        
        # Time embedding
        self.time_emb = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim * 4)
        )
        
        # Condition embedding
        self.cond_emb = nn.Linear(cond_emb_dim, cond_emb_dim * 4)
        
        # Contracting path (downsampling)
        self.conv_blocks_down = nn.ModuleList()
        in_ch = in_channels
        for feature in features:
            block = nn.Sequential(
                nn.Conv2d(in_ch, feature, kernel_size=3, padding=1),
                nn.BatchNorm2d(feature),
                nn.ReLU(),
                nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                nn.BatchNorm2d(feature),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.conv_blocks_down.append(block)
            in_ch = feature
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[-1], features[-1] * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(features[-1] * 2),
            nn.ReLU(),
            nn.Conv2d(features[-1] * 2, features[-1] * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(features[-1] * 2),
            nn.ReLU()
        )
        
        # Expanding path (upsampling)
        self.conv_blocks_up = nn.ModuleList()
        for i in range(len(features) - 1, -1, -1):
            out_ch = features[i] if i > 0 else out_channels
            block = nn.Sequential(
                nn.ConvTranspose2d(features[i] * 2 if i < len(features) - 1 else features[i] * 2, 
                                 out_ch, kernel_size=2, stride=2),
                nn.Conv2d(out_ch * 2, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU()
            )
            self.conv_blocks_up.append(block)
        
        # Skip connections will be handled in the forward pass
        
    def forward(self, x, t, cond):
        # Time embedding
        t = t.view(-1, 1)  # Reshape time to [batch_size, 1]
        t_emb = self.time_emb(t)
        
        # Condition embedding
        cond_emb = self.cond_emb(cond)
        
        # Store skip connections
        skip_connections = []
        
        # Contracting path
        for block in self.conv_blocks_down:
            x = block(x)
            skip_connections.append(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Expanding path
        for i, block in enumerate(self.conv_blocks_up):
            x = block(x)
            # Skip connection: concatenate with corresponding downsampled feature
            skip = skip_connections[len(skip_connections) - 1 - i]
            x = torch.cat([x, skip], dim=1)
        
        return x

# Rectified Flow Class
class RF:
    def __init__(self, model, timesteps, ln=True):
        self.model = model
        self.ln = ln
        self.timesteps = timesteps

    def forward(self, x, cond):
        b = x.size(0)
        if self.ln:
            nt = torch.randn((b,)).to(x.device)
            t = torch.sigmoid(nt)
        else:
            t = torch.rand((b,)).to(x.device)
        texp = t.view([b, *([1] * len(x.shape[1:]))])
        z1 = torch.randn_like(x)
        zt = (1 - texp) * x + texp * z1
        vtheta = self.model(zt, t, cond)  # U-Net takes zt, t, and cond as input
        batchwise_mse = ((z1 - x - vtheta) ** 2).mean(dim=list(range(1, len(x.shape))))
        tlist = batchwise_mse.detach().cpu().reshape(-1).tolist()
        ttloss = [(tv, tloss) for tv, tloss in zip(t, tlist)]
        return batchwise_mse.mean(), ttloss

    @torch.no_grad()
    def sample(self, z, cond, null_cond, cfg=2.0):
        b = z.size(0)
        dt = 1.0
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        images = [z]
        t = 1.0
        t = torch.tensor([t] * b).to(z.device)

        vc = self.model(z, t, cond)

        # Classifier Free Guidance
        if null_cond is not None:
            vu = self.model(z, t, null_cond)
            vc = vu + cfg * (vc - vu)

        z = z - dt * vc
        images.append(z)
        return images

# Main Function
def main():
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Choose dataset among: mnist, cifar, fashion_mnist, huggan")
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "cifar", "fashion_mnist", "huggan"],
                        help="Dataset to use (default: mnist)")
    args = parser.parse_args()

    # Load configuration based on dataset
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

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configuration parameters
    batch_size = config["batch_size"]
    timesteps = config["timesteps"]
    cfg = config["cfg"]
    channels = config["model"]["in_channels"]
    image_size = config["model"]["input_size"]

    # Dataset and preprocessing
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

        fdataset = load_dataset("huggan/afhqv2")
        dataset = fdataset["train"]
        dataset = dataset.filter(lambda x: x["label"] == 0)
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
    else:  # Default to MNIST
        dataset_name = "mnist"
        fdataset = datasets.MNIST
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x * 2) - 1)
        ])
        dataset = fdataset(root="./data", train=True, download=True, transform=transform)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Initialize U-Net model
    model = UNet(
        in_channels=channels,
        out_channels=channels,
        features=[64, 128, 256, 512],  # Adjust based on your needs
        time_emb_dim=32,  # Size of time embedding
        cond_emb_dim=16   # Size of condition embedding
    ).to(device)

    # Training parameters
    training_config = config["training"]
    epochs = training_config["epochs"]
    lr = training_config["learning_rate"]

    # Output directory
    output_dir = "outputs"
    img_dir = Path(output_dir, f"{config['dataset']}")
    img_dir.mkdir(exist_ok=True, parents=True)

    # Initialize RF and optimizer
    rf = RF(model, timesteps=timesteps)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(1, epochs + 1):
        losses = []
        bar = tqdm(dataloader, desc=f"Epoch {epoch}", total=len(dataloader))
        model.train()

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
            loss, blsct = rf.forward(x, c)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            bar.set_postfix({"Average Loss": f"{torch.mean(torch.tensor(losses)):.4f}"})

        # Sampling
        rf.model.eval()
        with torch.no_grad():
            cond = torch.arange(0, 16).to(device) % config["model"]["num_classes"]
            uncond = torch.ones_like(cond) * config["model"]["num_classes"]

            init_noise = torch.randn(16, channels, image_size, image_size).to(device)
            images = rf.sample(init_noise, cond, uncond, cfg)

            final_image = images[-1]
            final_image = final_image * 0.5 + 0.5  # Unnormalize from [-1, 1] to [0, 1]
            final_image = final_image.clamp(0, 1)
            grid = make_grid(final_image.float(), nrow=4)
            save_image(grid, f"{img_dir}/sample_{epoch}_last.png")

        # Save model
        torch.save(model.state_dict(), img_dir / "unet_rectified_flow.pth")

    print("Training complete.")

if __name__ == "__main__":
    main()
