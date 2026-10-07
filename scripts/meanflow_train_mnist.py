import os
import sys

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from mnist_dataset import MNISTDataset
from dit import DiT
from meanflow import MeanFlow

device = "cuda"
epochs = 100
batch_size = 256
lr = 1e-4
num_classes = 10
cfg_scale = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0  # ω of training-time CFG, 1.0 = none
p_uncond = 0.1 if cfg_scale != 1.0 else 0.0  # CFG needs the model to learn the unconditional velocity
out_dir = "results/train_mnist_meanflow" if cfg_scale == 1.0 else f"results/train_mnist_meanflow_cfg{cfg_scale}"
os.makedirs(out_dir, exist_ok=True)

dataset = MNISTDataset(root="data/mnist/mnist")
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
model = DiT(num_classes=num_classes, use_r=True).to(device)  # trained from scratch
mf = MeanFlow(cfg_scale=cfg_scale, null_label=num_classes)
opt = torch.optim.AdamW(model.parameters(), lr=lr)

for epoch in range(epochs):
    model.train()
    for step, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        y = torch.where(torch.rand(y.shape, device=device) < p_uncond, num_classes, y)
        loss, mse = mf.loss(model, x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(f"epoch {epoch} step {step} loss {loss.item():.4f} mse {mse.item():.4f}")

    if epoch % 10 == 0 or epoch == epochs - 1:
        model.eval()
        y = torch.arange(num_classes, device=device).repeat_interleave(10)
        for ts in [(1.0, 0.0), (1.0, 0.5, 0.0)]:
            samples = mf.sample(model, (len(y), 1, 28, 28), cond=y, ts=ts, device=device)
            save_image(samples, f"{out_dir}/epoch_{epoch:03d}_{len(ts) - 1}step.png", nrow=10, normalize=True, value_range=(-1, 1))
        torch.save(model.state_dict(), f"{out_dir}/model.pt")
