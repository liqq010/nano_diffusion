import os

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from mnist_dataset import MNISTDataset
from dit import DiT
from rf import RectifiedFlow

device = "cuda"
epochs = 50
batch_size = 256
lr = 3e-4
num_classes = 10
p_uncond = 0.1
t_sampling = "logit_normal"  # "uniform" / "logit_normal"
shift = 1.0
out_dir = "results/train_mnist_rf"
os.makedirs(out_dir, exist_ok=True)

dataset = MNISTDataset(root="data/mnist/mnist")
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
model = DiT(num_classes=num_classes).to(device)
rf = RectifiedFlow(t_sampling=t_sampling, shift=shift)
opt = torch.optim.AdamW(model.parameters(), lr=lr)

for epoch in range(epochs):
    model.train()
    for step, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        y = torch.where(torch.rand(y.shape, device=device) < p_uncond, num_classes, y)
        loss = rf.loss(model, x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(f"epoch {epoch} step {step} loss {loss.item():.4f}")

    if epoch % 10 == 0 or epoch == epochs - 1:
        model.eval()
        y = torch.arange(num_classes, device=device).repeat_interleave(10)
        samples = rf.sample(model, (len(y), 1, 28, 28), cond=y, steps=50, device=device)
        save_image(samples, f"{out_dir}/epoch_{epoch:03d}.png", nrow=10, normalize=True, value_range=(-1, 1))
        torch.save(model.state_dict(), f"{out_dir}/model.pt")
