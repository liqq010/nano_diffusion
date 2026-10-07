import os

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from mnist_dataset import MNISTDataset
from dit import DiT
from cm import ConsistencyModel

device = "cuda"
epochs = 100
batch_size = 256
lr = 1e-4
num_classes = 10
s0, s1 = 10, 1280  # start / end of the N curriculum
out_dir = "results/train_mnist_ct"
os.makedirs(out_dir, exist_ok=True)

dataset = MNISTDataset(root="data/mnist/mnist")
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
model = DiT(num_classes=num_classes).to(device)  # trained from scratch, no RF weights
cm = ConsistencyModel(s0=s0, s1=s1, total_iters=epochs * len(loader))
opt = torch.optim.AdamW(model.parameters(), lr=lr)

it = 0
for epoch in range(epochs):
    model.train()
    for step, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        loss, N = cm.loss_ct(model, x, y, it)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(f"epoch {epoch} step {step} N {N} loss {loss.item():.4f}")
        it += 1

    if epoch % 10 == 0 or epoch == epochs - 1:
        model.eval()
        y = torch.arange(num_classes, device=device).repeat_interleave(10)
        for ts in [(1.0,), (1.0, 0.5)]:
            samples = cm.sample(model, (len(y), 1, 28, 28), cond=y, ts=ts, device=device)
            save_image(samples, f"{out_dir}/epoch_{epoch:03d}_{len(ts)}step.png", nrow=10, normalize=True, value_range=(-1, 1))
        torch.save(model.state_dict(), f"{out_dir}/model.pt")
