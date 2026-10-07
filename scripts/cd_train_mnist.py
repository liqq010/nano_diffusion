import copy
import os

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from mnist_dataset import MNISTDataset
from dit import DiT
from cm import ConsistencyModel

device = "cuda"
epochs = 50
batch_size = 256
lr = 1e-4
num_classes = 10
N = 20  # fixed number of discretization steps (no curriculum in the original CD)
cfg_scale = 3.0  # teacher guidance, distilled into the student; the student only learns conditional generation, no label dropout needed
teacher_ckpt = "results/train_mnist_rf/model.pt"
out_dir = "results/train_mnist_cd"
os.makedirs(out_dir, exist_ok=True)

dataset = MNISTDataset(root="data/mnist/mnist")
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

teacher = DiT(num_classes=num_classes).to(device)
teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device))
teacher.eval().requires_grad_(False)

model = copy.deepcopy(teacher).train().requires_grad_(True)  # original CD: the student is initialized from the teacher
cm = ConsistencyModel()
opt = torch.optim.AdamW(model.parameters(), lr=lr)

for epoch in range(epochs):
    model.train()
    for step, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        loss = cm.loss_cd(model, teacher, x, y, N=N, cfg_scale=cfg_scale)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(f"epoch {epoch} step {step} loss {loss.item():.4f}")

    if epoch % 5 == 0 or epoch == epochs - 1:
        y = torch.arange(num_classes, device=device).repeat_interleave(10)
        model.eval()
        for ts in [(1.0,), (1.0, 0.5)]:
            samples = cm.sample(model, (len(y), 1, 28, 28), cond=y, ts=ts, device=device)
            save_image(samples, f"{out_dir}/epoch_{epoch:03d}_{len(ts)}step.png", nrow=10, normalize=True, value_range=(-1, 1))
        torch.save(model.state_dict(), f"{out_dir}/model.pt")
