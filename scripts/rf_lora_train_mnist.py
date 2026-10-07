import os
import sys

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from mnist_dataset import MNISTDataset
from dit import DiT
from rf import RectifiedFlow
from lora import apply_lora, lora_state_dict

# python scripts/rf_lora_train_mnist.py 4 [adaln]  -> LoRA rank 4 (optionally also on adaLN); python scripts/rf_lora_train_mnist.py full -> full fine-tuning
mode = sys.argv[1] if len(sys.argv) > 1 else "4"
adaln = "adaln" in sys.argv[2:]
device = "cuda"
epochs = 100
batch_size = 50  # 1000 images, 20 steps per epoch
lr = 1e-5 if mode == "full" else 1e-4
num_classes = 10
p_uncond = 0.1  # keep label dropout so CFG still works
base_ckpt = "results/train_mnist_rf/model.pt"
out_dir = "results/train_mnist_rf_ft" if mode == "full" else f"results/train_mnist_rf_lora_r{mode}" + ("_adaln" if adaln else "")
os.makedirs(out_dir, exist_ok=True)

dataset = MNISTDataset(root="data/mnist_outlined")
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
model = DiT(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(base_ckpt, map_location=device))
if mode != "full":
    targets = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2") + (("adaLN_modulation.1",) if adaln else ())
    apply_lora(model, r=int(mode), targets=targets)
params = [p for p in model.parameters() if p.requires_grad]
print(f"trainable params: {sum(p.numel() for p in params)}")
rf = RectifiedFlow()
opt = torch.optim.AdamW(params, lr=lr)

for epoch in range(epochs):
    for step, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        y = torch.where(torch.rand(y.shape, device=device) < p_uncond, num_classes, y)
        loss = rf.loss(model, x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 10 == 0:
            print(f"epoch {epoch} step {step} loss {loss.item():.4f}")

    if epoch % 10 == 0 or epoch == epochs - 1:
        y = torch.arange(num_classes, device=device).repeat_interleave(10)
        samples = rf.sample(model, (len(y), 1, 28, 28), cond=y, steps=50, device=device)
        save_image(samples, f"{out_dir}/epoch_{epoch:03d}.png", nrow=10, normalize=True, value_range=(-1, 1))
        torch.save(model.state_dict() if mode == "full" else lora_state_dict(model), f"{out_dir}/{'model' if mode == 'full' else 'lora'}.pt")
