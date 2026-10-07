import os
import torch
from torchvision.utils import save_image

from dit import DiT
from rf import RectifiedFlow

ckpt = "results/train_mnist_rf/model.pt"
shift = 1.0  # must match training
sampler = "euler"  # "euler" / "heun"
steps = 1  # heun uses 2 network calls per step
cfg_scale = 3.0  # 1.0 = no CFG
num_classes = 10

n_per_class = 10
seed = 0
out_dir = "results/test_mnist_rf"
device = "cuda"

torch.manual_seed(seed)

model = DiT(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(ckpt, map_location=device))
model.eval()
rf = RectifiedFlow(shift=shift)

y = torch.arange(num_classes, device=device).repeat_interleave(n_per_class)
shape = (len(y), 1, 28, 28)
if sampler == "euler":
    samples = rf.sample(model, shape, cond=y, steps=steps, cfg_scale=cfg_scale, device=device)
else:
    samples = rf.sample_heun(model, shape, cond=y, steps=steps, cfg_scale=cfg_scale, device=device)

os.makedirs(out_dir, exist_ok=True)
path = f"{out_dir}/{sampler}_{steps}_cfg{cfg_scale}.png"
save_image(samples, path, nrow=n_per_class, normalize=True, value_range=(-1, 1))
print(f"saved to {path}")
