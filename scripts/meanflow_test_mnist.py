import os
import torch
from torchvision.utils import save_image

from dit import DiT
from meanflow import MeanFlow

ckpt = "results/train_mnist_meanflow/model.pt"
ts = (1.0, 0.0)  # (1.0, 0.0) 1 step; (1.0, 0.5, 0.0) 2 steps
num_classes = 10

n_per_class = 10
seed = 0
out_dir = "results/test_mnist_meanflow"
device = "cuda"

torch.manual_seed(seed)

model = DiT(num_classes=num_classes, use_r=True).to(device)
model.load_state_dict(torch.load(ckpt, map_location=device))
model.eval()
mf = MeanFlow()

y = torch.arange(num_classes, device=device).repeat_interleave(n_per_class)
samples = mf.sample(model, (len(y), 1, 28, 28), cond=y, ts=ts, device=device)

os.makedirs(out_dir, exist_ok=True)
path = f"{out_dir}/{len(ts) - 1}step.png"
save_image(samples, path, nrow=n_per_class, normalize=True, value_range=(-1, 1))
print(f"saved to {path}")
