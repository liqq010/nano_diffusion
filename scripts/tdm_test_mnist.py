import os
import sys

import torch
from torchvision.utils import save_image

from dit import DiT
from tdm import TDM

num_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 2
ckpt = f"results/train_mnist_tdm_{num_steps}step/model.pt"
num_classes = 10

n_per_class = 10
seed = 0
out_dir = f"results/test_mnist_tdm_{num_steps}step"
device = "cuda"

torch.manual_seed(seed)

model = DiT(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(ckpt, map_location=device))
model.eval()
tdm = TDM(num_steps=num_steps)

y = torch.arange(num_classes, device=device).repeat_interleave(n_per_class)
samples = tdm.sample(model, (len(y), 1, 28, 28), cond=y, device=device)

os.makedirs(out_dir, exist_ok=True)
path = f"{out_dir}/{num_steps}step.png"
save_image(samples, path, nrow=n_per_class, normalize=True, value_range=(-1, 1))
print(f"saved to {path}")
