import os
import torch
from torchvision.utils import save_image

from dit import DiT
from dmd import DMD2

ckpt = "results/train_mnist_dmd2/model.pt"
num_classes = 10

n_per_class = 10
seed = 0
out_dir = "results/test_mnist_dmd2"
device = "cuda"

torch.manual_seed(seed)

model = DiT(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(ckpt, map_location=device))
model.eval()
dmd = DMD2()

y = torch.arange(num_classes, device=device).repeat_interleave(n_per_class)
samples = dmd.sample(model, (len(y), 1, 28, 28), cond=y, device=device)

os.makedirs(out_dir, exist_ok=True)
path = f"{out_dir}/1step.png"
save_image(samples, path, nrow=n_per_class, normalize=True, value_range=(-1, 1))
print(f"saved to {path}")
