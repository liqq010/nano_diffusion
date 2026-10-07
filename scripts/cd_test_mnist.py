import os
import torch
from torchvision.utils import save_image

from dit import DiT
from cm import ConsistencyModel

ckpt = "results/train_mnist_cd/model.pt"
ts = (1.0,)  # (1.0,) 1 step; (1.0, 0.5) 2 steps; (1.0, 0.75, 0.5, 0.25) 4 steps
num_classes = 10

n_per_class = 10
seed = 0
out_dir = "results/test_mnist_cd"
device = "cuda"

torch.manual_seed(seed)

model = DiT(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(ckpt, map_location=device))
model.eval()
cm = ConsistencyModel()

y = torch.arange(num_classes, device=device).repeat_interleave(n_per_class)
samples = cm.sample(model, (len(y), 1, 28, 28), cond=y, ts=ts, device=device)

os.makedirs(out_dir, exist_ok=True)
path = f"{out_dir}/{len(ts)}step.png"
save_image(samples, path, nrow=n_per_class, normalize=True, value_range=(-1, 1))
print(f"saved to {path}")
