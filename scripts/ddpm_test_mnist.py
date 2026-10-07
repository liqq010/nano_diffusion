import os
import torch
from torchvision.utils import save_image

from dit import DiT
from ddpm import DDPM

pred = "eps"  # must match training: "eps" / "x0" / "v"
ckpt = f"results/train_mnist_ddpm_{pred}/model.pt"
sampler = "ddim"  # "ddpm" / "ddim" / "dpm" / "dpm2" / "unipc"
steps = 50  # number of steps for ddim / dpm / unipc; ddpm always uses 1000
cfg_scale = 3.0  # 1.0 = no CFG
num_classes = 10

n_per_class = 10
seed = 0
out_dir = "results/test_mnist_ddpm"
device = "cuda"

torch.manual_seed(seed)

model = DiT(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(ckpt, map_location=device))
model.eval()
ddpm = DDPM(pred=pred, device=device)

y = torch.arange(num_classes, device=device).repeat_interleave(n_per_class)
shape = (len(y), 1, 28, 28)
if sampler == "ddpm":
    samples = ddpm.sample(model, shape, cond=y, cfg_scale=cfg_scale, device=device)
elif sampler == "ddim":
    samples = ddpm.ddim_sample(model, shape, steps=steps, cond=y, cfg_scale=cfg_scale, device=device)
elif sampler == "dpm2":  # 2 network calls per step
    samples = ddpm.dpm_solver2_sample(model, shape, steps=steps, cond=y, cfg_scale=cfg_scale, device=device)
elif sampler == "dpm":
    samples = ddpm.dpm_solver_sample(model, shape, steps=steps, cond=y, cfg_scale=cfg_scale, device=device)
else:
    samples = ddpm.unipc_sample(model, shape, steps=steps, cond=y, cfg_scale=cfg_scale, device=device)

os.makedirs(out_dir, exist_ok=True)
path = f"{out_dir}/{pred}_{sampler}_{steps}_cfg{cfg_scale}.png"
save_image(samples, path, nrow=n_per_class, normalize=True, value_range=(-1, 1))
print(f"saved to {path}")
