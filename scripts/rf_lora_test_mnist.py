import os
import sys

import torch
from torchvision.utils import save_image

from dit import DiT
from rf import RectifiedFlow
from lora import apply_lora, set_lora_scale

# python scripts/rf_lora_test_mnist.py lora_r4 [adaln] -> LoRA strength 0 / 0.5 / 1; python scripts/rf_lora_test_mnist.py ft -> full fine-tuning
run = sys.argv[1] if len(sys.argv) > 1 else "lora_r4"
adaln = "adaln" in sys.argv[2:]
name = run + ("_adaln" if adaln else "")
base_ckpt = "results/train_mnist_rf/model.pt"
steps = 50
num_classes = 10

n_per_class = 10
seed = 0
out_dir = f"results/test_mnist_rf_{name}"
device = "cuda"

model = DiT(num_classes=num_classes).to(device)
if run == "ft":
    model.load_state_dict(torch.load(f"results/train_mnist_rf_ft/model.pt", map_location=device))
    scales = [None]
else:
    model.load_state_dict(torch.load(base_ckpt, map_location=device))
    targets = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2") + (("adaLN_modulation.1",) if adaln else ())
    apply_lora(model, r=int(run.split("_r")[1]), targets=targets)
    model.load_state_dict(torch.load(f"results/train_mnist_rf_{name}/lora.pt", map_location=device), strict=False)
    scales = [0.0, 0.5, 1.0]
model.eval()
rf = RectifiedFlow()

y = torch.arange(num_classes, device=device).repeat_interleave(n_per_class)
os.makedirs(out_dir, exist_ok=True)
for scale in scales:
    if scale is not None:
        set_lora_scale(model, scale)
    for cfg_scale in [1.0, 3.0]:
        torch.manual_seed(seed)
        samples = rf.sample(model, (len(y), 1, 28, 28), cond=y, steps=steps, cfg_scale=cfg_scale, device=device)
        path = f"{out_dir}/euler_{steps}_cfg{cfg_scale}" + ("" if scale is None else f"_scale{scale}") + ".png"
        save_image(samples, path, nrow=n_per_class, normalize=True, value_range=(-1, 1))
        print(f"saved to {path}")
