"""
Create outlined MNIST (for LoRA fine-tuning)
"""
import io
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import save_image

src = "data/mnist/mnist/train-00000-of-00001.parquet"
out_dir = "data/mnist_outlined"
n_per_class = 100
seed = 0

df = pd.read_parquet(src)
rng = np.random.default_rng(seed)
idx = np.sort(np.concatenate([rng.choice(np.where(df["label"].to_numpy() == c)[0], n_per_class, replace=False) for c in range(10)]))
sub = df.iloc[idx]

x = torch.from_numpy(np.stack([np.array(Image.open(io.BytesIO(b["bytes"]))) for b in sub["image"]])).float().unsqueeze(1)  # (N,1,28,28), [0,255]
outlined = (F.max_pool2d(x, 3, 1, 1) - x).clamp(0, 255).round().to(torch.uint8)


def to_png(a):
    buf = io.BytesIO()
    Image.fromarray(a.squeeze(0).numpy()).save(buf, format="PNG")
    return buf.getvalue()


os.makedirs(out_dir, exist_ok=True)
pd.DataFrame({
    "image": [{"bytes": to_png(a), "path": None} for a in outlined],
    "label": sub["label"].to_numpy(),
    "orig_index": idx,
}).to_parquet(f"{out_dir}/train-00000-of-00001.parquet", index=False)

show = np.concatenate([np.where(sub["label"].to_numpy() == c)[0][:10] for c in range(10)])
pairs = torch.stack([torch.cat([x[i:i + 1] / 255, outlined[i:i + 1].float() / 255]) for i in show]).view(-1, 1, 28, 28)
save_image(pairs, f"{out_dir}/preview.png", nrow=20, padding=2, pad_value=0.5)
print(f"saved {len(sub)} images to {out_dir}, per-class counts: {np.bincount(sub['label'].to_numpy()).tolist()}")
