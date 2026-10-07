import io
import glob
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class MNISTDataset(Dataset):
    """MNIST train + test (70k) loaded from HF parquet files, images scaled to [-1, 1]."""

    def __init__(self, root="data/mnist/mnist"):
        df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{root}/*.parquet"))])
        imgs = np.stack([np.array(Image.open(io.BytesIO(b["bytes"]))) for b in df["image"]])
        self.images = torch.from_numpy(imgs).float().div(127.5).sub(1).unsqueeze(1)  # (N, 1, 28, 28)
        self.labels = torch.from_numpy(df["label"].to_numpy())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
