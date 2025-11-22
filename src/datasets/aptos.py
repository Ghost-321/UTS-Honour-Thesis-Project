from pathlib import Path
from typing import Optional, Callable
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset

APTOS_CLASS_NAMES = ["NoDR", "Mild", "Moderate", "Severe", "Proliferative"]

class APTOSDataset(Dataset):
    """APTOS 2019 dataset (id_code, diagnosis) loader."""
    def __init__(self, csv_path: str, img_dir: str, transform=None,
                 sr_preprocess: Optional[Callable[[Image.Image], Image.Image]] = None):
        self.csv = pd.read_csv(csv_path)
        if "id_code" not in self.csv.columns or "diagnosis" not in self.csv.columns:
            raise ValueError("CSV must have 'id_code' and 'diagnosis' columns")

        self.img_dir = Path(img_dir)
        self.transform = transform
        self.sr_preprocess = sr_preprocess

        self.paths = [self.img_dir / f"{row.id_code}.png" for _, row in self.csv.iterrows()]
        self.labels = self.csv["diagnosis"].astype(int).tolist()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        y = self.labels[idx]
        img = Image.open(path).convert("RGB")

        if self.sr_preprocess:
            img = self.sr_preprocess(img)
        if self.transform:
            img = self.transform(img)

        return img, y
