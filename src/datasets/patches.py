from pathlib import Path
import numpy as np, cv2, random
from PIL import Image
import torch
from torch.utils.data import Dataset

from .segmentation import load_masks_for_id, union_mask

def load_img(path):
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32)/255.0

class LesionPatchDataset(Dataset):
    """
    Returns (lr_up, hr, lesion_mask) for 2x SR training.
    """
    def __init__(self, image_paths, masks_root, hr_size=256, stride=128, lesion_frac=0.5):
        self.image_paths = list(map(str, image_paths))
        self.masks_root = masks_root
        self.hr = hr_size
        self.records = []

        for p in self.image_paths:
            img_id = Path(p).stem
            masks = load_masks_for_id(masks_root, img_id)
            # skip if no masks
            if not masks: continue
            img = load_img(p)
            H,W = img.shape[:2]
            m = union_mask(masks, shape=img.shape)
            m = cv2.resize(m, (W,H), interpolation=cv2.INTER_NEAREST)

            ys = list(range(0, H-self.hr+1, stride))
            xs = list(range(0, W-self.hr+1, stride))
            for y in ys:
                for x in xs:
                    has_lesion = m[y:y+self.hr, x:x+self.hr].sum() > 0
                    self.records.append((p, x, y, has_lesion))

        # reorder to emphasize lesion patches roughly at lesion_frac
        lesion = [r for r in self.records if r[3]]
        bg     = [r for r in self.records if not r[3]]
        k = int(len(self.records)*lesion_frac)
        mix = lesion[:k] + bg[:len(self.records)-k]
        random.shuffle(mix)
        self.records = mix

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        p, x, y, has_lesion = self.records[idx]
        img = load_img(p)
        hr = img[y:y+self.hr, x:x+self.hr, :]
        # 2x pipeline: downscale then bicubic up to HR size
        lr_small = cv2.resize(hr, (self.hr//2, self.hr//2), interpolation=cv2.INTER_CUBIC)
        lr_up    = cv2.resize(lr_small, (self.hr, self.hr), interpolation=cv2.INTER_CUBIC)

        # lesion crop
        img_id = Path(p).stem
        masks = load_masks_for_id(self.masks_root, img_id)
        m = union_mask(masks, shape=img.shape)
        m_patch = m[y:y+self.hr, x:x+self.hr]
        # to tensors
        hr_t = torch.from_numpy(hr.transpose(2,0,1)).float()
        lr_t = torch.from_numpy(lr_up.transpose(2,0,1)).float()
        mask_t = torch.from_numpy(m_patch[None,:,:].astype(np.float32))
        return lr_t, hr_t, mask_t
