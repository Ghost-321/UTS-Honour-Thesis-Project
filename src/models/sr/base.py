from abc import ABC, abstractmethod
import torch, torch.nn.functional as F
import numpy as np, cv2

class SRModel(ABC):
    name: str = "base"
    scale: int = 2

    def __init__(self, device="cuda"):
        self.device = device

    @abstractmethod
    def module(self): ...

    def to(self, device):
        self.device = device
        self.module().to(device)
        return self

    def train_mode(self): self.module().train()
    def eval_mode(self):  self.module().eval()

    @abstractmethod
    def forward_tensor(self, lr_up: torch.Tensor) -> torch.Tensor:
        """lr_up: (N,3,H,W) in [0,1] -> sr (N,3,H,W) in [0,1]"""
        ...

    @torch.no_grad()
    def enhance_np(self, img_float_hw3: np.ndarray):
        h,w = img_float_hw3.shape[:2]
        # build LR then upscale to original size (self.scale=2)
        lr_small = cv2.resize(img_float_hw3, (w//self.scale, h//self.scale), interpolation=cv2.INTER_CUBIC)
        lr_up    = cv2.resize(lr_small, (w, h), interpolation=cv2.INTER_CUBIC)
        x = torch.from_numpy(lr_up.transpose(2,0,1)).unsqueeze(0).float().to(self.device)
        self.eval_mode()
        y = self.forward_tensor(x).clamp(0,1).cpu().squeeze(0).numpy().transpose(1,2,0)
        return y
