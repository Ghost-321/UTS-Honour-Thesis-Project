import numpy as np, cv2
from .base import SRModel

class BicubicSR(SRModel):
    name = "bicubic"
    def module(self): return self  # no nn.Module

    def forward_tensor(self, lr_up):
        # passthrough: already bicubic-upsampled; mimic identity
        return lr_up.clamp(0,1)

    def enhance_np(self, img_float_hw3: np.ndarray):
        # direct bicubic upscale-back pipeline
        h,w = img_float_hw3.shape[:2]
        lr_small = cv2.resize(img_float_hw3, (w//self.scale, h//self.scale), interpolation=cv2.INTER_CUBIC)
        return cv2.resize(lr_small, (w, h), interpolation=cv2.INTER_CUBIC)
