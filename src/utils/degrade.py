# src/utils/degrade.py
import torch
import torch.nn.functional as F

def _gaussian_kernel2d(ksize: int = 3, sigma: float = 1.0, device="cpu", dtype=torch.float32):
    ax = torch.arange(ksize, device=device, dtype=dtype) - (ksize - 1) / 2
    xx, yy = torch.meshgrid(ax, ax, indexing="xy")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel

def gaussian_blur_torch(img_bchw: torch.Tensor, ksize: int = 3, sigma: float = 1.0):
    """
    img_bchw: [B,C,H,W] in [0,1]
    """
    if ksize <= 1 or sigma <= 0:
        return img_bchw
    B, C, H, W = img_bchw.shape
    k = _gaussian_kernel2d(ksize, sigma, device=img_bchw.device, dtype=img_bchw.dtype)
    k = k.expand(C, 1, ksize, ksize)
    return torch.conv2d(img_bchw, k, padding=ksize // 2, groups=C)

def bicubic_down_up_torch(hr_bchw: torch.Tensor, scale: int = 3,
                          blur_ksize: int = 0, blur_sigma: float = 0.0,
                          noise_std: float = 0.0):
    """
    hr_bchw: [B,C,H,W] in [0,1]
    Returns lr_up: bicubic down→up, same size as HR, with optional blur/noise.
    """
    B, C, H, W = hr_bchw.shape
    h2, w2 = max(1, H // scale), max(1, W // scale)
    lr = F.interpolate(hr_bchw, size=(h2, w2), mode="bicubic", align_corners=False)
    if blur_ksize and blur_sigma > 0:
        lr = gaussian_blur_torch(lr, blur_ksize, blur_sigma)
    if noise_std > 0:
        lr = (lr + noise_std * torch.randn_like(lr)).clamp(0, 1)
    lr_up = F.interpolate(lr, size=(H, W), mode="bicubic", align_corners=False)
    return lr_up

def rgb_to_y(bchw: torch.Tensor):
    """
    Approx BT.601 luma on [0,1] RGB.
    bchw: [B,3,H,W] -> [B,1,H,W]
    """
    r, g, b = bchw[:, 0:1], bchw[:, 1:2], bchw[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y
