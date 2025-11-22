import numpy as np
from skimage.metrics import peak_signal_noise_ratio as _psnr
from skimage.metrics import structural_similarity as _ssim

def _to_uint8(img):
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return img

def psnr(sr, hr):
    sr8, hr8 = _to_uint8(sr), _to_uint8(hr)
    return float(_psnr(hr8, sr8, data_range=255))

def ssim(sr, hr):
    sr8, hr8 = _to_uint8(sr), _to_uint8(hr)
    if sr8.ndim == 3 and sr8.shape[2] == 3:
        return float(_ssim(hr8, sr8, channel_axis=2, data_range=255))
    else:
        return float(_ssim(hr8, sr8, data_range=255))

def masked_metric(func, sr, hr, mask):
    m = mask.astype(bool)
    if m.sum() == 0:
        return float('nan')
    ys, xs = np.where(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return func(sr[y0:y1, x0:x1], hr[y0:y1, x0:x1])

def lpips_metric(sr, hr):
    try:
        import lpips, torch
        loss_fn = lpips.LPIPS(net='vgg')
        def to_t(img):
            if img.dtype != np.float32:
                img = img.astype(np.float32) / 255.0
            t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
            return (t * 2 - 1)
        d = loss_fn(to_t(sr), to_t(hr))
        return float(d.item())
    except Exception:
        return float('nan')

def niqe_metric(img):
    try:
        import piq, torch
        if img.dtype != np.float32:
            img = img.astype(np.float32) / 255.0
        # convert to grayscale (H, W)
        g = np.dot(img[..., :3], [0.299, 0.587, 0.114])
        t = torch.from_numpy(g).unsqueeze(0).unsqueeze(0).float()  # (1,1,H,W), float32
        return float(piq.niqe(t).item())
    except Exception:
        return float('nan')
