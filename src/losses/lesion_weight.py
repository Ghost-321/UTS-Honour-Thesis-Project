import torch
from .reconstruction import l1, ssim_loss

def lesion_weighted_l1(sr, hr, mask, alpha=0.7):
    """
    sr/hr: (N,3,H,W) in [0,1]
    mask:  (N,1,H,W) in {0,1}
    Emphasise pixels inside lesion mask by (1 + alpha).
    """
    w = 1.0 + alpha * mask
    return (w * torch.abs(sr - hr)).mean()

def total_loss(sr, hr, mask, alpha=0.7, lambda_ssim=0.1):
    """
    Returns:
        loss: scalar tensor
        logs: dict with human-friendly metrics
              - 'lw_l1': lesion-weighted L1 (lower is better)
              - 'ssim_score': SSIM in [0,1] (higher is better)
    Note: we still optimise with SSIM loss (1 - SSIM) internally.
    """
    lw = lesion_weighted_l1(sr, hr, mask, alpha=alpha)
    ssim_l = ssim_loss(sr, hr)          # = 1 - SSIM
    ssim_score = 1.0 - ssim_l           # readable score

    loss = lw + lambda_ssim * ssim_l
    return loss, {"lw_l1": float(lw.item()), "ssim_score": float(ssim_score.item())}
