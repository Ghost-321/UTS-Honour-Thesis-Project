import torch, torch.nn.functional as F

def l1(sr, hr): return torch.abs(sr-hr).mean()

def _ssim(img1, img2, C1=0.01**2, C2=0.03**2):
    # simple, fast SSIM approx on tensors in [0,1]
    mu1 = F.avg_pool2d(img1, 3, 1, 1)
    mu2 = F.avg_pool2d(img2, 3, 1, 1)
    sigma1 = F.avg_pool2d(img1*img1, 3,1,1) - mu1*mu1
    sigma2 = F.avg_pool2d(img2*img2, 3,1,1) - mu2*mu2
    sigma12= F.avg_pool2d(img1*img2, 3,1,1) - mu1*mu2
    num = (2*mu1*mu2 + C1) * (2*sigma12 + C2)
    den = (mu1*mu1 + mu2*mu2 + C1) * (sigma1 + sigma2 + C2)
    ssim_map = num / (den + 1e-8)
    return ssim_map.mean()

def ssim_loss(sr, hr): return 1.0 - _ssim(sr, hr)
