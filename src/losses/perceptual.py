import torch

class PerceptualLPIPS:
    """Thin wrapper around lpips with safe fallback."""
    def __init__(self, net: str = "vgg", device: str = "cuda"):
        try:
            import lpips
            self.fn = lpips.LPIPS(net=net).to(device)
            self.enabled = True
        except Exception:
            self.fn = None
            self.enabled = False
            print("[WARN] lpips not installed; PerceptualLPIPS will return 0.0")

    def __call__(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return torch.tensor(0.0, device=sr.device)
        # expects [-1,1]
        def norm(x): return x*2-1
        return self.fn(norm(sr), norm(hr)).mean()
