from .base import SRModel
import torch

class SwinIRx2(SRModel):
    name = "swinir"
    def __init__(self, device="cuda", ckpt=""):
        super().__init__(device)
        try:
            from basicsr.archs.swinir_arch import SwinIR
        except Exception as e:
            raise RuntimeError("Install 'basicsr' to use SwinIR.") from e
        if not ckpt:
            raise RuntimeError("SwinIR checkpoint path is empty. Set paths.swinir_ckpt in config.yaml.")
        self._m = SwinIR(
            upscale=2, in_chans=3, img_size=64, window_size=8,
            img_range=1.0, depths=[6,6,6,6], embed_dim=60, num_heads=[6,6,6,6],
            mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv'
        ).to(device)
        sd = torch.load(ckpt, map_location=device)
        key = 'params_ema' if 'params_ema' in sd else 'state_dict' if 'state_dict' in sd else None
        self._m.load_state_dict(sd[key] if key else sd, strict=False)
        self._m.eval()

    def module(self): return self._m
    @torch.no_grad()
    def forward_tensor(self, lr_up):
        return self._m(lr_up).clamp(0,1)
