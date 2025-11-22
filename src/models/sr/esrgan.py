from .base import SRModel
import numpy as np

class ESRGANx2(SRModel):
    name = "esrgan"
    def __init__(self, device="cuda", tile=0, tile_pad=10, pre_pad=0):
        super().__init__(device)
        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
        except Exception as e:
            raise RuntimeError("Install 'realesrgan' and 'basicsr' to use ESRGAN.") from e
        self.net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        self.upsampler = RealESRGANer(
            scale=2, model_path=None, model=self.net, tile=tile, tile_pad=tile_pad, pre_pad=pre_pad,
            half=True if device=="cuda" else False
        )

    def module(self): return self
    def forward_tensor(self, lr_up):
        import torch
        arr = lr_up.detach().clamp(0,1).cpu().numpy()
        arr = (arr * 255).astype(np.uint8)
        outs = []
        for i in range(arr.shape[0]):
            img = np.transpose(arr[i], (1,2,0))
            sr, _ = self.upsampler.enhance(img, outscale=1)
            outs.append(np.transpose(sr.astype(np.float32)/255.0, (2,0,1)))
        out = torch.from_numpy(np.stack(outs,0)).to(lr_up.device)
        return out
