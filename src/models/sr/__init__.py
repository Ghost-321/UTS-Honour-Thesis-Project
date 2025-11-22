from .bicubic import BicubicSR
from .srcnn import SRCNN
from .fsrcnn import FSRCNN
from .espcn import ESPCN

# Optional heavy models (guarded)
def safe_import_esrgan():
    try:
        from .esrgan import ESRGANx2
        return ESRGANx2
    except Exception:
        return None

def safe_import_swinir():
    try:
        from .swinir import SwinIRx2
        return SwinIRx2
    except Exception:
        return None

def build_sr(name: str, device="cuda", cfg=None):
    n = name.lower()
    if n == "bicubic": return BicubicSR(device)
    if n == "srcnn":   return SRCNN(device)
    if n == "fsrcnn":  return FSRCNN(device)
    if n == "espcn":   return ESPCN(device)
    if n == "esrgan":
        ESR = safe_import_esrgan()
        if ESR is None:
            raise RuntimeError("ESRGAN not available. Install 'realesrgan' & 'basicsr'.")
        es = (cfg.sr.get("esrgan",{}) if cfg else {})
        return ESR(device=device, tile=es.get("tile",0), tile_pad=es.get("tile_pad",10), pre_pad=es.get("pre_pad",0))
    if n == "swinir":
        SW = safe_import_swinir()
        if SW is None:
            raise RuntimeError("SwinIR not available. Install 'basicsr' and provide paths.swinir_ckpt.")
        ckpt = (cfg.paths.get("swinir_ckpt") or "").strip()
        if not ckpt:
            raise RuntimeError("paths.swinir_ckpt is empty; set path to a x2 SwinIR checkpoint.")
        return SW(device=device, ckpt=ckpt)
    raise ValueError(f"Unknown SR model: {name}")
