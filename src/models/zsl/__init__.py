from .clip_zeroshot import CLIPZeroShot


# --- Optional Safe Imports ---
def safe_import_medclip():
    try:
        from .medclip_zeroshot import MedCLIPZeroShot
        return MedCLIPZeroShot
    except Exception:
        return None


def safe_import_retclip():
    try:
        from .retclip_zeroshot import RETCLIPZeroShot
        return RETCLIPZeroShot
    except Exception:
        return None


# --- ZSL Builder ---
def build_zsl(name: str, device="cuda"):
    """
    Factory function to construct a Zero-Shot Learning (ZSL) model.
    Supports CLIP, MedCLIP, and RETCLIP backbones.

    Args:
        name (str): backbone name ("clip", "medclip", "retclip")
        device (str): "cuda" or "cpu"
    """
    n = name.lower()

    if n == "clip":
        return CLIPZeroShot(device)

    elif n == "medclip":
        M = safe_import_medclip()
        if M is None:
            raise RuntimeError("MedCLIP not available. Please install 'medclip'.")
        return M(device)

    elif n == "retclip":
        R = safe_import_retclip()
        if R is None:
            raise RuntimeError("RETCLIP not available. Please install 'open_clip_torch'.")
        return R(device)

    else:
        raise ValueError(f"Unknown ZSL backbone: {name}")
