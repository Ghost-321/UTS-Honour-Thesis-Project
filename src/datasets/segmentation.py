# src/datasets/segmentation.py
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from PIL import Image
import re

# Recognize lesion folder names → canonical keys
def _lesion_key_from_dirname(name: str) -> Optional[str]:
    n = name.lower()
    if "micro" in n and "aneur" in n: return "MA"
    if "haem" in n or "hemorr" in n:  return "HE"
    if "hard" in n and "exud" in n:   return "EX"
    if "soft" in n and "exud" in n:   return "SE"
    if "optic" in n and "disc" in n:  return "OD"
    if n in {"ma","he","ex","se","od"}: return n.upper()
    return None

_ALLOWED_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
_ID_RE = re.compile(r"^(IDRiD_\d+)", flags=re.I)

def _is_image_file(p: Path) -> bool:
    return p.suffix.lower() in _ALLOWED_EXTS

def _extract_base_id(stem: str) -> Optional[str]:
    """Return 'IDRiD_##' or 'IDRiD_###' from a filename stem, if present."""
    m = _ID_RE.match(stem)
    return m.group(1) if m else None

def list_mask_image_ids(masks_root: str) -> List[str]:
    """
    Collect unique base image IDs that have at least one mask anywhere under masks_root.
    Works with filenames like IDRiD_01_MA.tif, IDRiD_01_HE.tif, etc.
    """
    root = Path(masks_root)
    ids = set()
    if not root.exists(): return []
    for p in root.rglob("*"):
        if p.is_file() and _is_image_file(p):
            base = _extract_base_id(p.stem)
            if base:
                ids.add(base)
    return sorted(ids)

def _find_lesion_dirs(masks_root: str) -> Dict[str, List[Path]]:
    """
    Map lesion key -> list of directories that contain that lesion's masks.
    Searches recursively (handles a./b. Training/Testing).
    """
    root = Path(masks_root)
    out: Dict[str, List[Path]] = {"MA": [], "HE": [], "EX": [], "SE": [], "OD": []}
    if not root.exists(): return out
    for d in [p for p in root.rglob("*") if p.is_dir()]:
        key = _lesion_key_from_dirname(d.name)
        if key:
            out[key].append(d)
    return out

def _read_mask(p: Path) -> np.ndarray:
    m = np.array(Image.open(p).convert("L"))
    return (m > 0).astype(np.uint8)

def load_masks_for_id(masks_root: str, img_id: str) -> Dict[str, np.ndarray]:
    """
    Load available lesion masks for a given base img_id (e.g., 'IDRiD_01').
    Returns dict like {"MA": mask, "HE": mask, ...} for keys that exist.
    """
    dirs = _find_lesion_dirs(masks_root)
    out: Dict[str, np.ndarray] = {}
    for key, dlist in dirs.items():
        found = None
        for d in dlist:
            # search files starting with the base id (IDRiD_01_*.tif / .png / ...)
            for p in d.glob(f"{img_id}*"):
                if p.is_file() and _is_image_file(p):
                    found = p
                    break
            if found: break
        if found is not None:
            out[key] = _read_mask(found)
    return out

def union_mask(masks: Dict[str, np.ndarray], shape=None, include_od: bool = False) -> Optional[np.ndarray]:
    """
    Union MA/HE/EX/SE by default; include OD if include_od=True.
    If no masks and shape given, return a zero mask of that shape.
    """
    if not masks:
        if shape is None: return None
        H, W = (shape.shape[:2] if hasattr(shape, "shape") else shape[:2])
        return np.zeros((H, W), dtype=np.uint8)
    keys = ["MA", "HE", "EX", "SE"] + (["OD"] if include_od else [])
    acc = None
    for k in keys:
        if k in masks:
            m = masks[k].astype(np.uint8)
            acc = m if acc is None else (acc | m)
    if acc is None:
        if shape is None: return None
        H, W = (shape.shape[:2] if hasattr(shape, "shape") else shape[:2])
        return np.zeros((H, W), dtype=np.uint8)
    return acc
