# src/utils/paths.py
from pathlib import Path
from typing import Iterable, List, Union

# default set of image extensions (case-insensitive)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

def list_images(root: Union[str, Path], recursive: bool = True, exts: Iterable[str] = None) -> List[str]:
    """
    Return a sorted list of image file paths under `root`.
    - recursive=True walks subfolders (needed for IDRiD a./b. splits)
    - exts: custom iterable of extensions; defaults to IMG_EXTS
    Returns: list[str] of absolute/relative paths (as strings)
    """
    r = Path(root)
    if not r.exists():
        return []
    allow = set(e.lower() for e in (exts or IMG_EXTS))
    it = r.rglob("*") if recursive else r.glob("*")
    files = [str(p) for p in it if p.is_file() and p.suffix.lower() in allow]
    files.sort()
    return files

def list_images_all(roots: Iterable[Union[str, Path]], recursive: bool = True, exts: Iterable[str] = None) -> List[str]:
    """List images from multiple roots, de-duplicated and sorted."""
    out: List[str] = []
    for root in (roots or []):
        if root:
            out.extend(list_images(root, recursive=recursive, exts=exts))
    return sorted(set(out))

def ensure_dir(path: Union[str, Path]):
    Path(path).mkdir(parents=True, exist_ok=True)
