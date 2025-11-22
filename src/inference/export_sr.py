# src/inference/export_sr.py
import argparse, time, csv
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
import torch

from src.config.loader import load_cfg
from src.models.sr import build_sr
from src.utils.paths import list_images, ensure_dir


def _get_core_module(model):
    core = getattr(model, "module", None)
    if callable(core):  # some wrappers expose module() as a getter
        core = core()
    if core is None and hasattr(model, "_m"):  # your SR wrappers often keep inner nn.Module here
        core = model._m
    return core if core is not None else model


def _load_rgb(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _save_rgb(path: Path, arr: np.ndarray):
    arr8 = (np.clip(arr, 0, 1) * 255.0).astype(np.uint8)
    Image.fromarray(arr8).save(path)


def _degrade_bicubic(img: np.ndarray, scale: int, blur=False, noise_std=0.0) -> np.ndarray:
    """Downscale by `scale`, then bicubic up to original size. Optional blur/noise."""
    h, w = img.shape[:2]
    lr = cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)
    lr_up = cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC)
    if blur:
        lr_up = cv2.GaussianBlur(lr_up, (3, 3), 0)
    if noise_std > 0:
        lr_up = np.clip(lr_up + np.random.normal(0, noise_std, lr_up.shape), 0, 1)
    return lr_up


@torch.no_grad()
def _sr_enhance_np(model, img_np: np.ndarray, device: str = "cuda") -> np.ndarray:
    x = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).to(device).float()
    y = model.forward_tensor(x).clamp(0, 1).cpu().squeeze(0).numpy().transpose(1, 2, 0)
    return y


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--model", type=str, required=True,
                        help="bicubic|srcnn|fsrcnn|espcn|esrgan|swinir")
    parser.add_argument("--images", type=str, required=True, help="Folder of input images")
    parser.add_argument("--out", type=str, default="outputs/sr_images", help="Output root")
    parser.add_argument("--ckpt", type=str, default="", help="Path to trained checkpoint (*.pt) for the model")
    parser.add_argument("--skip_existing", action="store_true", help="Skip images already exported")
    parser.add_argument("--degrade_scale", type=int, default=2, help="Down/up scale factor for test (2/3/4)")
    parser.add_argument("--degrade_blur", action="store_true", help="Apply Gaussian blur to degraded image")
    parser.add_argument("--degrade_noise", type=float, default=0.0, help="Std of Gaussian noise to add (0..1)")
    parser.add_argument("--copy_gt", action="store_true", help="Also copy originals into out/original")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N images (debug)")
    a = parser.parse_args(args if args is not None else None)   # ✅ fixed

    cfg = load_cfg(a.config)
    device = getattr(cfg.run, "device", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[export_sr] device={device}")

    # Build & (optionally) load weights
    model = build_sr(a.model, device=device, cfg=cfg)
    if a.model.lower() != "bicubic" and a.ckpt:
        state = torch.load(a.ckpt, map_location=device)
        state_dict = state.get("model", state) if isinstance(state, dict) else state
        core = _get_core_module(model)
        missing_unexp = core.load_state_dict(state_dict, strict=False)
        if hasattr(missing_unexp, "missing_keys") and hasattr(missing_unexp, "unexpected_keys"):
            mk, uk = missing_unexp.missing_keys, missing_unexp.unexpected_keys
            if mk or uk:
                print(f"[export_sr] non-strict load → missing={len(mk)} unexpected={len(uk)}")
    elif a.model.lower() != "bicubic" and not a.ckpt:
        print("[export_sr] WARNING: no --ckpt given; using randomly initialized weights.")

    # IO
    in_dir = Path(a.images)
    imgs = list_images(str(in_dir))
    if not imgs:
        raise FileNotFoundError(f"No images found in {in_dir}")
    if a.limit:
        imgs = imgs[:a.limit]

    out_root = Path(a.out)
    out_bic = out_root / f"bicubic_x{a.degrade_scale}"
    out_sr  = out_root / f"{a.model}_x{a.degrade_scale}"
    out_gt  = out_root / "original" if a.copy_gt else None
    ensure_dir(out_bic); ensure_dir(out_sr)
    if out_gt is not None:
        ensure_dir(out_gt)

    # Timings CSV
    metrics_dir = Path("outputs/metrics"); ensure_dir(metrics_dir)
    time_csv = metrics_dir / f"export_times_{a.model}_x{a.degrade_scale}.csv"
    with open(time_csv, "w", newline="", encoding="utf-8") as ft:
        t_writer = csv.DictWriter(ft, fieldnames=["model","scale","image","seconds","skipped"])
        t_writer.writeheader()

        manifest_csv = out_root / f"manifest_{a.model}_x{a.degrade_scale}.csv"
        mf = open(manifest_csv, "w", newline="", encoding="utf-8")
        m_writer = csv.writer(mf)
        m_writer.writerow(["orig_path","bicubic_path","sr_path"])

        timings = []
        pbar = tqdm(imgs, desc=f"[Export:{a.model} x{a.degrade_scale}]")
        for p in pbar:
            name = Path(p).name
            out_bic_p = out_bic / name
            out_sr_p  = out_sr / name
            out_gt_p  = (out_gt / name) if out_gt is not None else None

            if a.skip_existing and out_bic_p.exists() and out_sr_p.exists():
                t_writer.writerow({"model": a.model, "scale": a.degrade_scale,
                                   "image": str(p), "seconds": "0.0000", "skipped": True})
                m_writer.writerow([str(Path(p).resolve()), str(out_bic_p.resolve()), str(out_sr_p.resolve())])
                continue

            img = _load_rgb(p)

            # 1) degrade
            deg = _degrade_bicubic(img, a.degrade_scale, blur=a.degrade_blur, noise_std=a.degrade_noise)
            _save_rgb(out_bic_p, deg)

            # 2) SR
            t0 = time.perf_counter()
            sr = deg if a.model.lower() == "bicubic" else _sr_enhance_np(model, deg, device=device)
            dt = time.perf_counter() - t0

            _save_rgb(out_sr_p, sr)
            if out_gt_p is not None and not out_gt_p.exists():
                _save_rgb(out_gt_p, img)

            timings.append(dt)
            avg = sum(timings) / len(timings)
            eta = avg * (len(imgs) - len(timings))
            pbar.set_postfix({"last_s": f"{dt:.2f}", "avg_s": f"{avg:.2f}", "eta_m": f"{eta/60:.1f}"})

            t_writer.writerow({"model": a.model, "scale": a.degrade_scale,
                               "image": str(p), "seconds": f"{dt:.4f}", "skipped": False})
            m_writer.writerow([str(Path(p).resolve()), str(out_bic_p.resolve()), str(out_sr_p.resolve())])

        mf.close()

    print(f"Export finished → {out_sr}")
    print(f"Timings saved → {time_csv}")
    print(f"Manifest saved → {manifest_csv}")
