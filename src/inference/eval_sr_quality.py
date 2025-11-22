# src/eval_sr_quality.py
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
import cv2

from src.config.loader import load_cfg
from src.models.sr import build_sr
from src.utils.paths import list_images, ensure_dir
from src.datasets.segmentation import load_masks_for_id, union_mask
from src.metrics.sr_metrics import psnr, ssim, masked_metric, lpips_metric, niqe_metric


def _load_image(path):
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _down_up(img, scale=2, blur=False, noise_std=0.0):
    """Downscale then upsample (bicubic), with optional blur/noise."""
    h, w = img.shape[:2]
    lr = cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)
    lr_up = cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC)
    if blur:
        lr_up = cv2.GaussianBlur(lr_up, (3, 3), 0)
    if noise_std > 0:
        lr_up = np.clip(lr_up + np.random.normal(0, noise_std, lr_up.shape), 0, 1)
    return lr_up


def _get_core_module(model):
    core = getattr(model, "module", None)
    if callable(core):
        core = core()
    if core is None:
        core = model
    return core


@torch.no_grad()
def main(args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--out_dir", type=str, default="outputs/metrics")
    ap.add_argument("--scale", type=int, default=2, help="Down/up scale factor")
    ap.add_argument("--blur", action="store_true", help="Apply Gaussian blur in degrade")
    ap.add_argument("--noise", type=float, default=0.0, help="Std of Gaussian noise (0..1)")
    a = ap.parse_args(args if args is not None else None)

    cfg = load_cfg(a.config)
    device = getattr(cfg.run, "device", "cuda" if torch.cuda.is_available() else "cpu")

    seg_imgs = list_images(getattr(cfg.paths, "segmentation_images"))
    out_dir = Path(a.out_dir)
    ensure_dir(out_dir)

    for model_name in getattr(cfg.sr, "models", ["bicubic", "srcnn", "fsrcnn", "espcn"]):
        rows = []

        model = None
        core = None
        if model_name != "bicubic":
            try:
                model = build_sr(model_name, device=device, cfg=cfg).to(device)
                core = _get_core_module(model)

                ckpt_path = Path("outputs/sr_models") / f"{model_name}_best.pt"
                if ckpt_path.exists():
                    state = torch.load(ckpt_path, map_location=device)
                    state_dict = state.get("model", state) if isinstance(state, dict) else state
                    core.load_state_dict(state_dict, strict=False)
                    print(f"[INFO] Loaded weights for {model_name} from {ckpt_path}")
                else:
                    print(f"[WARN] No checkpoint found for {model_name}; evaluating fresh init.")
                model.eval_mode()
            except Exception as e:
                print(f"[WARN] Skipping {model_name}: {e}")
                continue

        for p in tqdm(seg_imgs, desc=f"Eval {model_name}"):
            img = _load_image(p)
            lr_up = _down_up(img, scale=a.scale, blur=a.blur, noise_std=a.noise)

            if model_name == "bicubic":
                sr_eval = lr_up
            else:
                x = torch.from_numpy(lr_up.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
                y = model.forward_tensor(x).clamp(0, 1).cpu().squeeze(0).numpy().transpose(1, 2, 0)
                sr_eval = y

            met = {
                "model": model_name,
                "img_id": Path(p).stem,
                "psnr": psnr(sr_eval, img),
                "ssim": ssim(sr_eval, img),
                "lpips": lpips_metric(sr_eval, img),
                "niqe": niqe_metric(sr_eval),
            }

            try:
                masks = load_masks_for_id(getattr(cfg.paths, "masks_root"), Path(p).stem)
                u = union_mask(masks, shape=img)
                if u is not None:
                    met["psnr_lesion"] = masked_metric(psnr, sr_eval, img, u)
                    met["ssim_lesion"] = masked_metric(ssim, sr_eval, img, u)
            except Exception:
                pass

            rows.append(met)

        df = pd.DataFrame(rows)
        csv_path = out_dir / f"sr_eval_{model_name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved metrics → {csv_path}")


if __name__ == "__main__":
    main()
