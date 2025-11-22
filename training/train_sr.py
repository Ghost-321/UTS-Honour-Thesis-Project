# src/training/train_sr.py
import argparse, random, multiprocessing, time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F  # noqa: F401 (reserved if you add extra losses)
from torch.amp import autocast, GradScaler   # ← modern AMP API
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.config.loader import load_cfg
from src.utils.paths import list_images, ensure_dir
from src.utils.seed import set_seed
from src.utils.loggers import CSVLogger
from src.datasets.patches import LesionPatchDataset
from src.datasets.segmentation import list_mask_image_ids
from src.losses.lesion_weight import total_loss
from src.models.sr import build_sr  # trainable: srcnn, fsrcnn, espcn

SUPPORTED = {"srcnn", "fsrcnn", "espcn"}  # trainable here


def _get_core_module(model):
    """
    Return the inner nn.Module (handles wrappers exposing .module() or ._m).
    """
    core = getattr(model, "module", None)
    if callable(core):
        core = core()
    if core is None and hasattr(model, "_m"):
        core = model._m
    return core if core is not None else model


@torch.no_grad()
def _val_metrics(model, loader, device, amp: bool) -> Tuple[float, float, float]:
    """
    Compute average validation loss, PSNR, SSIM over the validation loader.
    """
    from skimage.metrics import peak_signal_noise_ratio as psnr_metric
    from skimage.metrics import structural_similarity as ssim_metric

    model.eval_mode()
    vloss_sum, n = 0.0, 0
    psnrs, ssims = [], []

    for lr_up, hr, mask in loader:
        lr_up = lr_up.to(device, non_blocking=True)
        hr    = hr.to(device,    non_blocking=True)
        mask  = mask.to(device,  non_blocking=True)

        with autocast('cuda', dtype=torch.float16, enabled=amp):
            sr = model.forward_tensor(lr_up).clamp(0, 1)
            l, _ = total_loss(sr, hr, mask)

        vloss_sum += float(l.detach().cpu().item())
        n += 1

        # (val loader uses batch_size=1; if you change it, average over batch here)
        sr_np = sr[0].detach().float().cpu().permute(1, 2, 0).numpy()
        hr_np = hr[0].detach().float().cpu().permute(1, 2, 0).numpy()
        psnrs.append(psnr_metric(hr_np, sr_np, data_range=1.0))
        ssims.append(ssim_metric(hr_np, sr_np, data_range=1.0, channel_axis=2))

    model.train_mode()
    vloss = vloss_sum / max(1, n)
    return vloss, float(np.mean(psnrs) if psnrs else 0.0), float(np.mean(ssims) if ssims else 0.0)


def main(args=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config",     type=str, default="config.yaml")
    p.add_argument("--model",      type=str, required=True, help="srcnn|fsrcnn|espcn")
    # training overrides (fall back to YAML if unset)
    p.add_argument("--epochs",     type=int,   default=None, help="override cfg.sr.epochs")
    p.add_argument("--batch",      type=int,   default=None, help="override cfg.sr.batch")
    p.add_argument("--lr",         type=float, default=None, help="override cfg.sr.lr")
    p.add_argument("--seed",       type=int,   default=None, help="override cfg.run.seed")
    p.add_argument("--weight_decay", type=float, default=0.0, help="Adam weight decay (L2)")
    p.add_argument("--grad_clip",  type=float, default=0.0, help="clip grad-norm to this value (0=off)")
    # runtime / eval
    p.add_argument("--amp",        action="store_true", help="use mixed precision (FP16 autocast)")
    p.add_argument("--val_every",  type=int, default=1, help="validate every N epochs")
    p.add_argument("--save_best",  action="store_true", help="save *_best.pt when best metric improves")
    p.add_argument("--best_metric",type=str, default="loss", choices=["loss","psnr","ssim"])
    p.add_argument("--epoch_ckpt_every", type=int, default=5, help="save epoch checkpoint every N epochs")
    # resume
    p.add_argument("--resume",     type=str, default="", help="path to checkpoint to resume (loads model + epoch)")
    a = p.parse_args([] if args is None else args)

    # -------- config + device -------
    cfg = load_cfg(a.config)
    if a.seed is not None:
        set_seed(a.seed)
    else:
        set_seed(getattr(cfg.run, "seed", 42))

    device = getattr(cfg.run, "device", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print("VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory/(1024**3), 2))
        torch.backends.cudnn.benchmark = True

    model_name = a.model.lower()
    if model_name not in SUPPORTED:
        raise ValueError(f"Trainable models: {sorted(SUPPORTED)} (got '{model_name}')")

    # -------- data (segmentation subset with masks) -------
    seg_root   = getattr(cfg.paths, "segmentation_images", None)
    masks_root = getattr(cfg.paths, "masks_root", None)
    if not seg_root or not masks_root:
        raise ValueError("Missing paths.segmentation_images or paths.masks_root in config.")

    seg_imgs = list_images(seg_root)
    mask_ids = set(list_mask_image_ids(masks_root))
    seg_img_paths = [p for p in seg_imgs if Path(p).stem in mask_ids]
    if not seg_img_paths:
        raise FileNotFoundError("No segmentation images with masks found.")

    random.shuffle(seg_img_paths)
    n = len(seg_img_paths)
    n_train = int(0.7 * n)
    n_val   = int(0.15 * n)
    train_paths = seg_img_paths[:n_train]
    val_paths   = seg_img_paths[n_train:n_train+n_val]

    patch  = getattr(cfg.sr, "patch", 256)
    stride = getattr(cfg.sr, "stride", 128)
    batch  = a.batch if a.batch is not None else getattr(cfg.sr, "batch", 16)
    lr0    = a.lr    if a.lr    is not None else getattr(cfg.sr, "lr", 1e-4)
    epochs = a.epochs if a.epochs is not None else getattr(cfg.sr, "epochs", 10)
    amp    = bool(a.amp)

    ds_train = LesionPatchDataset(train_paths, masks_root, hr_size=patch, stride=stride, lesion_frac=0.5)
    ds_val   = LesionPatchDataset(val_paths,   masks_root, hr_size=patch, stride=stride, lesion_frac=0.5)

    # dataloaders
    cpu_count = multiprocessing.cpu_count()
    num_workers     = int(getattr(cfg.run, "num_workers", max(2, cpu_count//2)))
    prefetch_factor = int(getattr(cfg.run, "prefetch_factor", 2))
    pin_mem = (device == "cuda")

    loader_train = DataLoader(
        ds_train, batch_size=batch, shuffle=True, num_workers=num_workers,
        pin_memory=pin_mem, drop_last=True, persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor
    )
    loader_val = DataLoader(
        ds_val, batch_size=1, shuffle=False, num_workers=num_workers,
        pin_memory=pin_mem, persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor
    )

    print(f"Train images={len(train_paths)} → batches/epoch={len(loader_train)}  |  Val images={len(val_paths)}")

    # -------- model / optim -------
    model = build_sr(model_name, device=device, cfg=cfg).to(device)
    core  = _get_core_module(model)
    model.train_mode()

    opt = torch.optim.Adam(
        [p for p in core.parameters() if p.requires_grad],
        lr=lr0, weight_decay=float(a.weight_decay)
    )
    scaler = GradScaler('cuda', enabled=amp)   # ← modern AMP API

    # -------- logging / outputs -------
    ensure_dir("outputs/sr_models"); ensure_dir("outputs/metrics")
    iter_csv   = Path(f"outputs/metrics/{model_name}_train_iter.csv")
    epoch_csv  = Path(f"outputs/metrics/{model_name}_train_epoch.csv")
    iter_logger  = CSVLogger(iter_csv,  ["epoch","iter","loss","secs","img_per_s","it_per_s","eta_s"])
    epoch_logger = CSVLogger(epoch_csv, ["epoch","train_loss","val_loss","val_psnr","val_ssim","best_metric"])

    best_val_loss = float("inf")
    best_psnr = -1.0
    best_ssim = -1.0
    best_ckpt = Path(f"outputs/sr_models/{model_name}_best.pt")

    alpha       = float(getattr(cfg.sr, "alpha", 0.7))
    lambda_ssim = float(getattr(cfg.sr, "lambda_ssim", 0.1))
    save_every  = int(max(1, a.epoch_ckpt_every))
    best_sel    = a.best_metric.lower()  # "loss" (min) or "psnr"/"ssim" (max)

    # -------- optional resume -------
    start_epoch = 1
    if a.resume:
        rp = Path(a.resume)
        if rp.exists():
            ck = torch.load(rp, map_location=device)
            state = ck.get("model", ck)
            core.load_state_dict(state, strict=False)
            start_epoch = int(ck.get("epoch", 0)) + 1
            print(f"[RESUME] Loaded {rp} (start at epoch {start_epoch})")
        else:
            print(f"[RESUME] Path not found: {rp} (starting fresh)")

    # If resume starts beyond target, do nothing but still ensure a best checkpoint exists
    if start_epoch > epochs:
        print(f"[INFO] Nothing to do: start_epoch({start_epoch}) > target({epochs}).")
        if not best_ckpt.exists():
            torch.save({"model": core.state_dict(), "epoch": start_epoch - 1}, best_ckpt)
            print(f" Wrote BEST (fallback) → {best_ckpt}")
        print(f"[FINAL] Best checkpoint: {best_ckpt.resolve()}")
        return

    # -------- train loop -------
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, epochs + 1):
        last_epoch = epoch
        t_epoch0 = time.perf_counter()
        losses = []

        pbar = tqdm(loader_train, desc=f"[Train:{model_name}] {epoch}/{epochs}", leave=True)
        for it, (lr_up, hr, mask) in enumerate(pbar, start=1):
            t0 = time.perf_counter()
            lr_up = lr_up.to(device, non_blocking=True)
            hr    = hr.to(device,    non_blocking=True)
            mask  = mask.to(device,  non_blocking=True)

            with autocast('cuda', dtype=torch.float16, enabled=amp):
                sr = model.forward_tensor(lr_up)
                loss, comps = total_loss(sr, hr, mask, alpha=alpha, lambda_ssim=lambda_ssim)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            # gradient clip (optional)
            if a.grad_clip and a.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(core.parameters(), max_norm=float(a.grad_clip))

            scaler.step(opt)
            scaler.update()

            # timings / logs
            dt = time.perf_counter() - t0
            bs = lr_up.shape[0]
            losses.append(float(loss.detach().cpu().item()))

            it_per_s  = 1.0 / max(dt, 1e-8)
            img_per_s = bs   / max(dt, 1e-8)
            rem_iters = len(loader_train) - it
            eta_s = rem_iters / max(it_per_s, 1e-8)

            # add SSIM in the bar if available from loss components
            ssim_val = comps.get("ssim", comps.get("ssim_score", None))
            postfix = {
                "loss": f"{np.mean(losses[-50:]):.4f}",
                "it/s": f"{it_per_s:.2f}",
                "img/s": f"{img_per_s:.1f}",
                "ETA": f"{eta_s/60:.1f}m",
            }
            if ssim_val is not None:
                postfix["ssim"] = f"{float(ssim_val):.3f}"
            pbar.set_postfix(postfix)

            iter_logger.write({
                "epoch": epoch, "iter": it, "loss": losses[-1],
                "secs": dt, "img_per_s": img_per_s, "it_per_s": it_per_s, "eta_s": eta_s
            })

        train_loss = float(np.mean(losses) if losses else 0.0)

        # ---- validate per schedule ----
        val_loss = val_psnr = val_ssim = float("nan")
        if epoch % max(1, a.val_every) == 0:
            val_loss, val_psnr, val_ssim = _val_metrics(model, loader_val, device, amp)

            improved = False
            tag = ""
            if best_sel == "loss":
                if val_loss < best_val_loss:
                    best_val_loss = val_loss; improved = True; tag = f"val_loss={val_loss:.4f}"
            elif best_sel == "psnr":
                if val_psnr > best_psnr:
                    best_psnr = val_psnr; improved = True; tag = f"val_psnr={val_psnr:.3f}"
            else:  # ssim
                if val_ssim > best_ssim:
                    best_ssim = val_ssim; improved = True; tag = f"val_ssim={val_ssim:.3f}"

            if a.save_best and improved:
                torch.save({"model": core.state_dict(), "epoch": epoch}, best_ckpt)
                print(f" Saved BEST → {best_ckpt}  ({tag})")

        # ---- periodic safety checkpoint ----
        if (epoch % save_every) == 0:
            ck = Path(f"outputs/sr_models/{model_name}_epoch{epoch}.pt")
            torch.save({"model": core.state_dict(), "epoch": epoch}, ck)
            print(f"[ckpt] {ck}")

        # ---- epoch log ----
        current_best = {
            "loss": best_val_loss if best_val_loss < float("inf") else float("nan"),
            "psnr": best_psnr if best_psnr > -1 else float("nan"),
            "ssim": best_ssim if best_ssim > -1 else float("nan"),
        }[best_sel]

        epoch_logger.write({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_psnr": val_psnr,
            "val_ssim": val_ssim,
            "best_metric": current_best
        })

        print(f"[Epoch {epoch}/{epochs}] "
              f"train={train_loss:.4f} | "
              f"val={val_loss:.4f} psnr={val_psnr:.3f} ssim={val_ssim:.3f} | "
              f"best({best_sel})={current_best:.4f} | "
              f"time={time.perf_counter()-t_epoch0:.1f}s")

    # ---- done + ensure best exists ----
    print("Done.")
    if not best_ckpt.exists():
        torch.save({"model": core.state_dict(), "epoch": last_epoch}, best_ckpt)
        print(f" Wrote BEST (fallback) → {best_ckpt}")
    print(f"[FINAL] Best checkpoint: {best_ckpt.resolve()}")


if __name__ == "__main__":
    main()
