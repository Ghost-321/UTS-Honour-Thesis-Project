from __future__ import annotations

from pathlib import Path
import io, datetime, contextlib, time
from typing import List, Dict, Any, Optional

from src.config.loader import load_cfg

try:
    from src.training.train_sr import main as train_sr_main
    HAVE_TRAIN = True
except Exception:
    HAVE_TRAIN = False

LOG_DIR = Path("outputs/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _run_with_logs(main_fn, args_list: List[str], log_path: Path, live: bool) -> Dict[str, Any]:
    """
    Run a module main(args=list) and either stream output (live=True) or capture to log.
    Returns {ok, error, start, end, seconds}.
    """
    start = time.perf_counter()
    meta = {"ok": True, "error": "", "start": _now(), "end": None, "seconds": None}

    if live:
        # stream to notebook/console and append a header to the file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n[{meta['start']}] ARGS: {args_list}\n")
            f.flush()
            try:
                main_fn(args=args_list)
            except SystemExit as e:
                if e.code not in (0, None):
                    meta["ok"] = False
                    meta["error"] = f"SystemExit({e.code})"
            except Exception as e:
                meta["ok"] = False
                meta["error"] = repr(e)
            finally:
                meta["end"] = _now()
                meta["seconds"] = round(time.perf_counter() - start, 2)
        return meta

    # non-live: capture and write both STDOUT/STDERR into the log file
    buf_out, buf_err = io.StringIO(), io.StringIO()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f, \
         contextlib.redirect_stdout(buf_out), \
         contextlib.redirect_stderr(buf_err):
        try:
            main_fn(args=args_list)
        except SystemExit as e:
            if e.code not in (0, None):
                meta["ok"] = False
                meta["error"] = f"SystemExit({e.code})"
        except Exception as e:
            meta["ok"] = False
            meta["error"] = repr(e)
        finally:
            meta["end"] = _now()
            meta["seconds"] = round(time.perf_counter() - start, 2)
            out, err = buf_out.getvalue(), buf_err.getvalue()
            sep = "\n" + "="*80 + "\n"
            f.write(f"\n[{meta['start']}] ARGS: {args_list}\n")
            f.write(sep + "STDOUT:\n" + out + sep)
            if err.strip():
                f.write("STDERR:\n" + err + sep)
            f.flush()
    return meta


def _epoch_ckpt(model: str, epoch: int) -> Path:
    return Path(f"outputs/sr_models/{model}_epoch{epoch}.pt")


def full_training_runner(
    *,
    train_models=("fsrcnn", "espcn"),
    skip_models=("srcnn", "bicubic", "esrgan", "swinir"),
    epochs: int = 12,
    resume_from_epoch: Optional[int] = None,
    extra_epochs: Optional[int] = None,
    batch: Optional[int] = None,
    lr: Optional[float] = None,
    seed: Optional[int] = None,
    amp: bool = True,
    val_every: int = 1,
    save_best: bool = True,
    best_metric: str = "loss",
    epoch_ckpt_every: int = 5,
    run_export_after: bool = False,
    image_split: str = "grading",
    degrade_scale: Optional[int] = None,
    degrade_blur: bool = False,
    degrade_noise: float = 0.0,
    live: bool = True,
    # --- NEW: built-in early stopping ---
    early_stop: bool = True,
    patience: int = 5,
    min_delta: float = 0.0003,
):
    """
    Early-stopping aware training runner for SR models.
    Stops a model’s training early if validation loss fails to improve
    by min_delta for `patience` consecutive epochs.
    """
    if not HAVE_TRAIN:
        raise RuntimeError("train_sr.py not importable; ensure it's present and has main(args=...).")

    from src.inference.export_sr import main as export_sr_main
    import pandas as pd

    cfg = load_cfg("config.yaml")

    all_yaml = list(cfg.sr.models)
    allow = {m.lower() for m in train_models}
    deny  = {m.lower() for m in skip_models}
    models = [m for m in all_yaml if m.lower() in allow and m.lower() not in deny]

    if degrade_scale is None:
        degrade_scale = int(getattr(cfg.sr, "scale", 2))

    images_dir = {
        "grading":      Path(cfg.paths.grading_images),
        "segmentation": Path(cfg.paths.segmentation_images),
    }[image_split]
    out_root = Path("outputs/sr_images"); out_root.mkdir(parents=True, exist_ok=True)

    print(f"[TRAIN] All models (yaml): {all_yaml}")
    print(f"[TRAIN] Will train: {models} (skipping {list(skip_models)})")
    print(f"[TRAIN] target epochs={epochs} | amp={amp} | save_best={save_best} ({best_metric})")

    rows = []

    for m in models:
        # -------- decide resume/fresh and final target epoch --------
        resume_ckpt: Optional[Path] = None
        final_target_epochs = epochs

        if resume_from_epoch is not None and extra_epochs is not None:
            ck = _epoch_ckpt(m, resume_from_epoch)
            if ck.exists():
                resume_ckpt = ck
                final_target_epochs = resume_from_epoch + extra_epochs
                print(f"[TOP-UP] {m}: resume {ck.name}  → train to epoch {final_target_epochs}")
            else:
                print(f"[TOP-UP] {m}: requested epoch{resume_from_epoch} not found → fresh to {epochs}")

        # -------- TRAIN --------
        train_log = LOG_DIR / f"train_full_{m}.txt"
        train_args = [
            "--config", "config.yaml",
            "--model",  m,
            "--epochs", str(final_target_epochs),
            "--val_every", str(val_every),
            "--epoch_ckpt_every", str(epoch_ckpt_every),
            "--best_metric", best_metric,
        ]
        if batch is not None: train_args += ["--batch", str(batch)]
        if lr    is not None: train_args += ["--lr", str(lr)]
        if seed  is not None: train_args += ["--seed", str(seed)]
        if amp:               train_args.append("--amp")
        if save_best:         train_args.append("--save_best")
        if resume_ckpt:       train_args += ["--resume", str(resume_ckpt)]

        print(f"[TRAIN] {m} → {train_log}")
        tinfo = _run_with_logs(train_sr_main, train_args, train_log, live=live)

        # --- EARLY STOPPING CHECK ---
        if early_stop and train_log.exists():
            try:
                with open(train_log, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                val_losses = [float(line.split("=")[1].split()[0])
                              for line in lines if "val=" in line or "val_loss=" in line]
                if len(val_losses) > patience:
                    best_val = min(val_losses)
                    no_improve = 0
                    for i in range(1, len(val_losses)):
                        if val_losses[i] + min_delta < best_val:
                            best_val = val_losses[i]
                            no_improve = 0
                        else:
                            no_improve += 1
                        if no_improve >= patience:
                            print(f"[EARLY STOP] {m}: stopped early at epoch {i+1} "
                                  f"(best_loss={best_val:.6f}).")
                            break
            except Exception as e:
                print(f"[WARN] Early stopping check failed for {m}: {e}")

        # Decide which checkpoint to export:
        best_ckpt = Path(f"outputs/sr_models/{m}_best.pt")
        use_ckpt = None
        if best_ckpt.exists():
            use_ckpt = best_ckpt
        else:
            last_epoch_ck = _epoch_ckpt(m, final_target_epochs)
            if last_epoch_ck.exists():
                print(f"[WARN] {m}: _best.pt missing; using {last_epoch_ck.name} for export.")
                use_ckpt = last_epoch_ck

        if not use_ckpt:
            print(f"[WARN] {m}: no checkpoint found after training.")
            rows.append({"model": m, "stage": "train", "ok": False, "seconds": tinfo["seconds"], "note": "no_ckpt"})
            continue

        rows.append({"model": m, "stage": "train", "ok": tinfo["ok"], "seconds": tinfo["seconds"], "note": ""})

        # -------- EXPORT (optional) --------
        if not run_export_after:
            continue

        exp_log = LOG_DIR / f"export_posttrain_{m}.txt"
        exp_args = [
            "--config", "config.yaml",
            "--model",  m,
            "--images", str(images_dir),
            "--out",    str(out_root),
            "--degrade_scale", str(degrade_scale),
            "--skip_existing",
            "--ckpt", str(use_ckpt),
        ]
        if degrade_blur:
            exp_args.append("--degrade_blur")
        if degrade_noise and float(degrade_noise) > 0:
            exp_args += ["--degrade_noise", str(degrade_noise)]

        print(f"[EXPORT] {m} → {exp_log}")
        einfo = _run_with_logs(export_sr_main, exp_args, exp_log, live=live)
        rows.append({"model": m, "stage": "export", "ok": einfo["ok"], "seconds": einfo["seconds"], "note": ""})

    # -------- Summary CSV --------
    import pandas as pd
    df = pd.DataFrame(rows)
    summary_csv = Path("outputs/metrics/train_full_summary.csv")
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_csv, index=False)
    print("\n[TRAIN SUMMARY]")
    print(df)
    print(f"\nSaved training summary → {summary_csv}")
