# src/inference/runner_a_smoke.py
import io, time, datetime, contextlib
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from src.config.loader import load_cfg
from src.inference.export_sr import main as export_sr_main

LOG_DIR = Path("outputs/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

def _now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _run_with_logs(main_fn, args_list: List[str], log_path: Path, live: bool=False) -> Dict[str, Any]:
    """
    Call a module main(args=...) and capture stdout+stderr to a .txt log.
    Returns dict with ok/error/start/end/seconds.
    If live=True, also print the captured output at the end so you see it in the notebook.
    """
    start = time.perf_counter()
    meta = {"ok": True, "error": "", "start": _now(), "end": None, "seconds": None}

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
    if live:
        print(buf_out.getvalue())
        e = buf_err.getvalue().strip()
        if e:
            print("\n[stderr]\n", e)
    return meta

def sanity_check_and_export(
    skip_models=("srcnn",),
    image_split="grading",
    degrade_scale=None,
    degrade_blur=False,
    degrade_noise=0.0,
    smoke_limit=2,
    only_smoke=True,
    copy_gt_once=True,
    live=True
):
    """
    Quick checks: for each model (except skip), run a tiny export of N images (smoke_limit).
    If only_smoke=False, it then runs a full export (with --skip_existing).
    Writes summary to outputs/metrics/sanity_summary.csv
    """
    cfg = load_cfg("config.yaml")

    images_dir = {
        "grading":      Path(cfg.paths.grading_images),
        "segmentation": Path(cfg.paths.segmentation_images),
    }[image_split]

    out_root = Path("outputs/sr_images")
    out_root.mkdir(parents=True, exist_ok=True)

    models = list(getattr(cfg.sr, "models", []))
    if degrade_scale is None:
        degrade_scale = getattr(cfg.sr, "scale", 2)

    print(f"[INFO] Images: {images_dir}")
    print(f"[INFO] Models: {models} (skipping: {list(skip_models)})")
    print(f"[INFO] Degrade scale: x{degrade_scale}")
    print(f"[INFO] Logs → {LOG_DIR.resolve()}")

    ran_copy_gt = False
    rows = []

    for m in models:
        if m.lower() in [s.lower() for s in skip_models]:
            print(f"[SKIP] {m} (user skip)")
            rows.append({"model": m, "stage": "skip", "ok": True, "seconds": 0, "note": "user_skip"})
            continue

        ckpt = Path(f"outputs/sr_models/{m}_best.pt")
        needs_ckpt = (m.lower() != "bicubic")
        if needs_ckpt and not ckpt.exists():
            note = f"missing_ckpt:{ckpt}"
            print(f"[WARN] {m}: checkpoint missing → {ckpt}")
            rows.append({"model": m, "stage": "smoke", "ok": False, "seconds": 0, "note": note})
            continue

        # ---- SMOKE EXPORT ----
        smoke_args = [
            "--config", "config.yaml",
            "--model",  m,
            "--images", str(images_dir),
            "--out",    str(out_root),
            "--degrade_scale", str(degrade_scale),
            "--limit",  str(smoke_limit),
            "--skip_existing",
        ]
        if copy_gt_once and not ran_copy_gt:
            smoke_args.append("--copy_gt")
            ran_copy_gt = True
        if degrade_blur:
            smoke_args.append("--degrade_blur")
        if degrade_noise > 0:
            smoke_args += ["--degrade_noise", str(degrade_noise)]
        if needs_ckpt:
            smoke_args += ["--ckpt", str(ckpt)]

        smoke_log = LOG_DIR / f"export_smoke_{m}.txt"
        print(f"[SMOKE] {m} → {smoke_log}")
        sinfo = _run_with_logs(export_sr_main, smoke_args, smoke_log, live=live)
        rows.append({"model": m, "stage": "smoke", "ok": sinfo["ok"], "seconds": sinfo["seconds"], "note": sinfo["error"]})

        if only_smoke or not sinfo["ok"]:
            if only_smoke:
                print(f"[ONLY_SMOKE] {m}: skipping full export by request.")
            else:
                print(f"[STOP] {m}: smoke failed, skipping full export.")
            continue

        # ---- FULL EXPORT ----
        full_args = [
            "--config", "config.yaml",
            "--model",  m,
            "--images", str(images_dir),
            "--out",    str(out_root),
            "--degrade_scale", str(degrade_scale),
            "--skip_existing",
        ]
        if degrade_blur:
            full_args.append("--degrade_blur")
        if degrade_noise > 0:
            full_args += ["--degrade_noise", str(degrade_noise)]
        if needs_ckpt:
            full_args += ["--ckpt", str(ckpt)]

        full_log = LOG_DIR / f"export_full_{m}.txt"
        print(f"[EXPORT] {m} → {full_log}")
        finfo = _run_with_logs(export_sr_main, full_args, full_log, live=live)
        rows.append({"model": m, "stage": "export_full", "ok": finfo["ok"], "seconds": finfo["seconds"], "note": finfo["error"]})

    # ---- Summary CSV ----
    df = pd.DataFrame(rows)
    summary_csv = Path("outputs/metrics/sanity_summary.csv")
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_csv, index=False)
    print("\n[SUMMARY]")
    print(df)
    print(f"\nSaved sanity summary → {summary_csv}")
