# src/inference/runner.py
import argparse
import pandas as pd
from pathlib import Path
from subprocess import run, CalledProcessError

from src.config.loader import load_cfg
from src.utils.paths import ensure_dir

def sh(cmd_list):
    # Prints + executes a subprocess command as a list (safer than shell str)
    print(">", " ".join(cmd_list))
    run(cmd_list, check=True)

def main(args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    # Optional degrade controls to fan through to export_sr
    ap.add_argument("--degrade_scale", type=int, default=None)
    ap.add_argument("--degrade_blur", action="store_true")
    ap.add_argument("--degrade_noise", type=float, default=0.0)
    a = ap.parse_args([] if args is None else args)

    cfg = load_cfg(a.config)
    ensure_dir("outputs/sr_images")
    ensure_dir("outputs/metrics")

    grading_images = str(Path(cfg.paths.grading_images))
    sr_models = list(getattr(cfg.sr, "models", ["bicubic", "srcnn", "fsrcnn", "espcn"]))
    zsl_backbones = list(getattr(cfg.zsl, "backbones", ["clip"]))
    classes = getattr(cfg.zsl, "classes", "no_vs_mild")

    # 1) Export SR images for all SR models (skip on error)
    manifests = {}
    for m in sr_models:
        out_root = "outputs/sr_images"
        cmd = [
            "python", "-m", "src.inference.export_sr",
            "--config", a.config,
            "--model", m,
            "--images", grading_images,
            "--out", out_root,
            "--skip_existing",
        ]
        # Degrade knobs (optional)
        if a.degrade_scale is not None:
            cmd += ["--degrade_scale", str(a.degrade_scale)]
        if a.degrade_blur:
            cmd += ["--degrade_blur"]
        if a.degrade_noise and a.degrade_noise > 0:
            cmd += ["--degrade_noise", str(a.degrade_noise)]

        # Pass ckpt only for non-bicubic if it exists
        if m.lower() != "bicubic":
            ckpt = Path(f"outputs/sr_models/{m}_best.pt")
            if ckpt.exists():
                cmd += ["--ckpt", str(ckpt)]
            else:
                print(f"[WARN] {m}: checkpoint not found at {ckpt} → exporting with random init.")

        try:
            sh(cmd)
            # Remember the SR folder produced for this model
            sr_dir = Path(out_root) / f"{m}_x{a.degrade_scale if a.degrade_scale else cfg.sr.scale}"
            if not sr_dir.exists():
                # Fallback to old layout if you change names later
                sr_dir = Path(out_root) / m
            manifests[m] = sr_dir
        except CalledProcessError as e:
            print(f"[WARN] export_sr failed for {m}: {e}")

    # 2) Evaluate ZSL (original vs SR) for each SR model across backbones
    rows = []
    for m in sr_models:
        sr_dir = manifests.get(m, None)
        if not sr_dir or not sr_dir.exists():
            print(f"[SKIP] No SR output folder for {m}")
            continue

        for z in zsl_backbones:
            cmd = [
                "python", "-m", "src.inference.eval_zsl",
                "--config", a.config,
                "--sr_dir", str(sr_dir),
                "--zsl", z,
            ]
            try:
                sh(cmd)
                # Metrics filename pattern mirrors eval_zsl.py output naming
                out_csv_dir = Path("outputs/metrics")
                # The ZSL script uses the *name* of the sr_dir (e.g., "srcnn_x2")
                sr_tag = sr_dir.name
                csv = out_csv_dir / f"zsl_{classes}_{z}_{sr_tag}.csv"
                if csv.exists():
                    df = pd.read_csv(csv)
                    # Expect two rows: original / sr
                    sr_row = df[df["condition"] == "sr"]
                    if not sr_row.empty:
                        acc = float(sr_row["accuracy"].values[0])
                        f1  = float(sr_row["macro_f1"].values[0])
                        rows.append({"sr_model": m, "sr_folder": sr_tag, "zsl": z, "accuracy": acc, "macro_f1": f1})
            except CalledProcessError as e:
                print(f"[WARN] ZSL eval failed for ({m},{z}): {e}")

    if rows:
        out = pd.DataFrame(rows)
        out.to_csv("outputs/metrics/comparison.csv", index=False)
        print("Saved comparison → outputs/metrics/comparison.csv")

if __name__ == "__main__":
    main()
