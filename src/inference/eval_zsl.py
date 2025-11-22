# ================================================================
# src/inference/eval_zsl.py — Notebook-Friendly APTOS + IDRiD
# ================================================================
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
from IPython.display import Image as IPImage, display

from src.config.loader import load_cfg
from src.models.zsl import build_zsl
from src.models.zsl.prompts import get_prompts
from src.metrics.cls_metrics import acc_f1


# --------------------------------------------------------------
# Data Loading Helper (APTOS + IDRiD compatible)
# --------------------------------------------------------------
def load_eval_df(labels_csv, images_dir, classes="binary_aptos", per_class=None):
    df = pd.read_csv(labels_csv) if labels_csv.endswith(".csv") else pd.read_excel(labels_csv)

    # --- Detect dataset type automatically ---
    if "id_code" in df.columns:          # APTOS
        img_col, grade_col = "id_code", "diagnosis"
    else:                                # IDRiD or similar
        img_col = next((c for c in df.columns if "image" in c.lower() or "file" in c.lower()), df.columns[0])
        grade_col = next((c for c in df.columns if "grade" in c.lower()), df.columns[1])

    df = df.rename(columns={img_col: "img_id", grade_col: "dr_grade"})
    df["img_id"] = df["img_id"].astype(str).str.replace(r"\.[A-Za-z0-9]+$", "", regex=True)

    # --- Label simplification ---
    if classes in ["binary_idrid", "binary_aptos"]:
        df["label"] = np.where(df["dr_grade"].astype(str).isin(["0", "no dr", "No DR", "NO DR"]), "no_dr", "dr")
    elif classes == "no_vs_mild":
        df["label"] = df["dr_grade"].astype(str).str.lower().map(
            {"no dr": "no_dr", "0": "no_dr", "no_dr": "no_dr", "mild": "mild", "1": "mild"}
        ).fillna("no_dr")
    else:
        df["label"] = df["dr_grade"].astype(str)

    # --- Class balancing ---
    if per_class:
        grp = [g.sample(min(per_class, len(g)), random_state=42) for _, g in df.groupby("label")]
        df = pd.concat(grp, ignore_index=True)

    # --- Build image lookup ---
    base = Path(images_dir)
    all_imgs = {p.stem.lower(): p for p in base.rglob("*") if p.suffix.lower() in [".jpg", ".png", ".jpeg", ".tif", ".tiff"]}

    def match_name(i):
        i_clean = str(i).lower().replace(".jpg", "").replace(".png", "")
        if i_clean in all_imgs:
            return str(all_imgs[i_clean])
        for name in all_imgs:
            if name.endswith(i_clean) or i_clean in name:
                return str(all_imgs[name])
        return None

    df["path"] = df["img_id"].apply(match_name)
    valid = df[df["path"].notnull()].reset_index(drop=True)

    if len(valid) == 0:
        print(f"[WARN] No matches found in {images_dir} — check filename format.")
    else:
        print(f"[OK] Matched {len(valid)} / {len(df)} images in {images_dir}")

    return valid[["img_id", "label", "path"]]


# --------------------------------------------------------------
# Core Evaluation Logic
# --------------------------------------------------------------
def run_eval(zsl_name, df, img_dir, device, classes, prompt_ensemble, tag):
    """
    Runs zero-shot evaluation on a directory of SR or original images.
    Ignores bicubic folders automatically.
    """
    zsl = build_zsl(zsl_name, device=device)
    prompts = get_prompts(classes, ensemble=prompt_ensemble)

    print(f"[ZSL] Searching under {img_dir}")
    all_imgs = {
        p.stem.lower(): p
        for p in Path(img_dir).rglob("*")
        if p.suffix.lower() in [".jpg", ".png", ".jpeg", ".tif", ".tiff"]
        and "bicubic" not in str(p).lower()
    }

    print(f" → Found {len(all_imgs)} images (excluding bicubic)")
    if not all_imgs:
        return {"acc": 0.0, "f1": 0.0, "confusion_path": None}

    y_true, y_pred = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"ZSL-{zsl_name}-{tag}"):
        key = Path(row.path).stem.lower()
        p = all_imgs.get(key)
        if not p:
            continue
        try:
            pred, _ = zsl.classify_pil(Image.open(p).convert("RGB"), prompts)
            y_true.append(row.label)
            y_pred.append(pred)
        except Exception as e:
            print(f"[WARN] Failed to classify {p.name}: {e}")
            continue

    if not y_true:
        print(f"[WARN] No valid images evaluated for {tag} — skipping metrics.")
        return {"acc": 0.0, "f1": 0.0, "confusion_path": None}

    metrics = acc_f1(y_true, y_pred)

    labels_sorted = sorted(list(set(y_true + y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_sorted)

    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix – {tag}")

    out_dir = Path("outputs/metrics/APTOS_ZSL_Evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"confmat_{classes}_{zsl_name}_{tag}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    df_pred = pd.DataFrame({"true": y_true, "pred": y_pred})
    df_pred.to_csv(out_dir / f"preds_{classes}_{zsl_name}_{tag}.csv", index=False)

    # inline confusion display in notebooks
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            display(IPImage(filename=out_path))
    except Exception:
        pass

    torch.cuda.empty_cache()
    metrics["confusion_path"] = str(out_path)
    return metrics


# --------------------------------------------------------------
# Main Notebook/Script Entry Point
# --------------------------------------------------------------
def run_zsl_eval(config="config.yaml", sr_dir="auto", zsl="clip"):
    cfg = load_cfg(config)
    classes = getattr(cfg.zsl, "classes", "binary_aptos")
    per_class = getattr(cfg.eval, "binary_balance", None)
    device = getattr(cfg.run, "device", "cuda")
    prompt_ensemble = getattr(cfg.zsl, "prompt_ensemble", True)

    df_base = load_eval_df(cfg.paths.aptos_labels, cfg.paths.aptos_original, classes, per_class)
    results = []

    # degraded baseline
    degraded_dir = Path(cfg.paths.aptos_degraded)
    if degraded_dir.exists():
        results.append({"condition": "degraded", **run_eval(zsl, df_base, degraded_dir, device, classes, prompt_ensemble, "degraded")})
    else:
        print("[WARN] Skipping degraded baseline — folder not found:", degraded_dir)

    # auto-detect SR folders
    if str(sr_dir).lower() == "auto":
        sr_root = Path(cfg.paths.aptos_sr_root)
        sr_dirs = [p for p in sr_root.rglob("*") if p.is_dir() and any(x in p.name.lower() for x in ["srcnn", "fsrcnn", "espcn", "esrgan", "swinir"])]
        print(f"[AUTO] Found {len(sr_dirs)} SR folders for evaluation.")
        for p in sr_dirs:
            results.append({"condition": p.name, **run_eval(zsl, df_base, p, device, classes, prompt_ensemble, p.name)})
    else:
        results.append({"condition": "original", **run_eval(zsl, df_base, cfg.paths.aptos_original, device, classes, prompt_ensemble, "original")})
        results.append({"condition": "sr", **run_eval(zsl, df_base, sr_dir, device, classes, prompt_ensemble, "sr")})

    # --- Save & display summary ---
    out_dir = Path("outputs/metrics/APTOS_ZSL_Evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"zsl_{classes}_{zsl}.csv"
    df_results = pd.DataFrame(results)
    df_results.to_csv(out_path, index=False)

    # notebook detection for plotting
    try:
        from IPython import get_ipython
        in_notebook = get_ipython() is not None
    except:
        in_notebook = False

    if in_notebook and {"acc", "f1"} <= set(df_results.columns):
        plt.figure(figsize=(8, 5))
        x = np.arange(len(df_results["condition"]))
        width = 0.35
        accs = df_results["acc"]
        f1s = df_results["f1"]

        bars1 = plt.bar(x - width / 2, accs, width, label="Accuracy", color="#2E86AB")
        bars2 = plt.bar(x + width / 2, f1s, width, label="F1-score", color="#F6AA1C")

        for bars in [bars1, bars2]:
            for b in bars:
                h = b.get_height()
                plt.text(b.get_x() + b.get_width()/2, h + 0.01, f"{h:.2f}", ha="center", va="bottom", fontsize=9)

        plt.xticks(x, df_results["condition"], rotation=25, ha="right")
        plt.ylabel("Score")
        plt.ylim(0, 1.05)
        plt.title(f"ZSL Evaluation — {classes.upper()} ({zsl.upper()})")
        plt.legend()
        plt.tight_layout()

        barplot_path = out_dir / f"zsl_summary_barplot_{classes}_{zsl}.png"
        plt.savefig(barplot_path, dpi=250, bbox_inches="tight")
        plt.close()
        display(IPImage(filename=barplot_path))

    print("\n[ZSL Evaluation Summary]")
    print(df_results)
    print(f"Saved summary CSV → {out_path}")
    return df_results


# --------------------------------------------------------------
# Optional CLI Mode
# --------------------------------------------------------------
if __name__ == "__main__":
    run_zsl_eval()
