from pathlib import Path
import pandas as pd

def load_grading_labels(labels_csv):
    labels_csv = Path(labels_csv)
    if labels_csv.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(labels_csv)
    else:
        df = pd.read_csv(labels_csv)
    # Normalize typical column names (adjust if your CSV differs)
    cols = {c.lower(): c for c in df.columns}
    # try to find image id and dr grade columns
    img_col = next((c for c in df.columns if "image" in c.lower() or "file" in c.lower()), df.columns[0])
    grade_col = next((c for c in df.columns if "dr" in c.lower() and "grade" in c.lower()), df.columns[1])
    out = df.rename(columns={img_col:"img_id", grade_col:"dr_grade"})[["img_id","dr_grade"]].copy()
    # strip extension from img_id if present
    out["img_id"] = out["img_id"].astype(str).str.replace(r"\.[A-Za-z0-9]+$", "", regex=True)
    return out

def attach_image_paths(df_labels, images_dir):
    images_dir = Path(images_dir)
    # try .jpg first then .png
    def guess_path(img_id):
        for ext in [".jpg",".JPG",".png",".jpeg",".tif",".tiff"]:
            p = images_dir / f"{img_id}{ext}"
            if p.exists(): return str(p)
        # fallback: search
        matches = list(images_dir.rglob(f"{img_id}.*"))
        return str(matches[0]) if matches else None
    df = df_labels.copy()
    df["path"] = df["img_id"].apply(guess_path)
    df = df[df["path"].notnull()].reset_index(drop=True)
    return df

def build_balanced_subset(df, class_map=None, per_class=150):
    # class_map maps raw labels to canonical: e.g., {0:"no_dr",1:"mild"...} if needed
    dd = df.copy()
    if class_map:
        dd["dr_grade"] = dd["dr_grade"].map(class_map).fillna(dd["dr_grade"])
    groups = []
    for cls, g in dd.groupby("dr_grade"):
        groups.append(g.sample(min(per_class, len(g)), random_state=42))
    return pd.concat(groups, ignore_index=True)
