import argparse, pandas as pd
from pathlib import Path

def main(labels_csv, out_csv, per_class=150):
    df = pd.read_csv(labels_csv) if labels_csv.endswith(".csv") else pd.read_excel(labels_csv)
    img_col = next((c for c in df.columns if "image" in c.lower() or "file" in c.lower()), df.columns[0])
    grade_col = next((c for c in df.columns if "dr" in c.lower() and "grade" in c.lower()), df.columns[1])
    df = df.rename(columns={img_col:"img_id", grade_col:"dr_grade"})
    df["img_id"] = df["img_id"].astype(str).str.replace(r"\.[A-Za-z0-9]+$", "", regex=True)

    dd = df[df["dr_grade"].astype(str).str.lower().isin(["no dr","0","no_dr","mild","1"])].copy()
    dd.loc[:, "label"] = dd["dr_grade"].astype(str).str.lower().map(
        {"no dr":"no_dr","0":"no_dr","no_dr":"no_dr","mild":"mild","1":"mild"}
    ).fillna("no_dr")

    grp=[]
    for lab,g in dd.groupby("label"):
        grp.append(g.sample(min(per_class, len(g)), random_state=42))
    out = pd.concat(grp, ignore_index=True)[["img_id","label"]]
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"Saved split → {out_csv} ({len(out)} rows)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", type=str, required=True)
    ap.add_argument("--out_csv", type=str, default="outputs/metrics/no_vs_mild_split.csv")
    ap.add_argument("--per_class", type=int, default=150)
    a = ap.parse_args()
    main(a.labels_csv, a.out_csv, a.per_class)
