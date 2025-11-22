# src/splits/aptos_split.py
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

def stratified_split(csv_path, val_size=0.2, random_state=42):
    df = pd.read_csv(csv_path)
    if "diagnosis" not in df.columns:
        raise ValueError("CSV must have 'diagnosis' column")
    tr, va = train_test_split(
        df, test_size=val_size, random_state=random_state, stratify=df["diagnosis"]
    )
    return tr.reset_index(drop=True), va.reset_index(drop=True)

def save_splits(out_dir, train_df, val_df, train_name="train_split.csv", val_name="val_split.csv"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trp, vap = out / train_name, out / val_name
    train_df.to_csv(trp, index=False); val_df.to_csv(vap, index=False)
    return str(trp), str(vap)

def class_counts(csv_path):
    df = pd.read_csv(csv_path)
    c = Counter(df["diagnosis"].astype(int).tolist())
    return dict(sorted(c.items()))
