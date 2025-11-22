from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np

# Flexible loader for IDRiD Localization annotations (OD/Fovea coordinates)
# Handles CSV/XLSX and varied column names.

CANDIDATE_COLS = {
    "img": ["image", "image_name", "filename", "img_id", "Image name", "Image", "File Name"],
    "od_x": ["od_x", "optic_disc_x", "od-x", "OD_x", "OD X", "OD Center X"],
    "od_y": ["od_y", "optic_disc_y", "od-y", "OD_y", "OD Y", "OD Center Y"],
    "fv_x": ["fovea_x", "fv_x", "fovea-x", "Fovea_x", "Fovea X", "Macula X"],
    "fv_y": ["fovea_y", "fv_y", "fovea-y", "Fovea_y", "Fovea Y", "Macula Y"],
}

def _find_col(df: pd.DataFrame, candidates) -> Optional[str]:
    lc = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lc: return lc[name.lower()]
    return None

def load_localization_table(path: str) -> pd.DataFrame:
    p = Path(path)
    df = pd.read_excel(p) if p.suffix.lower() in [".xlsx", ".xls"] else pd.read_csv(p)
    img_c = _find_col(df, CANDIDATE_COLS["img"]) or df.columns[0]
    odx   = _find_col(df, CANDIDATE_COLS["od_x"])
    ody   = _find_col(df, CANDIDATE_COLS["od_y"])
    fvx   = _find_col(df, CANDIDATE_COLS["fv_x"])
    fvy   = _find_col(df, CANDIDATE_COLS["fv_y"])
    cols = {"img_id": img_c}
    if odx and ody: cols.update({"od_x": odx, "od_y": ody})
    if fvx and fvy: cols.update({"fovea_x": fvx, "fovea_y": fvy})
    out = df.rename(columns=cols)[list(cols.values())].copy()
    out.rename(columns={v: k for k,v in cols.items()}, inplace=True)
    out["img_id"] = out["img_id"].astype(str).str.replace(r"\.[A-Za-z0-9]+$", "", regex=True)
    return out

def to_lookup(df: pd.DataFrame) -> Dict[str, Dict[str, Tuple[float,float]]]:
    look: Dict[str, Dict[str, Tuple[float,float]]] = {}
    for _, r in df.iterrows():
        d = {}
        if {"od_x","od_y"}.issubset(r.index) and pd.notnull(r["od_x"]) and pd.notnull(r["od_y"]):
            d["od"] = (float(r["od_x"]), float(r["od_y"]))
        if {"fovea_x","fovea_y"}.issubset(r.index) and pd.notnull(r["fovea_x"]) and pd.notnull(r["fovea_y"]):
            d["fovea"] = (float(r["fovea_x"]), float(r["fovea_y"]))
        look[str(r["img_id"])] = d
    return look

def load_localization(path: str) -> Dict[str, Dict[str, Tuple[float,float]]]:
    df = load_localization_table(path)
    return to_lookup(df)
