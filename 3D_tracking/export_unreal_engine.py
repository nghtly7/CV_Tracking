# export_unreal.py
# ---------------------------------------------------------------------
# Export offline dei dati 3D per Unreal Engine (UE usa centimetri, Z up).
# Output:
#  - unreal_tracks.csv   (per DataTable/Blueprint/Sequencer)  <-- include 'row_name'
#  - unreal_frames.jsonl (facoltativo, 1 riga JSON per frame)
#
# USO:
#   1) Imposta le VARIABILI GLOBALI qui sotto (sezione CONFIG)
#   2) python export_unreal.py
#
# PREREQUISITI:
#   - CSV input con colonne: t|frame, track_id, class, x|x_m, y|y_m, z|z_m  (in METRI, piano XY)
#   - 'class' può essere ["player","referee","ball"] o numerica [1,2,0] (mapping incluso)
#
# NOTE:
#   - WORLD_ROT_DEG ruota il piano XY in antiorario (per allineare X forward, Y right di UE)
#   - WORLD_OFFSET_M (metri) applicato DOPO la rotazione; Z viene solo traslata
#   - Lo yaw (deg) è derivato dalla direzione di marcia nel piano XY *in world space*
# ---------------------------------------------------------------------

import os
import json
import math
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

# =========================
# CONFIG (modifica qui)
# =========================
# Percorsi
INPUT_CSV   = r"c:\Users\nicol\Desktop\CV_Tracking\3D_tracking\tracks3d\tracks3d.csv"
OUT_DIR     = r"c:\Users\nicol\Desktop\CV_Tracking\3D_tracking\unreal"

# Tempo / unità
FPS_TRACKS  = 25.0         # fps delle tracce: usato per time_s
CM_PER_M    = 100.0        # Unreal usa centimetri

# Orientamento/posizione del mondo UE (metri/gradi)
WORLD_ROT_DEG   = 0.0                  # rotazione antioraria del piano XY (metri)
WORLD_OFFSET_M  = (0.0, 0.0, 0.0)      # offset (x,y,z) in METRI (dopo la rotazione)

# Filtri / opzioni
KEEP_CLASSES: Optional[List[str]] = ["player", "referee", "ball"]  # None per tenere tutto
REBASE_TIME  = True          # se True, time_s parte da t_min invece che t/fps
WRITE_JSONL  = True          # se False non scrive il .jsonl
SORT_OUTPUT  = True          # ordina CSV per (track_id, frame)

# Nomi file output
OUT_CSV_NAME   = "unreal_tracks.csv"
OUT_JSONL_NAME = "unreal_frames.jsonl"

# =========================
# CLASSI / MAPPING
# =========================
CLASS_ID2NAME = {0: "ball", 1: "player", 2: "referee"}

def pred_class_to_id(name: str) -> int:
    """Robusto: accetta 'player/referee/ball', sinonimi IT/EN o '0/1/2'."""
    if name is None:
        return 1
    s = str(name).strip().lower()
    if s in {"0", "1", "2"}:
        return int(s)
    if any(k in s for k in ["ball", "palla"]):
        return 0
    if any(k in s for k in ["ref", "referee", "arbitro", "arb"]):
        return 2
    if any(k in s for k in ["player", "giocatore", "person", "human"]):
        return 1
    # default: player
    return 1

def class_name_from_any(val) -> str:
    """Ritorna il nome classe canonico ('player/referee/ball') da input vario."""
    try:
        cid = pred_class_to_id(val)
        return CLASS_ID2NAME.get(cid, "player")
    except Exception:
        return "player"

# =========================
# GEOMETRIA / ORIENTAMENTO
# =========================
def apply_world_transform_xy(x_m: float, y_m: float,
                             rot_deg: float = 0.0,
                             offset_m: Tuple[float, float] = (0.0, 0.0)) -> Tuple[float, float]:
    """Ruota XY di rot_deg (gradi, antiorario) e trasla di offset_m (metri)."""
    th = math.radians(rot_deg)
    c, s = math.cos(th), math.sin(th)
    xr = c * x_m - s * y_m + offset_m[0]
    yr = s * x_m + c * y_m + offset_m[1]
    return xr, yr

# =========================
# STIMA YAW
# =========================
def yaw_from_velocity(prev_xy: Optional[Tuple[float, float]],
                      next_xy: Optional[Tuple[float, float]]) -> float:
    """Yaw (deg) dalla direzione nel piano XY (X forward, Y right)."""
    if not prev_xy or not next_xy:
        return 0.0
    dx = next_xy[0] - prev_xy[0]
    dy = next_xy[1] - prev_xy[1]
    if abs(dx) < 1e-8 and abs(dy) < 1e-8:
        return 0.0
    return math.degrees(math.atan2(dy, dx))

def compute_yaw_per_track(df_world_xy: pd.DataFrame) -> Dict[Tuple[int, int], float]:
    """
    Calcola yaw per (track_id, t) usando differenze centrate su XY *in world space* (metri).
    df_world_xy deve avere: t, track_id, xw_m, yw_m
    """
    yaw_lookup: Dict[Tuple[int, int], float] = {}
    for tid, g in df_world_xy.sort_values("t").groupby("track_id"):
        arr = [(int(r.t), float(r.xw_m), float(r.yw_m)) for r in g.itertuples(index=False)]
        n = len(arr)
        for i, (t, x, y) in enumerate(arr):
            prev_xy = (arr[i-1][1], arr[i-1][2]) if i-1 >= 0 else None
            next_xy = (arr[i+1][1], arr[i+1][2]) if i+1 < n else None
            yaw_lookup[(int(tid), int(t))] = yaw_from_velocity(prev_xy, next_xy)
    return yaw_lookup

# =========================
# NORMALIZZAZIONE INPUT
# =========================
_COL_ALIASES = {
    "t":        ["t", "frame", "frame_id", "frame_idx", "f"],
    "track_id": ["track_id", "tid", "trackid", "id"],
    "class":    ["class", "category", "label", "cls", "category_name", "name"],
    "x":        ["x", "x_m", "x_field", "x_world", "x_world_m", "field_x"],
    "y":        ["y", "y_m", "y_field", "y_world", "y_world_m", "field_y"],
    "z":        ["z", "z_m", "z_world", "z_world_m"],
}

def _remap_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename = {}
    for std_name, candidates in _COL_ALIASES.items():
        if std_name in df.columns:
            continue
        for c in candidates:
            if c in df.columns:
                rename[c] = std_name
                break
    if rename:
        df = df.rename(columns=rename)
    missing = {"t", "track_id", "class", "x", "y", "z"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV privo di colonne richieste (anche dopo remap): {sorted(missing)}")
    return df

# =========================
# CORE EXPORT
# =========================
def export_for_unreal(pred_df: pd.DataFrame,
                      out_csv: str,
                      out_jsonl: str,
                      fps_tracks: float,
                      world_rot_deg: float = 0.0,
                      world_offset_m: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                      cm_per_m: float = 100.0,
                      rebase_time: bool = False,
                      keep_classes: Optional[List[str]] = None,
                      write_jsonl: bool = True,
                      sort_output: bool = True) -> None:
    """
    Converte predizioni in payload Unreal.
    - pred_df colonne richieste (accetta alias): t|frame, track_id, class, x|x_m, y|y_m, z|z_m (metri)
    - world_rot_deg: rotazione antioraria del piano XY (metri)
    - world_offset_m: traslazione (metri) applicata dopo la rotazione (anche Z)
    - cm_per_m: conversione unità (UE = centimetri)
    - rebase_time: se True, time_s parte da 0 al primo frame (min(t))
    - keep_classes: lista di classi da tenere (es. ["player","referee","ball"])
    """
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # 1) Normalizzazione colonne e tipi
    df = _remap_columns(pred_df)
    df = df.copy()
    df["t"]        = df["t"].astype(int)
    df["track_id"] = df["track_id"].astype(int)
    df["x"]        = df["x"].astype(float)
    df["y"]        = df["y"].astype(float)
    df["z"]        = df["z"].astype(float)
    df["class"]    = df["class"].map(class_name_from_any)

    # 2) Filtro classi (se richiesto)
    if keep_classes is not None:
        keep = {c.strip().lower() for c in keep_classes}
        df = df[df["class"].str.lower().isin(keep)].copy()

    # 3) Rebase tempo opzionale
    t0 = int(df["t"].min()) if rebase_time else 0

    # 4) Prima passata: calcolo posizioni in *world space* (metri)
    tmp_rows = []
    for r in df.itertuples(index=False):
        xw_m, yw_m = apply_world_transform_xy(float(r.x), float(r.y),
                                              rot_deg=world_rot_deg,
                                              offset_m=(world_offset_m[0], world_offset_m[1]))
        zw_m = float(r.z) + world_offset_m[2]
        tmp_rows.append({
            "t": int(r.t),
            "track_id": int(r.track_id),
            "class": str(r.class_ if hasattr(r, "class_") else r._asdict().get("class", "player")),
            "xw_m": xw_m,
            "yw_m": yw_m,
            "zw_m": zw_m,
        })
    tmp_df = pd.DataFrame(tmp_rows)
    # sistemazione classe (tupla/nome campo)
    if "class" not in tmp_df.columns:
        tmp_df["class"] = df["class"].values

    # 5) Yaw su world XY (metri)
    yaw_lookup = compute_yaw_per_track(tmp_df[["t", "track_id", "xw_m", "yw_m"]])

    # 6) Costruzione righe finali, conversione metri -> cm, tempo, row_name
    rows_csv = []
    frames_map = {}
    for r in tmp_df.itertuples(index=False):
        t   = int(r.t)
        tid = int(r.track_id)
        cls = str(r.class_ if hasattr(r, "class_") else r._asdict().get("class"))

        x_cm = float(r.xw_m) * cm_per_m
        y_cm = float(r.yw_m) * cm_per_m
        z_cm = float(r.zw_m) * cm_per_m

        time_s = (t - t0) / max(fps_tracks, 1e-6)
        yaw_deg = float(yaw_lookup.get((tid, t), 0.0))

        row_name = f"f{t:06d}_t{tid:04d}_{cls}"
        rows_csv.append({
            "row_name": row_name,     # <-- chiave riga per UE (Import Key Field)
            "track_id": tid,
            "class": cls,
            "frame": t,
            "time_s": round(time_s, 6),
            "x_cm": round(x_cm, 3),
            "y_cm": round(y_cm, 3),
            "z_cm": round(z_cm, 3),
            "yaw_deg": round(yaw_deg, 3),
        })

        frames_map.setdefault(t, []).append({
            "id": tid, "class": cls, "x": x_cm, "y": y_cm, "z": z_cm, "yaw": yaw_deg
        })

    # 7) DataFrame finale + ordinamento
    df_out = pd.DataFrame(rows_csv)

    # Deduplica 'row_name' (in rari casi può ripetersi: aggiungo suffisso _nn)
    if df_out["row_name"].duplicated().any():
        df_out["_dup"] = df_out.groupby("row_name").cumcount()
        mask = df_out["_dup"] > 0
        df_out.loc[mask, "row_name"] = (
            df_out.loc[mask, "row_name"] + "_" + df_out.loc[mask, "_dup"].astype(int).astype(str).str.zfill(2)
        )
        df_out = df_out.drop(columns=["_dup"])

    if sort_output:
        df_out = df_out.sort_values(["track_id", "frame"])

    # Ordine colonne: 'row_name' deve stare davanti (comodo per UE)
    col_order = ["row_name", "track_id", "class", "frame", "time_s", "x_cm", "y_cm", "z_cm", "yaw_deg"]
    df_out = df_out[col_order]

    # 8) Salvataggi
    df_out.to_csv(out_csv, index=False, encoding="utf-8", lineterminator="\n")

    if write_jsonl:
        with open(out_jsonl, "w", encoding="utf-8") as f:
            for t in sorted(frames_map.keys()):
                time_s = (t - t0) / max(fps_tracks, 1e-6)
                rec = {"frame": t, "time_s": float(time_s), "actors": frames_map[t]}
                f.write(json.dumps(rec) + "\n")

    # 9) Log
    n_tracks = df_out["track_id"].nunique()
    span_s = (df_out["time_s"].max() - df_out["time_s"].min()) if not df_out.empty else 0.0
    print(f"[Unreal Export] CSV  -> {out_csv}  (tracks={n_tracks}, rows={len(df_out)})")
    if write_jsonl:
        print(f"[Unreal Export] JSONL-> {out_jsonl}")
    print(f"[Unreal Export] time span: {span_s:.2f}s, fps={fps_tracks}, rebase={rebase_time}")

# =========================
# MAIN
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv   = os.path.join(OUT_DIR, OUT_CSV_NAME)
    out_jsonl = os.path.join(OUT_DIR, OUT_JSONL_NAME)

    # Autodetect separatore semplice: prova ',' poi ';'
    try:
        df = pd.read_csv(INPUT_CSV)
    except Exception:
        df = pd.read_csv(INPUT_CSV, sep=';')

    export_for_unreal(
        pred_df=df,
        out_csv=out_csv,
        out_jsonl=out_jsonl,
        fps_tracks=FPS_TRACKS,
        world_rot_deg=WORLD_ROT_DEG,
        world_offset_m=WORLD_OFFSET_M,
        cm_per_m=CM_PER_M,
        rebase_time=REBASE_TIME,
        keep_classes=KEEP_CLASSES,
        write_jsonl=WRITE_JSONL,
        sort_output=SORT_OUTPUT,
    )

if __name__ == "__main__":
    main()
