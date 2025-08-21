"""
Visualizzazione 3D di giocatori e palla da CSV.

Input atteso (in METRI, Z up):
  colonne richieste (alias accettati): t|frame, track_id, class, x|x_m, y|y_m, z|z_m
  class può essere: ["player","referee","ball"] oppure id [1,2,0]

Controlli tastiera durante la riproduzione:
  SPACE  -> pausa/play
  ← / →  -> frame precedente / successivo
  ↑ / ↓  -> velocità *2 / ÷2
  Q      -> esci

Uso:
  Imposta le variabili nella sezione CONFIG e lancia:
    python displayData.py
"""

import argparse
import math
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --------------------------
# CONFIG (modifica qui)
# --------------------------
CSV_PATH = r"c:\Users\nicol\Desktop\CV_Tracking\3D_tracking\tracks3d\tracks3d.csv"
FPS = 25.0
# None per usare il bounding box dei dati, oppure (LUNGHEZZA, LARGHEZZA) in metri, es. (105.0, 68.0)
FIELD_SIZE: Optional[Tuple[float, float]] = None

# --------------------------
# mapping classi e alias
# --------------------------
CLASS_ID2NAME = {0: "ball", 1: "player", 2: "referee"}

COL_ALIASES = {
    "t":        ["t", "frame", "frame_id", "frame_idx", "f"],
    "track_id": ["track_id", "tid", "trackid", "id"],
    "class":    ["class", "cls", "label", "category", "name"],
    "x":        ["x", "x_m", "x_world", "x_field", "field_x"],
    "y":        ["y", "y_m", "y_world", "y_field", "field_y"],
    "z":        ["z", "z_m", "z_world", "z_field"],
}

def _remap_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename = {}
    for std, cands in COL_ALIASES.items():
        if std in df.columns:  # già ok
            continue
        for c in cands:
            if c in df.columns:
                rename[c] = std
                break
    if rename:
        df = df.rename(columns=rename)
    missing = {"t", "track_id", "class", "x", "y", "z"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV privo di colonne richieste (anche dopo remap): {sorted(missing)}")
    return df

def _class_name(val) -> str:
    s = str(val).strip().lower()
    if s in {"0","1","2"}:
        return CLASS_ID2NAME[int(s)]
    if any(k in s for k in ["ball","palla"]): return "ball"
    if any(k in s for k in ["ref","referee","arbitro","arb"]): return "referee"
    if any(k in s for k in ["player","giocatore","person","human"]): return "player"
    return "player"

# --------------------------
# yaw (facoltativo, per frecce direzione)
# --------------------------
def compute_yaw(df: pd.DataFrame) -> Dict[Tuple[int,int], float]:
    """
    Yaw (deg) per (track_id, t) calcolato con differenze centrate nel piano XY.
    """
    out = {}
    for tid, g in df.sort_values("t").groupby("track_id"):
        xy = g[["t","x","y"]].to_numpy()
        n = len(xy)
        for i in range(n):
            t = int(xy[i,0])
            prev_xy = xy[i-1,1:3] if i-1 >= 0 else None
            next_xy = xy[i+1,1:3] if i+1 < n else None
            if prev_xy is None or next_xy is None:
                out[(int(tid), t)] = 0.0
            else:
                dx = float(next_xy[0]-prev_xy[0])
                dy = float(next_xy[1]-prev_xy[1])
                out[(int(tid), t)] = 0.0 if (abs(dx)<1e-8 and abs(dy)<1e-8) else math.degrees(math.atan2(dy, dx))
    return out

# --------------------------
# plot helper
# --------------------------
def set_equal_3d(ax, X, Y, Z):
    x_min, x_max = np.min(X), np.max(X)
    y_min, y_max = np.min(Y), np.max(Y)
    z_min, z_max = np.min(Z), np.max(Z)
    max_range = max(x_max-x_min, y_max-y_min, z_max-z_min)
    if max_range == 0:
        max_range = 1.0
    cx = (x_max+x_min)/2
    cy = (y_max+y_min)/2
    cz = (z_max+z_min)/2
    r = max_range/2
    ax.set_xlim(cx-r, cx+r)
    ax.set_ylim(cy-r, cy+r)
    ax.set_zlim(max(0, cz-r), cz+r)

# --------------------------
# main visualizer
# --------------------------
def run():
    # 1) carica CSV (prova separatore , poi ;)
    try:
        raw = pd.read_csv(CSV_PATH)
    except Exception:
        raw = pd.read_csv(CSV_PATH, sep=";")

    df = _remap_columns(raw)
    # tipi
    df["t"]        = df["t"].astype(int)
    df["track_id"] = df["track_id"].astype(int)
    df["x"]        = df["x"].astype(float)
    df["y"]        = df["y"].astype(float)
    df["z"]        = df["z"].astype(float)
    df["class"]    = df["class"].map(_class_name)

    # ordina e indicizza per frame
    frames = sorted(df["t"].unique().tolist())
    by_frame = {t: g for t, g in df.groupby("t")}
    yaw = compute_yaw(df)

    # 2) setup figura
    plt.style.use("default")
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    fig.canvas.manager.set_window_title("3D Tracks Viewer")

    # campo: se non specificato, prendo il bounding box XY dei dati
    if FIELD_SIZE is not None:
        Lx, Ly = FIELD_SIZE  # (lunghezza, larghezza) in metri
        xs = np.array([-Lx/2,  Lx/2,  Lx/2, -Lx/2, -Lx/2])
        ys = np.array([-Ly/2, -Ly/2,  Ly/2,  Ly/2, -Ly/2])
        ax.plot(xs, ys, np.zeros_like(xs), lw=1.0, alpha=0.6)
    else:
        x_min, x_max = df["x"].min(), df["x"].max()
        y_min, y_max = df["y"].min(), df["y"].max()
        xs = np.array([x_min, x_max, x_max, x_min, x_min])
        ys = np.array([y_min, y_min, y_max, y_max, y_min])
        ax.plot(xs, ys, np.zeros_like(xs), lw=1.0, alpha=0.4)

    # scatter placeholders per classe (tre collezioni)
    scatters = {
        "player": ax.scatter([], [], [], s=14, marker="o", label="player"),
        "referee": ax.scatter([], [], [], s=24, marker="^", label="referee"),
        "ball": ax.scatter([], [], [], s=40, marker="*", label="ball"),
    }
    quiv = ax.quiver([], [], [], [], [], [], length=1.0, normalize=True)

    txt = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

    set_equal_3d(ax, df["x"].values, df["y"].values, df["z"].values)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("3D position of players and ball")
    ax.legend(loc="upper right")

    state = {"playing": True, "speed": 1.0, "i": 0}

    def on_key(event):
        if event.key == " ":
            state["playing"] = not state["playing"]
        elif event.key == "right":
            state["i"] = min(state["i"]+1, len(frames)-1)
            update(state["i"])
        elif event.key == "left":
            state["i"] = max(state["i"]-1, 0)
            update(state["i"])
        elif event.key == "up":
            state["speed"] = min(state["speed"]*2, 16)
        elif event.key == "down":
            state["speed"] = max(state["speed"]/2, 0.25)
        elif event.key in ("q","Q"):
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)

    def update(idx):
        t = frames[idx]
        g = by_frame[t]
        for cls, sc in scatters.items():
            sub = g[g["class"] == cls]
            if len(sub) == 0:
                sc._offsets3d = ([], [], [])
            else:
                sc._offsets3d = (sub["x"].values, sub["y"].values, sub["z"].values)

        pr = g[g["class"].isin(["player","referee"])]
        if len(pr) > 0:
            xs, ys, zs = pr["x"].values, pr["y"].values, pr["z"].values
            ang = np.array([yaw.get((int(tid), int(t)), 0.0) for tid, t in zip(pr["track_id"], pr["t"])], dtype=float)
            u = np.cos(np.deg2rad(ang))
            v = np.sin(np.deg2rad(ang))
            w = np.zeros_like(u)
            nonlocal quiv
            quiv.remove()
            quiv = ax.quiver(xs, ys, zs, u, v, w, length=1.2, normalize=True, linewidth=0.8)
        else:
            try:
                quiv.remove()
            except Exception:
                pass

        txt.set_text(f"frame: {t}   time: {t/FPS:.2f}s   speed: {state['speed']}x")
        return tuple(scatters.values()) + (txt,)

    def step(_):
        if state["playing"]:
            state["i"] += int(max(1, round(state["speed"])))
            if state["i"] >= len(frames):
                state["i"] = 0  # loop
        return update(state["i"])

    interval_ms = 1000.0 / FPS
    anim = FuncAnimation(fig, step, interval=interval_ms, blit=False)
    plt.show()

# --------------------------
# CLI
# --------------------------
if __name__ == "__main__":
    run()
