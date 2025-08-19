import csv, json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

# ===============================
# Percorsi globali (modificabili)
# ===============================
TRACKS3D_CSV = r"c:\Users\nicol\Desktop\CV_Tracking\3D_tracking\tracks3d\tracks3d.csv"
GT3D_PATH    = r"c:\Users\nicol\Desktop\CV_Tracking\gt\gt3d.csv"  # atteso CSV: t,track_id,class,x,y,z
OUT_DIR      = r"c:\Users\nicol\Desktop\CV_Tracking\3D_tracking\eval"

# Parametri matching/metriche
MAX_MATCH_DIST_M = 0.5  # gating per associazione GT<->Pred (metri)
CLASSES_TO_EVAL = ["player", "ball"]  # scegli cosa valutare

# ===============================
# Loader
# ===============================
def _class_key(c: str) -> str:
    if not c: return "unknown"
    s = str(c).lower()
    if "ball" in s: return "ball"
    if "ref" in s:  return "referee"
    if "play" in s or "person" in s or "human" in s: return "player"
    return s

def load_tracks_csv(path: str) -> Dict[str, Dict[int, List[Tuple[int, np.ndarray]]]]:
    """
    Ritorna: per classe -> { track_id -> [(t, xyz), ...] } (ordinati per t)
    """
    by_cls: Dict[str, Dict[int, List[Tuple[int, np.ndarray]]]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                t = int(float(row["t"]))
                tid = int(row["track_id"])
                cls = _class_key(row.get("class", "unknown"))
                x = float(row["x"]); y = float(row["y"]); z = float(row["z"])
            except Exception:
                continue
            by_cls.setdefault(cls, {}).setdefault(tid, []).append((t, np.array([x,y,z], dtype=float)))
    # ordina per tempo
    for cls in by_cls:
        for tid in by_cls[cls]:
            by_cls[cls][tid].sort(key=lambda p: p[0])
    return by_cls

def load_gt_csv(path: str) -> Dict[str, Dict[int, List[Tuple[int, np.ndarray]]]]:
    """
    Atteso CSV con intestazione: t,track_id,class,x,y,z
    """
    by_cls: Dict[str, Dict[int, List[Tuple[int, np.ndarray]]]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                t = int(float(row["t"]))
                tid = int(row["track_id"])
                cls = _class_key(row.get("class", "unknown"))
                x = float(row["x"]); y = float(row["y"]); z = float(row["z"])
            except Exception:
                continue
            by_cls.setdefault(cls, {}).setdefault(tid, []).append((t, np.array([x,y,z], dtype=float)))
    for cls in by_cls:
        for tid in by_cls[cls]:
            by_cls[cls][tid].sort(key=lambda p: p[0])
    return by_cls

# ===============================
# Associazione per frame e metriche
# ===============================
def _frame_index(by_cls: Dict[str, Dict[int, List[Tuple[int, np.ndarray]]]]) -> Dict[str, Dict[int, List[Tuple[int, np.ndarray, int]]]]:
    """
    Per classe, crea indice per frame: t -> lista di (track_id, xyz, idx_local).
    """
    idx: Dict[str, Dict[int, List[Tuple[int, np.ndarray, int]]]] = {}
    for cls, tracks in by_cls.items():
        idx_cls: Dict[int, List[Tuple[int, np.ndarray, int]]] = {}
        for tid, seq in tracks.items():
            for k, (t, xyz) in enumerate(seq):
                idx_cls.setdefault(t, []).append((tid, xyz, k))
        idx[cls] = idx_cls
    return idx

def evaluate_class(pred_cls: Dict[int, List[Tuple[int, np.ndarray]]],
                   gt_cls: Dict[int, List[Tuple[int, np.ndarray]]],
                   max_match_dist: float = MAX_MATCH_DIST_M) -> Dict:
    """
    Associa per frame con Hungarian (distanza euclidea), accumula ADE/FDE/RMSE e ID metriche approssimate.
    """
    pred_idx = _frame_index({"cls": pred_cls})["cls"]
    gt_idx   = _frame_index({"cls": gt_cls})["cls"]
    all_frames = sorted(set(pred_idx.keys()) & set(gt_idx.keys()))
    # mapping storico GT->Pred per IDSW/FRAG
    prev_map: Dict[int, int] = {}  # gt_id -> pred_id
    seen_map: Dict[int, bool] = {} # per FRAG: se gt_id era matchato nel frame precedente
    idsw = 0
    frag = 0
    idtp = 0
    idfp = 0
    idfn = 0

    # error accumulators
    err_list = []    # norm error per match (ADE)
    err_xyz = []     # component-wise for RMSE
    pair_last_err: Dict[Tuple[int,int], float] = {}  # (gt,pred)-> last distance for FDE

    for t in all_frames:
        P = pred_idx.get(t, [])
        G = gt_idx.get(t, [])
        if not P and not G:
            continue
        if not P:
            idfn += len(G)
            # FRAG handling: se GT non è matchato ora ma lo era prima, frammentazione
            for gtid, _, _ in G:
                if seen_map.get(gtid, False):
                    frag += 1
                    seen_map[gtid] = False
            continue
        if not G:
            idfp += len(P)
            continue

        # build cost matrix
        C = np.full((len(G), len(P)), fill_value=1e6, dtype=float)
        for i, (gtid, gxyz, _) in enumerate(G):
            for j, (ptid, pxyz, _) in enumerate(P):
                d = float(np.linalg.norm(gxyz - pxyz))
                if d <= max_match_dist:
                    C[i,j] = d
        r, c = linear_sum_assignment(C)
        matched = [(int(i), int(j)) for i,j in zip(r,c) if C[i,j] < 1e6]

        matched_gt = set()
        matched_pr = set()
        # counts
        idtp += len(matched)
        idfp += (len(P) - len(matched))
        idfn += (len(G) - len(matched))

        # FRAG/IDSW per GT
        current_map: Dict[int, int] = {}
        for i,j in matched:
            gtid, gxyz, _ = G[i]
            ptid, pxyz, _ = P[j]
            matched_gt.add(gtid); matched_pr.add(ptid)
            # metriche di errore
            dvec = (pxyz - gxyz)
            err_list.append(float(np.linalg.norm(dvec)))
            err_xyz.append(dvec.tolist())
            pair_last_err[(gtid, ptid)] = float(np.linalg.norm(dvec))
            # ID switch: GT mappato a pred diverso da prima
            if gtid in prev_map and prev_map[gtid] != ptid:
                idsw += 1
            current_map[gtid] = ptid
            # FRAG: transizione unmatched->matched
            if not seen_map.get(gtid, False):
                seen_map[gtid] = True

        # FRAG: GT non matchati ora ma lo erano prima => frammentazione
        for gtid, _, _ in G:
            if gtid not in matched_gt and seen_map.get(gtid, False):
                frag += 1
                seen_map[gtid] = False

        prev_map = current_map

    # ADE / RMSE
    ade = float(np.mean(err_list)) if err_list else None
    err_xyz_arr = np.array(err_xyz, dtype=float) if err_xyz else np.zeros((0,3))
    rmse_xyz = (float(np.sqrt(np.mean(err_xyz_arr[:,0]**2))) if err_xyz_arr.size else None,
                float(np.sqrt(np.mean(err_xyz_arr[:,1]**2))) if err_xyz_arr.size else None,
                float(np.sqrt(np.mean(err_xyz_arr[:,2]**2))) if err_xyz_arr.size else None)

    # FDE: media sulle coppie (gt,pred) che hanno avuto almeno un match
    fde_vals = list(pair_last_err.values())
    fde = float(np.mean(fde_vals)) if fde_vals else None

    # IDF1
    idf1 = (2*idtp) / (2*idtp + idfp + idfn) if (2*idtp + idfp + idfn) > 0 else None

    return {
        "ADE": ade,
        "FDE": fde,
        "RMSE_xyz": {"x": rmse_xyz[0], "y": rmse_xyz[1], "z": rmse_xyz[2]},
        "IDF1": idf1,
        "IDSW": int(idsw),
        "FRAG": int(frag),
        "IDTP": int(idtp), "IDFP": int(idfp), "IDFN": int(idfn),
        "match_gate_m": max_match_dist
    }

# ===============================
# Visualizzazioni
# ===============================
def plot_bev(tracks_by_cls: Dict[str, Dict[int, List[Tuple[int, np.ndarray]]]], out_path: Path, title: str):
    plt.figure(figsize=(8,6))
    for cls, color in [("player", "#1f77b4"), ("ball", "#d62728"), ("referee", "#2ca02c")]:
        if cls not in tracks_by_cls: continue
        for tid, seq in tracks_by_cls[cls].items():
            xs = [p[1][0] for p in seq]; ys = [p[1][1] for p in seq]
            plt.plot(xs, ys, '-', lw=1, alpha=0.7, color=color)
    plt.xlabel("x [m]"); plt.ylabel("y [m]")
    plt.title(title)
    plt.axis("equal"); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_3d_ball(tracks_by_cls: Dict[str, Dict[int, List[Tuple[int, np.ndarray]]]], out_path: Path, title: str):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    for tid, seq in tracks_by_cls.get("ball", {}).items():
        xs = [p[1][0] for p in seq]; ys = [p[1][1] for p in seq]; zs = [p[1][2] for p in seq]
        ax.plot(xs, ys, zs, '-', lw=2, alpha=0.9)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_z_time_ball(tracks_by_cls: Dict[str, Dict[int, List[Tuple[int, np.ndarray]]]], out_path: Path, title: str):
    plt.figure(figsize=(8,4))
    for tid, seq in tracks_by_cls.get("ball", {}).items():
        ts = [p[0] for p in seq]; zs = [p[1][2] for p in seq]
        plt.plot(ts, zs, '-', lw=2, alpha=0.9, label=f"ball_{tid}")
    plt.xlabel("t [frame]"); plt.ylabel("z [m]")
    plt.title(title); plt.grid(True, alpha=0.3)
    if tracks_by_cls.get("ball"): plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# ===============================
# Main
# ===============================
def main():
    out_dir = Path(OUT_DIR); out_dir.mkdir(parents=True, exist_ok=True)

    # Carica pred e GT
    pred = load_tracks_csv(TRACKS3D_CSV)
    gt   = load_gt_csv(GT3D_PATH)

    # Salva qualche info
    summary = {
        "pred_tracks_by_class": {k: len(v) for k,v in pred.items()},
        "gt_tracks_by_class": {k: len(v) for k,v in gt.items()},
        "params": {"MAX_MATCH_DIST_M": MAX_MATCH_DIST_M, "CLASSES_TO_EVAL": CLASSES_TO_EVAL}
    }

    # Metriche per classe e globali
    metrics = {}
    ade_vals = []; fde_vals = []; rmse_all = []
    idf1_vals = []; idsw_sum = 0; frag_sum = 0
    for cls in CLASSES_TO_EVAL:
        m = evaluate_class(pred.get(cls, {}), gt.get(cls, {}), MAX_MATCH_DIST_M)
        metrics[cls] = m
        if m["ADE"] is not None: ade_vals.append(m["ADE"])
        if m["FDE"] is not None: fde_vals.append(m["FDE"])
        if all(v is not None for v in m["RMSE_xyz"].values()):
            rmse_all.append(list(m["RMSE_xyz"].values()))
        if m["IDF1"] is not None: idf1_vals.append(m["IDF1"])
        idsw_sum += m["IDSW"]; frag_sum += m["FRAG"]

    overall = {
        "ADE": float(np.mean(ade_vals)) if ade_vals else None,
        "FDE": float(np.mean(fde_vals)) if fde_vals else None,
        "RMSE_xyz_mean": dict(zip(["x","y","z"], np.mean(np.array(rmse_all), axis=0))) if rmse_all else {"x":None,"y":None,"z":None},
        "IDF1_mean": float(np.mean(idf1_vals)) if idf1_vals else None,
        "IDSW_sum": int(idsw_sum),
        "FRAG_sum": int(frag_sum)
    }
    metrics["overall"] = overall
    (out_dir / "metrics.json").write_text(json.dumps({"summary": summary, "metrics": metrics}, indent=2), encoding="utf-8")

    # Plot
    plot_bev(pred, out_dir / "bev_pred.png", "BEV Pred (x-y)")
    plot_bev(gt,   out_dir / "bev_gt.png",   "BEV GT (x-y)")
    plot_3d_ball(pred, out_dir / "traj3d_ball_pred.png", "Ball 3D Pred")
    plot_z_time_ball(pred, out_dir / "z_time_ball_pred.png", "Ball z(t) Pred")

    print(f"Salvati: {out_dir/'metrics.json'}, {out_dir/'bev_pred.png'}, {out_dir/'traj3d_ball_pred.png'}, {out_dir/'z_time_ball_pred.png'}")

if __name__ == "__main__":
    main()