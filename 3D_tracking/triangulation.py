"""
- carica calibrazioni (K,R,t) e osservazioni per camera
- esegue matching epipolare per coppie di camere (Sampson + costo composito) e clustering multi-vista
- triangola (DLT), rifinisce con Levenberg-Marquardt e stima la covarianza
- filtra per errore di riproiezione e salva, per ogni frame, associazioni e triangolazioni in JSON
"""

import json
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from collections import Counter  # aggiunta
import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.optimize import linear_sum_assignment

# ===============================
# Percorsi globali (modificabili)
# ===============================
CALIB_ROOT = r"c:\Users\nicol\Desktop\CV_Tracking\support_material\3D_tracking_material\camera_data"
OBSERVATIONS_PATH = r"c:\Users\nicol\Desktop\CV_Tracking\2D_tracking\rTracked"  # file outX_tracks.json o cartella
OUT_DIR = r"c:\Users\nicol\Desktop\CV_Tracking\3D_tracking\triangulations"

# Mappa class_id comuni -> nome classe normalizzato
CLASS_ID_MAP = {
    0: "ball",   # COCO: person
    1: "player",   # alcuni tracker usano 1 per player
    2: "referee",  # COCO: referee
}

# Unità t: abilita conversione esplicita mm->m come in GTdataManipulation
ASSUME_T_MM = False

# ===============================
# 1) Loader calibrazione (adattato a camera_data/cam_X)
# ===============================
def _parse_image_size_from_metadata(meta_path: Path) -> Tuple[int, int] | None:
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    # tentativi comuni
    if "image_size" in meta and isinstance(meta["image_size"], (list, tuple)) and len(meta["image_size"]) == 2:
        return int(meta["image_size"][0]), int(meta["image_size"][1])
    if "resolution" in meta and isinstance(meta["resolution"], (list, tuple)) and len(meta["resolution"]) == 2:
        return int(meta["resolution"][0]), int(meta["resolution"][1])
    w = meta.get("image_width") or meta.get("width")
    h = meta.get("image_height") or meta.get("height")
    if w is not None and h is not None:
        return int(w), int(h)
    return None

# Helper: forza qualsiasi rappresentazione (anche annidata) a vettore 3x1
def _first_vec3(v) -> np.ndarray:
    arr = np.array(v, dtype=float)
    if arr.ndim == 1 and arr.size >= 3:
        return arr[:3].reshape(3, 1)
    if arr.ndim == 2:
        if arr.shape == (3, 1):
            return arr
        if arr.shape == (1, 3):
            return arr.reshape(3, 1)
        if arr.shape[0] >= 3 and arr.shape[1] >= 1:
            return arr[:3, :1]
        if arr.shape[1] >= 3 and arr.shape[0] >= 1:
            return arr[:1, :3].reshape(3, 1)
    if arr.ndim == 3:
        # es. (N,3,1) o (N,1,3)
        if arr.shape[0] >= 1 and arr.shape[1:] == (3, 1):
            return arr[0]
        if arr.shape[0] >= 1 and arr.shape[1:] == (1, 3):
            return arr[0].reshape(3, 1)
    raise ValueError("Impossibile interpretare il vettore 3x1 da struttura fornita.")

def _parse_KRt(calib_json: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # K
    K = None
    for k in ("K", "camera_matrix", "mtx", "intrinsic_matrix"):
        if k in calib_json:
            K = np.array(calib_json[k], dtype=float).reshape(3, 3)
            break
    if K is None:
        raise ValueError("Calib JSON missing intrinsic matrix (K/camera_matrix).")

    # R / rvec / rvecs
    R = None
    if "R" in calib_json:
        R = np.array(calib_json["R"], dtype=float).reshape(3, 3)
    elif "rvec" in calib_json:
        rvec = _first_vec3(calib_json["rvec"])
        R, _ = cv2.Rodrigues(rvec)
    elif "rvecs" in calib_json:
        rvec = _first_vec3(calib_json["rvecs"])
        R, _ = cv2.Rodrigues(rvec)
    else:
        raise ValueError("Calib JSON missing R or rvec/rvecs.")

    # t / tvec / tvecs
    if "t" in calib_json:
        t = _first_vec3(calib_json["t"])
    elif "tvec" in calib_json:
        t = _first_vec3(calib_json["tvec"])
    elif "tvecs" in calib_json:
        t = _first_vec3(calib_json["tvecs"])
    else:
        raise ValueError("Calib JSON missing t or tvec/tvecs.")

    # Conversione unità coerente con GTdataManipulation
    units = (calib_json.get("units") or calib_json.get("t_units") or "").lower()
    if units in ("mm", "millimeter", "millimeters"):
        t = t / 1000.0
    elif units in ("m", "meter", "meters"):
        pass  # già in metri
    else:
        if ASSUME_T_MM:
            t = t / 1000.0
        elif np.linalg.norm(t) > 2000.0:  # evita falsi positivi quando t è già in metri (~20-40)
            t = t / 1000.0

    return K, R, t

def load_calibration(calib_path: str) -> Dict[str, dict]:
    """
    Supporta:
      - ROOT con sottocartelle cam_*/calib/camera_calib.json (+metadata.json)
      - Singola cartella cam_X con calib/camera_calib.json
      - Fallback: file camera_*.json nella cartella (formato generico K/R/t)
    Ritorna: {cam_id: {K,R,t,P,image_size}}
    """
    calib: Dict[str, dict] = {}
    root = Path(calib_path)

    # Caso 1: singola cam folder
    single_cam_file = root / "calib" / "camera_calib.json"
    if single_cam_file.exists():
        cam_dir = root
        cam_id = cam_dir.name if cam_dir.name.startswith("cam_") else "cam_0"
        data = json.loads(single_cam_file.read_text(encoding="utf-8"))
        K, R, t = _parse_KRt(data)
        img_size = _parse_image_size_from_metadata(cam_dir / "metadata.json") \
                   or tuple(data.get("image_size", [1920, 1080]))
        Rt = np.hstack([R, t])
        P = K @ Rt
        calib[cam_id] = {"K": K, "R": R, "t": t, "P": P, "image_size": (int(img_size[0]), int(img_size[1]))}
        return calib

    # Caso 2: root con molte cam_*
    cam_dirs = sorted([d for d in root.glob("cam_*") if (d / "calib" / "camera_calib.json").exists()])
    if cam_dirs:
        for cam_dir in cam_dirs:
            cam_id = cam_dir.name
            data = json.loads((cam_dir / "calib" / "camera_calib.json").read_text(encoding="utf-8"))
            K, R, t = _parse_KRt(data)
            img_size = _parse_image_size_from_metadata(cam_dir / "metadata.json") \
                       or tuple(data.get("image_size", [1920, 1080]))
            Rt = np.hstack([R, t])
            P = K @ Rt
            calib[cam_id] = {"K": K, "R": R, "t": t, "P": P, "image_size": (int(img_size[0]), int(img_size[1]))}
        return calib

    # Caso 3: fallback vecchio (camera_*.json piatti)
    for f in sorted(root.glob("camera_*.json")):
        cam_id = f.stem.replace("camera_", "")
        data = json.loads(f.read_text(encoding="utf-8"))
        K, R, t = _parse_KRt(data)
        img_size = tuple(data.get("image_size", [1920, 1080]))
        Rt = np.hstack([R, t])
        P = K @ Rt
        calib[cam_id] = {"K": K, "R": R, "t": t, "P": P, "image_size": (int(img_size[0]), int(img_size[1]))}

    if not calib:
        raise FileNotFoundError(f"Nessuna calibrazione trovata in: {root}")
    return calib

# ===============================
# 2) Fundamental/Essential
# ===============================
def fundamental_from_krti(K1, R1, t1, K2, R2, t2) -> np.ndarray:
    """
    E = [t_rel]_x R_rel nel frame 1;
    F = K2^{-T} E K1^{-1}
    """
    R_rel = R2 @ R1.T
    t_rel = (t2 - R_rel @ t1).reshape(3,)
    tx = np.array([[0, -t_rel[2], t_rel[1]],
                   [t_rel[2], 0, -t_rel[0]],
                   [-t_rel[1], t_rel[0], 0]], dtype=float)
    E = tx @ R_rel
    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
    return F

# ===============================
# 3) Metriche e costi
# ===============================
def sampson_distance(p1: np.ndarray, p2: np.ndarray, F: np.ndarray) -> float:
    """
    p1, p2 in pixel (u,v). Restituisce Sampson distance.
    """
    x1 = np.array([p1[0], p1[1], 1.0])
    x2 = np.array([p2[0], p2[1], 1.0])
    Fx1 = F @ x1
    Ftx2 = F.T @ x2
    num = (x2.T @ F @ x1) ** 2
    den = Fx1[0] ** 2 + Fx1[1] ** 2 + Ftx2[0] ** 2 + Ftx2[1] ** 2
    if den <= 1e-12:
        return 1e6
    return float(num / den)

def pairwise_cost_matrix(detA: List[dict], detB: List[dict], weights: dict, F: np.ndarray, image_size=(1920,1080)) -> np.ndarray:
    """
    Cost = w_epi * d_epi + w_2d * d_2d (+ opzionali app/col/ocr)
    """
    w_epi = weights.get("w_epi", 1.0)
    w_2d = weights.get("w_2d", 0.0)

    H, W = image_size[1], image_size[0]
    norm = np.hypot(W, H)

    C = np.zeros((len(detA), len(detB)), dtype=float)
    for i, a in enumerate(detA):
        for j, b in enumerate(detB):
            d_epi = sampson_distance(a["xy"], b["xy"], F)
            d_2d = np.linalg.norm(np.array(a["xy"]) - np.array(b["xy"])) / (norm + 1e-9)
            C[i, j] = w_epi * d_epi + w_2d * d_2d
    # Normalizzazione robusta (min-max clippato)
    if C.size:
        lo, hi = np.percentile(C, [5, 95])
        C = (C - lo) / (max(hi - lo, 1e-9))
        C = np.clip(C, 0.0, 1.0)
    return C

# ===============================
# 4) Matching + Clustering
# ===============================
def match_pairs(cost: np.ndarray, max_cost: float) -> List[Tuple[int, int]]:
    if cost.size == 0:
        return []
    r, c = linear_sum_assignment(cost)
    matches = [(int(i), int(j)) for i, j in zip(r, c) if cost[i, j] <= max_cost]
    return matches

def build_clusters(pairwise_matches: Dict[Tuple[str, str], List[Tuple[Tuple[str,int], Tuple[str,int]]]]) -> List[dict]:
    """
    Crea componenti connesse su grafo: nodi=(cam_id, idx_loc), edge=match accettati.
    Ritorna lista cluster: [{"nodes":[(cam,idx),...]}]
    """
    # Union-Find semplice
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a,b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for (camA, camB), pairs in pairwise_matches.items():
        for (na, nb) in pairs:
            union(na, nb)

    groups = {}
    for node in list(parent.keys()):
        root = find(node)
        groups.setdefault(root, []).append(node)

    clusters = [{"cluster_id": f"C_{k:03d}", "nodes": nodes} for k, nodes in enumerate(groups.values(), start=1)]
    return clusters

# ===============================
# 5) Triangolazione (DLT) + 6) Refine + 7) Covarianza
# ===============================
def triangulate_linear(points_2d: List[np.ndarray], Ps: List[np.ndarray]) -> np.ndarray:
    """
    DLT N-view: restituisce X omogeneo (4,)
    """
    A = []
    for (u, v), P in zip(points_2d, Ps):
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    A = np.asarray(A)
    _, _, vh = np.linalg.svd(A)
    X = vh[-1]
    X = X / (X[3] + 1e-12)
    return X  # [x,y,z,1]

def _project(K, R, t, X):
    Xc = R @ X.reshape(3,1) + t
    x = K @ Xc
    u = (x[0]/x[2]).item()
    v = (x[1]/x[2]).item()
    return np.array([u, v], dtype=float)

def refine_point_lm(X0, points_2d, cams, sigma_px=1.5):
    """
    Minimizza residui di reproiezione; restituisce X_refined (3,), per_cam_errors, J (jacobiano numerico)
    """
    cam_list = [(cams[i]["K"], cams[i]["R"], cams[i]["t"]) for i in range(len(points_2d))]

    def residuals(x):
        res = []
        for (K,R,t), obs in zip(cam_list, points_2d):
            uhat = _project(K,R,t,x)
            res.extend(uhat - obs)
        return np.array(res)

    # LM
    r = least_squares(residuals, X0, method="lm")
    x_ref = r.x
    # errori per camera
    per_cam_err = {}
    res = residuals(x_ref).reshape(-1,2)
    for i, e in enumerate(res):
        per_cam_err[f"cam_{i+1}"] = float(np.linalg.norm(e))
    # Jacobiano (approssimato da least_squares)
    if r.jac is not None and r.jac.size > 0:
        J = r.jac
    else:
        # fallback numerico semplice
        eps = 1e-6
        J = []
        f0 = residuals(x_ref)
        for k in range(3):
            dx = np.zeros(3); dx[k] = eps
            f1 = residuals(x_ref + dx)
            J.append((f1 - f0)/eps)
        J = np.array(J).T
    return x_ref, per_cam_err, J

def estimate_covariance(J: np.ndarray, sigma_px: float, W: np.ndarray|None=None) -> np.ndarray:
    """
    Sigma_X ≈ (J^T W J)^{-1} * sigma_px^2
    """
    if W is None:
        W = np.eye(J.shape[0])
    JTJ = J.T @ W @ J
    # regolarizza per stabilità
    JTJ += 1e-9 * np.eye(JTJ.shape[0])
    cov = np.linalg.inv(JTJ) * (sigma_px ** 2)
    return cov

def solve_pnp_from_field(cam_dir: Path, cam_data: dict, use_rectified_2d: bool = False) -> dict:
    """
    Ristima R, t utilizzando solvePnP dai punti del campo annotati in img_points.json.
    Formato atteso:
    {"world_points": [[x,y,z], ...], "image_points": [[u,v], ...]}
    """
    points_file = cam_dir / "img_points.json"
    if not points_file.exists():
        raise FileNotFoundError(f"File punti campo non trovato: {points_file}")
    
    data = json.loads(points_file.read_text(encoding="utf-8"))
    world_points = np.array(data.get("world_points"), dtype=np.float32)
    image_points = np.array(data.get("image_points"), dtype=np.float32)
    
    if world_points.shape[0] < 4 or image_points.shape[0] < 4:
        raise ValueError(f"Servono almeno 4 punti per solvePnP, trovati: {world_points.shape[0]}")
    
    if world_points.shape[0] != image_points.shape[0]:
        raise ValueError(f"Numero di punti mondo ({world_points.shape[0]}) e immagine ({image_points.shape[0]}) non corrispondono")
    
    K = cam_data["K"]
    dist_coeffs = np.zeros(5)  # Se use_rectified_2d=True, assumiamo punti già rettificati
    
    # Usa K_rect se specificato e presente
    if use_rectified_2d and "K_rect" in cam_data:
        K = cam_data["K_rect"]
    
    # Stima posa con solvePnP
    ret, rvec, tvec = cv2.solvePnP(
        world_points, image_points, K, dist_coeffs, 
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not ret:
        raise RuntimeError(f"solvePnP fallito per {cam_dir.name}")
    
    # Converti rvec in matrice di rotazione
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3, 1)
    
    # Calcola errore di riproiezione
    projected, _ = cv2.projectPoints(world_points, rvec, tvec, K, dist_coeffs)
    errors = np.squeeze(projected) - image_points
    errors_norm = np.linalg.norm(errors, axis=1)
    rmse = np.sqrt(np.mean(errors_norm ** 2))
    max_err = np.max(errors_norm)
    
    # Aggiorna e ritorna i parametri della camera
    result = cam_data.copy()
    result.update({
        "K": K,
        "R": R,
        "t": t,
        "P": K @ np.hstack([R, t]),
        "reproj_rmse_px": float(rmse),
        "reproj_max_px": float(max_err),
    })
    
    return result

# ===============================
# Loader osservazioni da outX_tracks.json / outX_tracked.json
# ===============================
def _deduce_cam_id_from_filename(name: str) -> str | None:
    m = re.search(r"out(\d+)_", name)
    if m:
        return f"cam_{m.group(1)}"
    return None

def _extract_xy(track: dict) -> Tuple[float, float] | None:
    # 1) xy diretto
    if "xy" in track and isinstance(track["xy"], (list, tuple)) and len(track["xy"]) == 2:
        return float(track["xy"][0]), float(track["xy"][1])
    # 2) center / cx, cy
    if "center" in track and isinstance(track["center"], (list, tuple)) and len(track["center"]) == 2:
        return float(track["center"][0]), float(track["center"][1])
    if "cx" in track and "cy" in track:
        return float(track["cx"]), float(track["cy"])
    # 3) bbox -> usa bottom-center per player/referee, center per altri
    if "bbox" in track and isinstance(track["bbox"], (list, tuple)) and len(track["bbox"]) == 4:
        x, y, w_or_x2, h_or_y2 = track["bbox"]
        is_xyxy = (w_or_x2 > x) and (h_or_y2 > y)
        if is_xyxy:
            w = float(w_or_x2 - x); h = float(h_or_y2 - y)
        else:
            w = float(w_or_x2); h = float(h_or_y2)
        cls = str(track.get("class", track.get("label", track.get("name", "")))).lower()
        if ("play" in cls or "person" in cls or "human" in cls or "ref" in cls):
            # bottom-center per punti “a terra”
            return float(x + 0.5 * w), float(y + h)
        else:
            # center per palla/altro
            return float(x + 0.5 * w), float(y + 0.5 * h)
    return None

def _class_from_id(cid, name: str | None = None) -> str:
    # Normalizza un nome testo se presente
    def _norm_name(s: str) -> str:
        s = s.strip().lower()
        if s in ("person", "player", "human"):
            return "player"
        if "ball" in s:
            return "ball"
        if "ref" in s or "arb" in s:
            return "referee"
        return s or "unknown"

    if name:
        return _norm_name(name)

    try:
        iv = int(cid)
    except Exception:
        return "unknown"
    return CLASS_ID_MAP.get(iv, "unknown")

def _parse_tracks_file(fp: Path, cam_id: str) -> Dict[int, List[dict]]:
    """
    Ritorna: dict frame -> list of detections per quella camera
    """
    data = json.loads(fp.read_text(encoding="utf-8"))
    per_frame: Dict[int, List[dict]] = defaultdict(list)

    # Due formati possibili: lista di frames {"frame":i,"tracks":[...]} oppure dizionario
    if isinstance(data, list):
        frames_iter = data
    elif isinstance(data, dict) and "frames" in data:
        frames_iter = data["frames"]
    else:
        raise ValueError(f"Formato sconosciuto per {fp}")

    for frame_entry in frames_iter:
        if not isinstance(frame_entry, dict):
            continue
        fidx = int(frame_entry.get("frame", frame_entry.get("id", -1)))
        tracks = frame_entry.get("tracks", []) or frame_entry.get("detections", [])
        for k, tr in enumerate(tracks):
            xy = _extract_xy(tr)
            if xy is None:
                continue
            raw_cid = tr.get("class_id", tr.get("cls", None))
            raw_name = tr.get("label", tr.get("name", None))
            det = {
                "cam_id": cam_id,
                "obj_id": tr.get("track_id", tr.get("id", k)),
                "xy": [float(xy[0]), float(xy[1])],
                "conf": float(tr.get("score", tr.get("conf", 1.0))),
                "class_id": int(raw_cid) if isinstance(raw_cid, (int, float, str)) and str(raw_cid).isdigit() else None,
                "class": _class_from_id(raw_cid, raw_name),
            }
            per_frame[fidx].append(det)
    return per_frame

def load_observations(path_or_dir: str, known_cams: List[str]) -> dict:
    """
    Se path_or_dir è file: usa solo quel file e deduce cam_id dal nome.
    Se è cartella: unisce tutti i file out*_track*.json / out*_tracked*.json.
    Output: {"timestamps":[{"t": frame_idx, "detections":[...]}, ...]}
    """
    p = Path(path_or_dir)
    all_frames: Dict[int, List[dict]] = defaultdict(list)

    if p.is_file():
        cam_id = _deduce_cam_id_from_filename(p.name)
        if cam_id is None:
            # se non deducibile ma c'è una sola cam calibrata, usa quella
            if len(known_cams) == 1:
                cam_id = list(known_cams)[0]
            else:
                raise ValueError(f"Impossibile dedurre cam_id dal nome file {p.name}. Rinominare come out<id>_....json")
        per_frame = _parse_tracks_file(p, cam_id)
        for fidx, dets in per_frame.items():
            all_frames[fidx].extend(dets)
    else:
        # Cerca file multipli
        files = sorted(list(p.glob("out*_track*.json")) + list(p.glob("out*_tracked*.json")) + list(p.glob("*tracks*.json")))
        if not files:
            raise FileNotFoundError(f"Nessun file di tracking trovato in {p}")
        for fp in files:
            cam_id = _deduce_cam_id_from_filename(fp.name)
            if cam_id is None:
                # prova a mappare con cam note se matcha il numero nel nome della cartella
                cam_id = next((c for c in known_cams if c in fp.stem), None)
            if cam_id is None:
                continue
            per_frame = _parse_tracks_file(fp, cam_id)
            for fidx, dets in per_frame.items():
                all_frames[fidx].extend(dets)

    timestamps = []
    for fidx in sorted(all_frames.keys()):
        timestamps.append({"t": int(fidx), "detections": all_frames[fidx]})
    return {"timestamps": timestamps}

# ===============================
# 8) Pipeline per timestamp
# ===============================
def process_timestamp(t, detections_t: List[dict], calib: Dict[str,dict], weights: dict, params: dict) -> dict:
    """
    Esegue: costi → matching → cluster → triangolazione → refine → covarianza → filtri
    """
    # Raggruppa detections per camera
    by_cam: Dict[str, List[dict]] = {}
    for d in detections_t:
        by_cam.setdefault(d["cam_id"], []).append(d)

    cams = sorted([c for c in by_cam.keys() if c in calib])
    pairwise_matches = {}

    # Costi e matching per coppie
    for i in range(len(cams)):
        for j in range(i+1, len(cams)):
            ca, cb = cams[i], cams[j]
            Fa = calib[ca]; Fb = calib[cb]
            F = fundamental_from_krti(Fa["K"], Fa["R"], Fa["t"], Fb["K"], Fb["R"], Fb["t"])
            C = pairwise_cost_matrix(by_cam[ca], by_cam[cb], weights, F, image_size=Fa["image_size"])
            matches_idx = match_pairs(C, max_cost=params.get("max_cost", 0.5))
            matched = []
            max_epi = params.get("max_epi", None)
            for ia, ib in matches_idx:
                # Gating epipolare opzionale
                if max_epi is not None:
                    d_epi = sampson_distance(by_cam[ca][ia]["xy"], by_cam[cb][ib]["xy"], F)
                    if d_epi > max_epi:
                        continue
                na = (ca, ia); nb = (cb, ib)
                matched.append((na, nb))
            pairwise_matches[(ca, cb)] = matched

    clusters = build_clusters(pairwise_matches)

    points3d = []
    for cl in clusters:
        # Prepara liste per triangolazione
        P_list, xy_list, used_cams, dets_used = [], [], [], []
        for (cam_id, det_idx) in cl["nodes"]:
            if cam_id not in calib:
                continue
            det = by_cam[cam_id][det_idx]
            P_list.append(calib[cam_id]["P"])
            xy_list.append(np.array(det["xy"], dtype=float))
            used_cams.append(cam_id)
            dets_used.append(det)

        if len(P_list) < 2:
            continue

        Xh = triangulate_linear(xy_list, P_list)
        X0 = Xh[:3]

        X_ref, per_cam_err, J = refine_point_lm(X0, xy_list, [calib[c] for c in used_cams], sigma_px=params.get("sigma_px", 1.5))
        cov = estimate_covariance(J, sigma_px=params.get("sigma_px", 1.5))

        # Filtra per errore medio
        mean_err = float(np.mean(list(per_cam_err.values()))) if per_cam_err else 1e9
        if mean_err > params.get("max_reproj_px", 3.0):
            continue

        # Propagazione classe: se c'è almeno una 'ball', usa 'ball', altrimenti maggioranza
        classes = [d.get("class", "unknown") for d in dets_used if d.get("class")]
        if any(c == "ball" for c in classes):
            clazz = "ball"
        elif classes:
            clazz = Counter(classes).most_common(1)[0][0]
        else:
            clazz = "unknown"

        points3d.append({
            "cluster_id": cl["cluster_id"],
            "class": clazz,
            "X": X_ref.tolist(),
            "cov": cov.tolist(),
            "reproj_error_px": per_cam_err,
            "num_views": len(P_list)
        })

    return {"t": t, "clusters": clusters, "points3d": points3d}

# ===============================
# 9) Main / CLI
# ===============================
def main():
    ap = argparse.ArgumentParser()
    # Percorsi fissi via globali
    ap.add_argument("--w_epi", type=float, default=1.0)
    ap.add_argument("--w_2d", type=float, default=0.2)
    ap.add_argument("--max_cost", type=float, default=0.8)
    ap.add_argument("--max_epi", type=float, default=None)  # gating epipolare opzionale
    ap.add_argument("--sigma_px", type=float, default=1.5)
    ap.add_argument("--max_reproj_px", type=float, default=7.0)
    # Nuovi flag PnP/rettifica
    ap.add_argument("--pnp_from_field", action="store_true",
                    help="Ristima R,t da img_points.json per ciascuna cam (world=campo)")
    ap.add_argument("--use_rectified_2d", action="store_true",
                    help="Punti 2D già senza distorsione/da video rettificato (usa K_rect, dist=0)")
    ap.add_argument("--write_rectified_calib", action="store_true",
                    help="Scrive un file camera_calib_rectified.json con K_rect, R, t, units=m")
    args = ap.parse_args()

    weights = {"w_epi": args.w_epi, "w_2d": args.w_2d}
    params = {
        "max_cost": args.max_cost,
        "max_epi": args.max_epi,
        "sigma_px": args.sigma_px,
        "max_reproj_px": args.max_reproj_px
    }

    # Carica calibrazioni
    calib_root = Path(CALIB_ROOT)
    if not calib_root.exists():
        raise FileNotFoundError(f"CALIB_ROOT non esiste: {calib_root}")
    calib = load_calibration(str(calib_root))

    # Opzionale: ristima posa con solvePnP dai punti campo
    if args.pnp_from_field:
        updated = {}
        for cam_id, cam_data in calib.items():
            cam_dir = calib_root / cam_id
            try:
                cam_upd = solve_pnp_from_field(cam_dir, cam_data, use_rectified_2d=args.use_rectified_2d)
                updated[cam_id] = cam_upd
                print(f"[PnP] {cam_id}: reproj RMSE={cam_upd['reproj_rmse_px']:.2f}px "
                      f"(max={cam_upd['reproj_max_px']:.2f}px)")
                if args.write_rectified_calib:
                    out_path = cam_dir / "calib" / "camera_calib_rectified.json"
                    to_save = {
                        "mtx": cam_upd["K"].tolist(),
                        "R": cam_upd["R"].tolist(),
                        "t": cam_upd["t"].tolist(),
                        "units": "m",
                        "image_size": list(cam_upd["image_size"]),
                    }
                    out_path.write_text(json.dumps(to_save, indent=2), encoding="utf-8")
            except FileNotFoundError:
                # nessun img_points.json: salta
                updated[cam_id] = cam_data
                print(f"[PnP] {cam_id}: img_points.json mancante, uso calib esistente.")
        calib = updated

    # Carica osservazioni da file o cartella (costruisce struttura timestamps)
    obs_src = Path(OBSERVATIONS_PATH)
    if not obs_src.exists():
        raise FileNotFoundError(f"OBSERVATIONS_PATH non esiste: {obs_src}")
    obs = load_observations(str(obs_src), list(calib.keys()))

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ts in obs.get("timestamps", []):
        t = ts["t"]
        detections = ts["detections"]

        cams_in_frame = sorted({d["cam_id"] for d in detections if d["cam_id"] in calib})
        print(f"[t={t}] cams={cams_in_frame} dets={len(detections)}", flush=True)

        result = process_timestamp(t, detections, calib, weights, params)
        print(f"   clusters={len(result['clusters'])} points3d={len(result['points3d'])}", flush=True)

        # associazioni
        (out_dir / f"associations_{t}.json").write_text(json.dumps(
            {"t": t, "clusters": result["clusters"]}, indent=2), encoding="utf-8")

        # triangolati
        (out_dir / f"triangulated_{t}.json").write_text(json.dumps(
            {"t": t, "points3d": result["points3d"]}, indent=2), encoding="utf-8")

    print(f"Done. Output in: {out_dir}")

if __name__ == "__main__":
    main()