import json, csv
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

# ===============================
# Percorsi globali (modificabili)
# ===============================
TRIANG_DIR = r"c:\Users\nicol\Desktop\CV_Tracking\3D_tracking\triangulations"
OUT_DIR = r"c:\Users\nicol\Desktop\CV_Tracking\3D_tracking\tracks3d"

# Filtri qualità
MAX_REPROJ_PX_PLAYER = 7.0
MAX_REPROJ_PX_BALL = 5.0
MIN_VIEWS = 2

# Deduplica per-classe (metri)
DEDUP_THRESH_PLAYER_M = 0.05   # 5 cm (più stretto per evitare fusioni tra soggetti vicini)
DEDUP_THRESH_REFEREE_M = 0.07  # 7 cm
DEDUP_THRESH_BALL_M   = 0.12   # 12 cm

# Gating (chi^2 @ dof=3)
CHI2_GATE_3D = 9.35  # se noti drop rapidi, prova 9.35
# Soglia chi^2 per il merge
CHI2_MERGE_3D = 5.0  # più stretto del gate per evitare merge aggressivi

# Rumore di accelerazione (m/s^2) per modello di moto
ACCEL_NOISE_PLAYER = 3.0
ACCEL_NOISE_REFEREE = 3.0
ACCEL_NOISE_BALL   = 30.0  # più alto per la palla

# Robustezza sulla covarianza di misura R (gonfiaggio/clamping)
R_SCALE = 1.0          # >=1 per allargare il gate
R_REG_LAMBDA = 1e-4    # aggiunge λI in m^2
R_EIG_MIN = 1e-4       # ↑ floor a 1 cm std se cov ottimiste
R_EIG_MAX = 1.0        # max autovalore (m^2)

# Miss handling e conferma
MAX_MISSES_PLAYER = 26   # più persistenza per occlusioni lunghe
MAX_MISSES_REFEREE = 10
MAX_MISSES_BALL = 12     # palla più continua

MIN_HITS_CONFIRM_PLAYER = 3
MIN_HITS_CONFIRM_REFEREE = 3
MIN_HITS_CONFIRM_BALL = 1

# Gravità (usata per la palla)
GRAVITY = 9.81

# FPS del video (usato per convertire i frame in secondi)
FPS = 25.0

# ===============================
# Utility
# ===============================
def _mean_reproj_err(d: dict) -> float:
    # Preferisci la media già calcolata se presente
    if "reproj_err_mean_px" in d and d["reproj_err_mean_px"] is not None:
        try:
            return float(d["reproj_err_mean_px"])
        except Exception:
            pass
    e = d.get("reproj_error_px") or {}
    if isinstance(e, dict) and e:
        try:
            vals = [float(abs(v)) for v in e.values()]
            return float(np.mean(vals)) if vals else float("inf")
        except Exception:
            return float("inf")
    return float("inf")

def _class_key(c: str) -> str:
    if not c:
        return "unknown"
    c = str(c).lower()
    if "ball" in c:
        return "ball"
    if "ref" in c:
        return "referee"
    if "play" in c or "person" in c or "human" in c:
        return "player"
    return c

def _load_triangulated(triang_dir: str) -> List[dict]:
    p = Path(triang_dir)
    items = []
    for fp in sorted(p.glob("triangulated_*.json"), key=lambda x: int(x.stem.split("_")[-1])):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            # garantisci presenza di t anche se non nel JSON
            if "t" not in data:
                try:
                    data["t"] = int(fp.stem.split("_")[-1])
                except Exception:
                    pass
            items.append(data)
        except Exception:
            continue
    return items

def _fuse_gaussian(points: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    Prec = np.zeros((3,3), dtype=float)
    b = np.zeros((3,1), dtype=float)
    for x, S in points:
        S = np.asarray(S, dtype=float)
        S = 0.5*(S+S.T) + 1e-6*np.eye(3)
        try:
            W = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            W = np.linalg.pinv(S)
        Prec += W
        b += W @ x.reshape(3,1)
    Prec = 0.5*(Prec+Prec.T) + 1e-9*np.eye(3)
    try:
        Sigma = np.linalg.inv(Prec)
    except np.linalg.LinAlgError:
        Sigma = np.linalg.pinv(Prec)
    mu = (Sigma @ b).reshape(3)
    return mu, Sigma

def _regularize_cov(R: np.ndarray) -> np.ndarray:
    R = 0.5 * (R + R.T)
    try:
        vals, vecs = np.linalg.eigh(R)
        vals = np.clip(vals, R_EIG_MIN, R_EIG_MAX)
        R = (vecs @ np.diag(vals) @ vecs.T)
    except np.linalg.LinAlgError:
        d = np.clip(np.diag(R), R_EIG_MIN, R_EIG_MAX)
        R = np.diag(d)
    R = R + R_REG_LAMBDA * np.eye(3)
    R = R * R_SCALE
    return R

def _mahalanobis2(delta: np.ndarray, S: np.ndarray) -> float:
    S = 0.5 * (S + S.T)
    try:
        y = np.linalg.solve(S, delta.reshape(3,1))
        return float((delta.reshape(1,3) @ y).item())
    except np.linalg.LinAlgError:
        return 1e12

def _dedup_threshold_for_class(cls: str) -> float:
    c = cls.lower()
    if "ball" in c: return DEDUP_THRESH_BALL_M
    if "ref" in c:  return DEDUP_THRESH_REFEREE_M
    return DEDUP_THRESH_PLAYER_M

def _should_merge(cls: str, z1: np.ndarray, R1: np.ndarray, z2: np.ndarray, R2: np.ndarray) -> bool:
    thr = _dedup_threshold_for_class(cls)
    de = float(np.linalg.norm(z1 - z2))
    if de > thr:
        return False
    S = R1 + R2
    d2 = _mahalanobis2(z1 - z2, S)
    return d2 <= CHI2_MERGE_3D

def _dedup_frame_measurements(meas: List[dict]) -> List[dict]:
    """
    Deduplica misure nello stesso frame per classe usando merge gaussiano se:
    (eucl < soglia per-classe) AND (Mahalanobis^2 < chi^2).
    Conserva: class, X, cov, num_views=max, reproj_err_mean_px=media.
    """
    if not meas:
        return meas
    by_cls: Dict[str, List[int]] = {}
    for i, m in enumerate(meas):
        by_cls.setdefault(_class_key(m.get("class","unknown")), []).append(i)

    keep: List[dict] = []
    for cls, idxs in by_cls.items():
        if not idxs: continue
        idxs_sorted = sorted(
            idxs,
            key=lambda i: float(np.trace(np.array(meas[i]["cov"], dtype=float).reshape(3,3)))
        )
        visited = set()
        for i in idxs_sorted:
            if i in visited: continue
            visited.add(i)
            zi = np.array(meas[i]["X"], dtype=float).reshape(3)
            Ri = np.array(meas[i]["cov"], dtype=float).reshape(3,3)
            cluster = [(zi, Ri)]
            views = [int(meas[i].get("num_views", 1))]
            reproj = [float(meas[i].get("reproj_err_mean_px", 0.0))]
            for j in idxs_sorted:
                if j in visited: continue
                zj = np.array(meas[j]["X"], dtype=float).reshape(3)
                Rj = np.array(meas[j]["cov"], dtype=float).reshape(3,3)
                if _should_merge(cls, zi, Ri, zj, Rj):
                    visited.add(j)
                    cluster.append((zj, Rj))
                    views.append(int(meas[j].get("num_views", 1)))
                    reproj.append(float(meas[j].get("reproj_err_mean_px", 0.0)))
                    # aggiorna pivot con la fusione corrente per migliorare il clustering
                    mu_p, Sigma_p = _fuse_gaussian(cluster)
                    zi, Ri = mu_p, Sigma_p
            mu, Sigma = _fuse_gaussian(cluster)
            keep.append({
                "class": cls,
                "X": mu.reshape(3).tolist(),
                "cov": Sigma.tolist(),
                "num_views": int(max(views)),
                "reproj_err_mean_px": float(np.mean(reproj)) if reproj else None
            })
    return keep

# ===============================
# Kalman Tracker 3D CV / Ballistica palla
# ===============================
class Track3D:
    _next_id = 1
    def __init__(self, t: float, z: np.ndarray, R: np.ndarray, cls: str,
                 accel_noise: float, min_hits_confirm: int, max_misses: int):
        self.id = Track3D._next_id; Track3D._next_id += 1
        self.cls = _class_key(cls)
        self.x = np.zeros((6,1), dtype=float)
        self.x[:3,0] = z.reshape(3)
        self.P = np.eye(6, dtype=float) * 1.0
        self.P[:3,:3] = R + 1e-3*np.eye(3)
        self.P[3:,3:] *= 100.0
        self.last_t = float(t)
        self.hits = 1
        self.misses = 0
        self.confirmed = False
        self.last_meas_err_px = None
        self.accel_noise = float(accel_noise)
        self.min_hits_confirm = int(min_hits_confirm)
        self.max_misses = int(max_misses)
        # history per export
        self.history: List[dict] = []
        self._updated_this_step = False

    def _FQ(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        F = np.eye(6)
        F[0,3] = dt; F[1,4] = dt; F[2,5] = dt
        q = (self.accel_noise**2)
        dt2 = dt*dt; dt3 = dt2*dt; dt4 = dt2*dt2
        Qpos = (dt4/4.0) * q
        Qcross = (dt3/2.0) * q
        Qvel = dt2 * q
        Q = np.zeros((6,6))
        for k in range(3):
            Q[k,k] = Qpos
            Q[k,3+k] = Qcross
            Q[3+k,k] = Qcross
            Q[3+k,3+k] = Qvel
        return F, Q

    def predict(self, t: float):
        # usa dt in secondi (t è un indice di frame)
        dt = max(1e-6, (float(t) - self.last_t) / FPS)
        F, Q = self._FQ(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.last_t = float(t)
        self._updated_this_step = False

    def innovation(self, z: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        H = np.zeros((3,6)); H[0,0]=H[1,1]=H[2,2]=1.0
        y = z.reshape(3,1) - (H @ self.x)
        S = H @ self.P @ H.T + R
        S = 0.5*(S+S.T) + 1e-9*np.eye(3)
        K = self.P @ H.T @ np.linalg.inv(S)
        return y, S, K

    def update(self, z: np.ndarray, R: np.ndarray, meas_err_px: float | None):
        y, S, K = self.innovation(z, R)
        self.x = self.x + K @ y
        I = np.eye(6)
        H = np.zeros((3,6)); H[0,0]=H[1,1]=H[2,2]=1.0
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T
        self.hits += 1
        self.misses = 0
        if self.hits >= self.min_hits_confirm:
            self.confirmed = True
        self.last_meas_err_px = float(meas_err_px) if meas_err_px is not None else None
        self._updated_this_step = True

    def miss(self):
        self.misses += 1
        self._updated_this_step = False

    def log_state(self, t: float):
        self.history.append({
            "t": int(t),
            "track_id": int(self.id),
            "class": self.cls,
            "x": float(self.x[0,0]),
            "y": float(self.x[1,0]),
            "z": float(self.x[2,0]),
            "vx": float(self.x[3,0]),
            "vy": float(self.x[4,0]),
            "vz": float(self.x[5,0]),
            "updated": bool(self._updated_this_step),
            "confirmed": bool(self.confirmed),
            "meas_err_px": self.last_meas_err_px if self.last_meas_err_px is not None else ""
        })

    def as_row_from_hist(self, h: dict) -> List:
        return [
            h["t"], h["track_id"], h["class"],
            h["x"], h["y"], h["z"], h["vx"], h["vy"], h["vz"], h["meas_err_px"]
        ]

class BallTrack(Track3D):
    def predict(self, t: float):
        # usa dt in secondi (t è un indice di frame)
        dt = max(1e-6, (float(t) - self.last_t) / FPS)
        F, Q = self._FQ(dt)
        self.x = F @ self.x
        self.x[2,0] += -0.5 * GRAVITY * dt * dt
        self.x[5,0] += -GRAVITY * dt
        self.P = F @ self.P @ F.T + Q
        self.last_t = float(t)
        self._updated_this_step = False

# ===============================
# Tracker 3D
# ===============================
class Tracker3D:
    def __init__(self, debug: bool = False):
        self.debug = bool(debug)
        # tracce vive per classe
        self.tracks_by_cls = {"player":[], "ball":[], "referee":[], "unknown":[]}
        # tracce concluse (prunate) per export
        self.finished_by_cls = {"player":[], "ball":[], "referee":[], "unknown":[]}

    def _assoc(self, tracks: List[Track3D], meas: List[Tuple[np.ndarray, np.ndarray, float]]) -> Tuple[List[Tuple[int,int]], List[int], List[int]]:
        """
        Associazione inter-frame tra tracce esistenti e misure del frame corrente.
        - Costruisce una matrice dei costi con distanza di Mahalanobis d^2 = y^T S^{-1} y.
        - Applica un gating chi^2 (CHI2_GATE_3D) per invalidare accoppiamenti implausibili.
        - Risolve l'assegnamento con Hungarian (linear_sum_assignment).
        Ritorna:
          - matches: lista di coppie (i_track, j_meas)
          - un_t: indici delle tracce non assegnate
          - un_m: indici delle misure non assegnate
        """
        if len(tracks) == 0 or len(meas) == 0:
            return [], list(range(len(tracks))), list(range(len(meas)))
        C = np.full((len(tracks), len(meas)), fill_value=1e6, dtype=float)
        for i, tr in enumerate(tracks):
            for j, (z, R, _) in enumerate(meas):
                y, S, _ = tr.innovation(z, R)
                try:
                    S_inv_y = np.linalg.solve(S, y)
                    d2 = float((y.T @ S_inv_y).item())
                except np.linalg.LinAlgError:
                    d2 = 1e9
                if d2 <= CHI2_GATE_3D:
                    C[i,j] = d2
        r, c = linear_sum_assignment(C)
        matches = [(int(i), int(j)) for i, j in zip(r, c) if C[i,j] < 1e6]
        assigned_t = {i for i,_ in matches}
        assigned_m = {j for _,j in matches}
        un_t = [i for i in range(len(tracks)) if i not in assigned_t]
        un_m = [j for j in range(len(meas)) if j not in assigned_m]
        return matches, un_t, un_m

    def step(self, t: float, measurements: List[dict]):
        meas_by_cls: Dict[str, List[Tuple[np.ndarray, np.ndarray, float]]] = {"player":[], "ball":[], "referee":[], "unknown":[]}
        for m in measurements:
            cls = _class_key(m.get("class", "unknown"))
            z = np.array(m["X"], dtype=float).reshape(3)
            R = _regularize_cov(np.array(m["cov"], dtype=float).reshape(3,3))
            merr = m.get("reproj_err_mean_px", None)
            if merr is None or not np.isfinite(merr):
                merr = _mean_reproj_err(m)

            # pesa R in base alla qualità della misura e al numero di viste
            scale = 1.0 + ((merr or 0.0) / 3.0)**2
            num_views = int(m.get("num_views", 1))
            if num_views >= 3:
                scale *= 0.8
            elif num_views == 2:
                scale *= 1.2
            scale = float(np.clip(scale, 0.5, 10.0))
            R = R * scale

            meas_by_cls[cls].append((z, R, merr))
        for cls, tracks in self.tracks_by_cls.items():
            # predict
            for tr in tracks:
                tr.predict(t)
            matches, un_t, un_m = self._assoc(tracks, meas_by_cls[cls])
            if self.debug:
                pair_str = ", ".join(f"tr#{tracks[i].id}->m{j}" for i, j in matches)
                print(f"[t={int(t)}] {cls}: matches [{pair_str}] | un_t={len(un_t)} | un_m={len(un_m)}")

            # update
            for i, j in matches:
                z, R, merr = meas_by_cls[cls][j]
                tracks[i].update(z, R, merr)
            # miss
            for i in un_t:
                tracks[i].miss()
            # births (per-classe)
            for j in un_m:
                z, R, merr = meas_by_cls[cls][j]
                if cls == "ball":
                    tr = BallTrack(
                        t, z, R, cls,
                        accel_noise=ACCEL_NOISE_BALL,
                        min_hits_confirm=MIN_HITS_CONFIRM_BALL,
                        max_misses=MAX_MISSES_BALL
                    )
                elif cls == "referee":
                    tr = Track3D(
                        t, z, R, cls,
                        accel_noise=ACCEL_NOISE_REFEREE,
                        min_hits_confirm=MIN_HITS_CONFIRM_REFEREE,
                        max_misses=MAX_MISSES_REFEREE
                    )
                else:  # player/unknown
                    tr = Track3D(
                        t, z, R, cls,
                        accel_noise=ACCEL_NOISE_PLAYER,
                        min_hits_confirm=MIN_HITS_CONFIRM_PLAYER,
                        max_misses=MAX_MISSES_PLAYER
                    )
                tr.last_meas_err_px = float(merr) if merr is not None else None
                tracks.append(tr)

            # log history e prune
            for tr in tracks:
                tr.log_state(t)
            # sposta le finite in finished_by_cls e mantieni le vive
            alive, finished = [], []
            for tr in tracks:
                if tr.misses <= tr.max_misses:
                    alive.append(tr)
                else:
                    finished.append(tr)
            if finished:
                self.finished_by_cls[cls].extend(finished)
            self.tracks_by_cls[cls] = alive

# ===============================
# Pipeline
# ===============================
def build_tracks():
    tri = _load_triangulated(TRIANG_DIR)
    if not tri:
        print("Nessun triangulated_*.json trovato.")
        return
    out_dir = Path(OUT_DIR); out_dir.mkdir(parents=True, exist_ok=True)

    # mappa t -> points
    tri_by_t: Dict[int, List[dict]] = {}
    for it in tri:
        try:
            t = int(it.get("t"))
            pts = it.get("points3d", it.get("points", it.get("data", [])))
            if isinstance(pts, list):
                tri_by_t[t] = pts
        except Exception:
            continue

    tracker = Tracker3D()
    rows: List[List] = []
    tri_sorted = sorted(tri_by_t.items(), key=lambda kv: kv[0])

    for t, points in tri_sorted:
        # Filtra qualità e mappa in misure
        measurements = []
        for p in points:
            cls = _class_key(p.get("class","unknown"))
            numv = int(p.get("num_views", 0))
            if numv < MIN_VIEWS:
                continue
            err_px = _mean_reproj_err(p)

            # soglie di reproiezione adattive per num_views
            if cls == "ball":
                base_thr = MAX_REPROJ_PX_BALL
            else:
                base_thr = MAX_REPROJ_PX_PLAYER
            # allenta di ~1 px quando hai solo 2 viste
            err_thr = base_thr + (1.0 if numv == 2 else 0.0)

            if err_px > err_thr:
                continue

            measurements.append({
                "class": cls,
                "X": p["X"],
                "cov": p["cov"],
                "num_views": numv,
                "reproj_err_mean_px": err_px
            })

        # Dedup per frame con Mahalanobis + soglia per-classe
        measurements = _dedup_frame_measurements(measurements)

        # Step tracking
        tracker.step(t, measurements)

    # Export: solo tracce confermate e “recenti” (misses <= 2 al frame)
    header = ["t","track_id","class","x","y","z","vx","vy","vz","meas_err_px"]
    for cls, tracks in tracker.tracks_by_cls.items():
        for tr in tracks:
            # calcola age_since_update sugli snapshot
            age = 999
            for h in tr.history:
                if h["updated"]:
                    age = 0
                else:
                    age = age + 1
                # esporta snapshot confermati se aggiornati, oppure se "recenti"
                if h["confirmed"] and (h["updated"] or age <= 8):
                    rows.append(tr.as_row_from_hist(h))

    # includi anche le tracce finite
    for cls, tracks in tracker.finished_by_cls.items():
        for tr in tracks:
            age = 999
            for h in tr.history:
                if h["updated"]:
                    age = 0
                else:
                    age = age + 1
                if h["confirmed"] and (h["updated"] or age <= 8):
                    rows.append(tr.as_row_from_hist(h))

    csv_path = out_dir / "tracks3d.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"Salvato: {csv_path}")

    # aggregazioni per stats
    all_tracks = []
    for v in tracker.tracks_by_cls.values():
        all_tracks.extend(v)
    for v in tracker.finished_by_cls.values():
        all_tracks.extend(v)
    durations_confirmed = [sum(1 for h in tr.history if h["confirmed"]) for tr in all_tracks]

    stats = {
        "frames": len(tri_sorted),
        "rows_csv": len(rows),
        "tracks_alive_by_class": {k: len(v) for k,v in tracker.tracks_by_cls.items()},
        "tracks_finished_by_class": {k: len(v) for k,v in tracker.finished_by_cls.items()},
        "tracks_total": sum(len(v) for v in tracker.tracks_by_cls.values()) + sum(len(v) for v in tracker.finished_by_cls.values()),
        "confirmed_durations_frames": {
            "mean": float(np.mean(durations_confirmed)) if durations_confirmed else 0.0,
            "median": float(np.median(durations_confirmed)) if durations_confirmed else 0.0,
            "count": len(durations_confirmed)
        },
        "params": {
            "MIN_VIEWS": MIN_VIEWS,
            "MAX_REPROJ_PX_PLAYER": MAX_REPROJ_PX_PLAYER,
            "MAX_REPROJ_PX_BALL": MAX_REPROJ_PX_BALL,
            "CHI2_GATE_3D": CHI2_GATE_3D,
            "ACCEL_NOISE_PLAYER": ACCEL_NOISE_PLAYER,
            "ACCEL_NOISE_REFEREE": ACCEL_NOISE_REFEREE,
            "ACCEL_NOISE_BALL": ACCEL_NOISE_BALL,
            "MAX_MISSES_PLAYER": MAX_MISSES_PLAYER,
            "MAX_MISSES_REFEREE": MAX_MISSES_REFEREE,
            "MAX_MISSES_BALL": MAX_MISSES_BALL,
            "MIN_HITS_CONFIRM_PLAYER": MIN_HITS_CONFIRM_PLAYER,
            "MIN_HITS_CONFIRM_REFEREE": MIN_HITS_CONFIRM_REFEREE,
            "MIN_HITS_CONFIRM_BALL": MIN_HITS_CONFIRM_BALL,
            "GRAVITY": GRAVITY,
            "FPS": FPS,
            "R_SCALE": R_SCALE,
            "R_REG_LAMBDA": R_REG_LAMBDA,
            "R_EIG_MIN": R_EIG_MIN,
            "R_EIG_MAX": R_EIG_MAX,
            "DEDUP_THRESH_PLAYER_M": DEDUP_THRESH_PLAYER_M,
            "DEDUP_THRESH_REFEREE_M": DEDUP_THRESH_REFEREE_M,
            "DEDUP_THRESH_BALL_M": DEDUP_THRESH_BALL_M,
            "CHI2_MERGE_3D": CHI2_MERGE_3D
        }
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Salvate stats: {out_dir/'stats.json'}")

if __name__ == "__main__":
    build_tracks()