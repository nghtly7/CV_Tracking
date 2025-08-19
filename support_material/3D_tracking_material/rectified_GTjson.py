"""
Rettifica delle annotazioni COCO usando i parametri di calibrazione per-camera.

Configurazione:
- Imposta in testa al file:
    INPUT_JSON : path del file COCO di input
    OUTPUT_JSON: path del file di output (se vuoto, viene generato aggiungendo _rectified)
    ALPHA      : parametro per getOptimalNewCameraMatrix (0=crop, 1=FOV massimo)
"""

import os
import re
import json
from collections import defaultdict
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np

# =========================
# CONFIGURAZIONE UTENTE
# =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Imposta qui il path del JSON COCO di input (obbligatorio)
INPUT_JSON = r"GroundTruthData/train/_annotations.coco.json"  # es: r"c:\Users\nicol\Desktop\dataset\_annotations.coco.json"

# Imposta qui il path del JSON di output (opzionale).
# Se vuoto, sarà creato accanto all'input con suffisso "_rectified".
OUTPUT_JSON = r"GroundTruthData/train/_annotations_rectified.coco.json"

# Parametro alpha per getOptimalNewCameraMatrix (0=crop, 1=FOV massimo con bordi neri)
ALPHA = 0.0
# =========================


def load_calibration(calib_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carica i parametri di calibrazione da un file JSON.
    Attesi i campi:
      - "mtx": matrice intrinseca 3x3
      - "dist": vettore coefficienti di distorsione (k1, k2, p1, p2, [k3, ...])
    """
    with open(calib_path, 'r') as f:
        calib = json.load(f)
    mtx = np.array(calib["mtx"], dtype=np.float32)
    dist = np.array(calib["dist"], dtype=np.float32)
    return mtx, dist


def extract_image_name(img_entry: Dict[str, Any]) -> str:
    """
    Estrae il nome dell’immagine dall’entry COCO.
    Precedenza a "file_name", fallback su "extra.name" (come nel tuo dataset).
    """
    if "file_name" in img_entry and img_entry["file_name"]:
        return img_entry["file_name"]
    extra = img_entry.get("extra", {})
    return extra.get("name", "")


def extract_cam_index_from_name(name: str) -> str:
    """
    Estrae l’indice camera dal nome immagine.
    Cerca pattern tipo "out13_..." o "out_13_...".
    Ritorna stringa numerica (es. "13") oppure "" se non trova nulla.
    """
    m = re.search(r'out[_]?(\d+)', name)
    return m.group(1) if m else ""


def get_image_size(img_entry: Dict[str, Any]) -> Tuple[int, int]:
    """
    Legge width/height dall’entry immagine COCO.
    Sono necessari per calcolare la new_mtx coerente con la risoluzione.
    """
    w = img_entry.get("width", None)
    h = img_entry.get("height", None)
    if w is None or h is None:
        raise ValueError(f"Image entry id={img_entry.get('id')} missing width/height; add them to the JSON.")
    return int(w), int(h)


class Rectifier:
    """
    Gestisce la rettifica di punti 2D tramite:
      - Parametri di calibrazione (mtx, dist) per camera
      - Matrice intrinseca ottimizzata (new_mtx) dipendente da (width, height, alpha)

    Usa una cache per evitare ricalcoli per la stessa (camera, width, height).
    """
    def __init__(self, alpha: float = 0.0):
        self.alpha = float(alpha)
        # Cache: key=(cam_idx, width, height) -> (mtx, dist, new_mtx)
        self._cache_cam_params: Dict[Tuple[str, int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        # Cartella del presente script (per path calibrazione)
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

    def _get_cam_params(self, cam_idx: str, width: int, height: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Carica (o recupera dalla cache) mtx, dist e new_mtx per la camera e risoluzione dati.
        new_mtx è calcolata con getOptimalNewCameraMatrix in base ad alpha.
        """
        key = (cam_idx, width, height)
        if key in self._cache_cam_params:
            return self._cache_cam_params[key]

        calib_rel_path = os.path.join("camera_data", f"cam_{cam_idx}", "calib", "camera_calib.json")
        calib_path = os.path.join(self.script_dir, calib_rel_path)
        if not os.path.isfile(calib_path):
            raise FileNotFoundError(f"Missing calibration for cam {cam_idx}: {calib_path}")

        mtx, dist = load_calibration(calib_path)

        new_mtx, _ = cv2.getOptimalNewCameraMatrix(
            mtx, dist, (width, height),
            alpha=self.alpha,
            newImgSize=(width, height)
        )

        self._cache_cam_params[key] = (mtx, dist, new_mtx)
        return mtx, dist, new_mtx

    def undistort_points(self, pts: np.ndarray, cam_idx: str, width: int, height: int) -> np.ndarray:
        """
        Converte punti pixel dal frame distorto al frame rettificato.
        """
        if pts.size == 0:
            return pts.astype(np.float32)

        mtx, dist, new_mtx = self._get_cam_params(cam_idx, width, height)
        pts_in = pts.reshape(-1, 1, 2).astype(np.float32)
        pts_und = cv2.undistortPoints(pts_in, mtx, dist, R=None, P=new_mtx)
        return pts_und.reshape(-1, 2)

    @staticmethod
    def clamp_bbox(xyxy: np.ndarray, width: int, height: int) -> np.ndarray:
        """
        Clippa i punti della bbox entro i bordi dell’immagine rettificata.
        """
        xyxy[:, 0] = np.clip(xyxy[:, 0], 0, width)
        xyxy[:, 1] = np.clip(xyxy[:, 1], 0, height)
        return xyxy

    def transform_bbox(self, bbox: List[float], cam_idx: str, width: int, height: int) -> List[float]:
        """
        Trasforma una bbox COCO [x,y,w,h].
        """
        x, y, w, h = bbox
        corners = np.array([
            [x, y],
            [x + w, y],
            [x, y + h],
            [x + w, y + h]
        ], dtype=np.float32)
        und = self.undistort_points(corners, cam_idx, width, height)
        und = self.clamp_bbox(und, width, height)
        xmin = float(np.min(und[:, 0])); ymin = float(np.min(und[:, 1]))
        xmax = float(np.max(und[:, 0])); ymax = float(np.max(und[:, 1]))
        nx = xmin; ny = ymin
        nw = max(0.0, xmax - xmin); nh = max(0.0, ymax - ymin)
        return [nx, ny, nw, nh]

    def transform_segmentation(self, seg: List[List[float]], cam_idx: str, width: int, height: int) -> List[List[float]]:
        """
        Trasforma segmentazioni in formato poligonale COCO.
        """
        new_seg: List[List[float]] = []
        for poly in seg:
            if not poly:
                new_seg.append(poly)
                continue
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            und = self.undistort_points(pts, cam_idx, width, height)
            und[:, 0] = np.clip(und[:, 0], 0, width)
            und[:, 1] = np.clip(und[:, 1], 0, height)
            new_seg.append(und.reshape(-1).astype(float).tolist())
        return new_seg

    def transform_keypoints(self, kps: List[float], cam_idx: str, width: int, height: int) -> List[float]:
        """
        Trasforma keypoints COCO (lista flat [x,y,v,...]).
        """
        if not kps:
            return kps
        arr = np.array(kps, dtype=np.float32).reshape(-1, 3)
        pts = arr[:, :2]; vis = arr[:, 2:3]
        mask = vis.flatten() > 0
        if np.any(mask):
            und = self.undistort_points(pts[mask], cam_idx, width, height)
            pts[mask] = und
        pts[:, 0] = np.clip(pts[:, 0], 0, width)
        pts[:, 1] = np.clip(pts[:, 1], 0, height)
        out = np.concatenate([pts, vis], axis=1).reshape(-1).astype(float).tolist()
        return out


def rectify_coco(input_json: str, output_json: str, alpha: float = 0.0) -> None:
    """
    Pipeline principale per rettificare bbox/segmentazioni/keypoints nel COCO.
    """
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    rectifier = Rectifier(alpha=alpha)

    images: List[Dict[str, Any]] = data.get("images", [])
    annotations: List[Dict[str, Any]] = data.get("annotations", [])

    img_by_id: Dict[int, Dict[str, Any]] = {}
    for img in images:
        iid = img.get("id", None)
        if iid is None:
            raise ValueError("Each image entry must have an 'id'.")
        img_by_id[int(iid)] = img

    annos_by_img: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for ann in annotations:
        img_id = ann.get("image_id", None)
        if img_id is None:
            raise ValueError("Each annotation must have an 'image_id'.")
        annos_by_img[int(img_id)].append(ann)

    updated_annotations: List[Dict[str, Any]] = []

    for idx, (img_id, img) in enumerate(img_by_id.items(), 1):
        name = extract_image_name(img)
        cam_idx = extract_cam_index_from_name(name)
        width, height = get_image_size(img)

        if not cam_idx:
            print(f"[WARN] Could not infer camera index from image name '{name}' (image_id={img_id}). Skipping rectification for this image.")
            updated_annotations.extend(annos_by_img.get(img_id, []))
            continue

        try:
            _ = rectifier._get_cam_params(cam_idx, width, height)
        except FileNotFoundError as e:
            print(f"[WARN] {e}. Skipping image_id={img_id}.")
            updated_annotations.extend(annos_by_img.get(img_id, []))
            continue

        for ann in annos_by_img.get(img_id, []):
            ann_new = dict(ann)

            if "bbox" in ann_new and ann_new["bbox"]:
                ann_new["bbox"] = rectifier.transform_bbox(ann_new["bbox"], cam_idx, width, height)
                x, y, w, h = ann_new["bbox"]
                ann_new["area"] = float(w * h)

            if "segmentation" in ann_new and ann_new["segmentation"]:
                seg = ann_new["segmentation"]
                if isinstance(seg, list) and (len(seg) == 0 or isinstance(seg[0], list)):
                    ann_new["segmentation"] = rectifier.transform_segmentation(seg, cam_idx, width, height)

            if "keypoints" in ann_new and ann_new["keypoints"]:
                ann_new["keypoints"] = rectifier.transform_keypoints(ann_new["keypoints"], cam_idx, width, height)

            updated_annotations.append(ann_new)

        if idx % 100 == 0:
            print(f"Processed {idx}/{len(img_by_id)} images...")

    info = data.get("info", {})
    info_rect = dict(info)
    info_rect["description"] = f"{info.get('description', '')} (rectified alpha={alpha})".strip()

    data_out = {
        "info": info_rect,
        "licenses": data.get("licenses", []),
        "categories": data.get("categories", []),
        "images": images,
        "annotations": updated_annotations,
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data_out, f, ensure_ascii=False)

    print(f"Saved rectified COCO to: {output_json}")


def main():
    # Usa le variabili di configurazione definite in testa al file.
    input_json = os.path.abspath(INPUT_JSON) if INPUT_JSON else ""
    if not input_json:
        raise ValueError("Imposta INPUT_JSON in testa al file.")

    if OUTPUT_JSON:
        output_json = os.path.abspath(OUTPUT_JSON)
    else:
        base, ext = os.path.splitext(input_json)
        output_json = base + "_rectified" + ext

    rectify_coco(input_json, output_json, alpha=ALPHA)


if __name__ == "__main__":
    main()