"""
1_generar_dataset.py
════════════════════
Genera automáticamente un dataset imagen+máscara binaria desde
uno o varios videos del campo, usando el índice ExGR como
etiquetador inicial (sin necesidad de anotar a mano).

Estructura de salida:
  dataset/
    images/   ← frames RGB  (PNG)
    masks/    ← máscaras binarias: 255=suelo/surco, 0=vegetación (PNG)

Flujo completo:
  1. python 1_generar_dataset.py video.mp4     → crea dataset/
  2. (Opcional) corregir máscaras en editor de imagen
  3. python 2_entrenar.py --data dataset        → entrena modelo .pth + .onnx
  4. python 3_inferir_video.py video.mp4        → genera video anotado

Uso:
  python 1_generar_dataset.py video.mp4 [--skip 5] [--max 300] [--out dataset]
"""

import cv2
import numpy as np
import argparse
from pathlib import Path


# ── Parámetros ExGR ────────────────────────────────────────────────
ROI_TOP_FRAC   = 0.30   # ignorar cielo
EXGR_THRESH    = None   # None = Otsu automático
MIN_SOIL_RATIO = 0.05   # descartar frames casi sin tierra
MAX_SOIL_RATIO = 0.70   # descartar frames casi sin plantas


def calcular_exgr(bgr):
    f = bgr.astype(np.float32)
    B, G, R = f[:,:,0], f[:,:,1], f[:,:,2]
    tot = R + G + B + 1e-6
    r, g, b = R/tot, G/tot, B/tot
    return (2*g - r - b) - (1.3*r - g)


def mascara_suelo(bgr, thresh=None):
    exgr = calcular_exgr(bgr)
    norm = cv2.normalize(exgr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if thresh is None:
        _, veg = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, veg = cv2.threshold(norm, int(thresh), 255, cv2.THRESH_BINARY)
    soil = cv2.bitwise_not(veg)
    h = bgr.shape[0]
    soil[:int(h * ROI_TOP_FRAC), :] = 0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    soil = cv2.morphologyEx(soil, cv2.MORPH_OPEN,  k, iterations=2)
    soil = cv2.morphologyEx(soil, cv2.MORPH_CLOSE, k, iterations=3)
    return soil


def procesar_video(video_path, out_dir, skip, max_frames, idx_start):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [!] No se pudo abrir {video_path.name}")
        return idx_start

    total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fi        = 0
    guardados = 0
    idx       = idx_start
    print(f"  {video_path.name}  ({total} frames totales)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fi += 1
        if (fi - 1) % skip != 0:
            continue
        if max_frames and (idx - idx_start) >= max_frames:
            break

        mask  = mascara_suelo(frame, EXGR_THRESH)
        ratio = float(mask.sum()) / 255.0 / float(mask.size)
        if ratio < MIN_SOIL_RATIO or ratio > MAX_SOIL_RATIO:
            continue

        name = f"{idx:06d}"
        cv2.imwrite(str(out_dir / "images" / f"{name}.png"), frame)
        cv2.imwrite(str(out_dir / "masks"  / f"{name}.png"), mask)
        idx       += 1
        guardados += 1

    cap.release()
    print(f"  => {guardados} pares guardados (indices {idx_start}..{idx-1})")
    return idx


def main():
    parser = argparse.ArgumentParser(
        description="Genera dataset imagen+mascara para entrenamiento CNN de surco")
    parser.add_argument("videos", nargs="+", help="Archivos de video de entrada")
    parser.add_argument("--skip", type=int,  default=5,
                        help="Usar 1 de cada N frames  (default: 5)")
    parser.add_argument("--max",  type=int,  default=0,
                        help="Max frames por video  (0=sin limite)")
    parser.add_argument("--out",  default="dataset",
                        help="Carpeta de salida  (default: dataset)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Destino : {out_dir.resolve()}")
    print(f"[INFO] Skip    : {args.skip}   Max/video: {args.max or 'sin limite'}\n")

    idx = 0
    for vpath in args.videos:
        p = Path(vpath)
        if not p.exists():
            print(f"[!] No encontrado: {p}")
            continue
        idx = procesar_video(p, out_dir, args.skip, args.max, idx)

    print(f"\n[OK] Dataset listo: {idx} pares en '{out_dir}'")
    print(f"     Siguiente paso -> python 2_entrenar.py --data {args.out}")


if __name__ == "__main__":
    main()
