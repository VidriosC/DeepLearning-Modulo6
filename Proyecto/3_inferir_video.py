"""
3_inferir_video.py
══════════════════
Inferencia en video usando el modelo ONNX exportado por 2_entrenar.py.
Combina la máscara CNN con el pipeline de detección de línea del
surco_detector.py (columnas + Kalman) y dibuja la línea guía.

Optimizaciones para Raspberry Pi 5:
  - Inferencia ONNX con onnxruntime (CPU, sin dependencia de CUDA)
  - Resolución de inferencia configurable (320px recomendado)
  - Skip de frames + media móvil de máscara entre frames

Fallback automático:
  Si no se cuenta con modelo ONNX, vuelve al método ExGR clásico.

Uso:
  python 3_inferir_video.py video.mp4
  python 3_inferir_video.py video.mp4 --model modelo_surco.onnx
  python 3_inferir_video.py video.mp4 --model modelo_surco.onnx --output resultado.mp4
  python 3_inferir_video.py video.mp4 --model modelo_surco.onnx --debug
  python 3_inferir_video.py video.mp4 --fallback       # forzar ExGR clásico
"""

import cv2
import numpy as np
import argparse
import sys
import time
from pathlib import Path


# ── Constantes de visualización ────────────────────────────────────
COLOR_LINE   = (0,   0,   255)
COLOR_START  = (0,   255, 255)
COLOR_TARGET = (0,   150, 255)
LINE_THICK   = 3
START_FRAC   = (0.50, 0.95)
ROI_TOP_FRAC = 0.35
MIN_SOIL_PX  = 10
KALMAN_Q     = 1e-2
KALMAN_R     = 1e-1

# ── Normalización ImageNet ─────────────────────────────────────────
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════
#  KALMAN 1-D
# ══════════════════════════════════════════════════════════════════

class KalmanX:
    def __init__(self, x0):
        self.kf = cv2.KalmanFilter(2, 1)
        self.kf.transitionMatrix    = np.array([[1,1],[0,1]], np.float32)
        self.kf.measurementMatrix   = np.array([[1,0]],       np.float32)
        self.kf.processNoiseCov     = np.eye(2, dtype=np.float32) * KALMAN_Q
        self.kf.measurementNoiseCov = np.eye(1, dtype=np.float32) * KALMAN_R
        self.kf.errorCovPost        = np.eye(2, dtype=np.float32)
        self.kf.statePost           = np.array([[x0],[0]], np.float32)

    def update(self, x):
        self.kf.predict()
        state = self.kf.correct(np.array([[x]], np.float32))
        return float(state[0])

    def predict_only(self):
        return float(self.kf.predict()[0])


# ══════════════════════════════════════════════════════════════════
#  FALLBACK ExGR (cuando no hay modelo ONNX)
# ══════════════════════════════════════════════════════════════════

def exgr_mask(bgr):
    f = bgr.astype(np.float32)
    B, G, R = f[:,:,0], f[:,:,1], f[:,:,2]
    tot = R + G + B + 1e-6
    r, g, b = R/tot, G/tot, B/tot
    exgr = (2*g - r - b) - (1.3*r - g)
    norm = cv2.normalize(exgr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, veg = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    soil = cv2.bitwise_not(veg)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    soil = cv2.morphologyEx(soil, cv2.MORPH_OPEN,  k, iterations=2)
    soil = cv2.morphologyEx(soil, cv2.MORPH_CLOSE, k, iterations=3)
    return soil


# ══════════════════════════════════════════════════════════════════
#  INFERENCIA ONNX
# ══════════════════════════════════════════════════════════════════

def cargar_onnx(model_path, inf_size):
    try:
        import onnxruntime as ort
    except ImportError:
        print("[WARN] onnxruntime no instalado. Usando fallback ExGR.")
        print("       pip install onnxruntime")
        return None, None

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4   # usar 4 núcleos del RPi 5
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess = ort.InferenceSession(model_path,
                                 sess_options=sess_opts,
                                 providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    print(f"[INFO] Modelo ONNX cargado: {model_path}")
    print(f"[INFO] Input: {inp_name}  |  Inf size: {inf_size}x{inf_size}")
    return sess, inp_name


def inferir_mascara_onnx(sess, inp_name, frame, inf_size):
    """Preprocesa, corre inferencia ONNX, devuelve máscara (h_orig, w_orig) uint8."""
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (inf_size, inf_size)).astype(np.float32) / 255.0
    tensor  = ((resized - MEAN) / STD).transpose(2, 0, 1)[np.newaxis]  # (1,3,H,W)

    logits = sess.run(None, {inp_name: tensor})[0]   # (1,2,H,W)
    pred   = np.argmax(logits[0], axis=0).astype(np.uint8)  # (H,W) 0=veg, 1=suelo

    # Redimensionar a resolución original
    mask = cv2.resize(pred * 255, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask


# ══════════════════════════════════════════════════════════════════
#  DETECCIÓN DE LÍNEA DEL SURCO  (a partir de máscara binaria)
# ══════════════════════════════════════════════════════════════════

def detectar_linea_surco(mask, roi_top_frac=ROI_TOP_FRAC):
    """
    Dado una máscara binaria de suelo (255), extrae la posición X
    del surco central por proyección de columnas ponderada.
    """
    h, w = mask.shape
    roi_y0 = int(h * roi_top_frac)
    roi    = mask[roi_y0:, :]

    col_sum = roi.sum(axis=0) / 255.0
    valid   = np.where(col_sum >= MIN_SOIL_PX)[0]
    if len(valid) == 0:
        return w // 2, roi_y0

    cx = int(np.average(valid, weights=col_sum[valid]))
    return cx, roi_y0


def dibujar_guia(frame, cx_kalman, roi_y0, metodo="CNN", angulo=0.0, fps_str=""):
    h, w   = frame.shape[:2]
    start_x = int(w * START_FRAC[0])
    start_y = int(h * START_FRAC[1])
    target_x = cx_kalman
    target_y = roi_y0 + int((h - roi_y0) * 0.12)

    cv2.line(frame, (0, roi_y0), (w, roi_y0), (60, 60, 60), 1)
    cv2.line(frame, (start_x, start_y), (target_x, target_y),
             COLOR_LINE, LINE_THICK, cv2.LINE_AA)
    cv2.circle(frame, (start_x,  start_y),  9, COLOR_START,  -1)
    cv2.circle(frame, (target_x, target_y), 7, COLOR_TARGET, -1)

    dir_txt = "RECTO"
    if   angulo >  5: dir_txt = f"DERECHA  {angulo:.1f}deg"
    elif angulo < -5: dir_txt = f"IZQUIERDA {abs(angulo):.1f}deg"

    cv2.putText(frame, f"Surco: {dir_txt}", (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, COLOR_LINE, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Metodo: {metodo}  X={cx_kalman}px  {fps_str}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Inferencia CNN MobileNetV3 - deteccion de surco en cana de azucar")
    parser.add_argument("video",           help="Video de entrada")
    parser.add_argument("--model",         help="Ruta al modelo .onnx", default=None)
    parser.add_argument("--size",  type=int, default=320,
                        help="Resolucion de inferencia (default 320)")
    parser.add_argument("--output",        help="Video de salida", default=None)
    parser.add_argument("--skip",  type=int, default=1,
                        help="Procesar 1/N frames (default 1)")
    parser.add_argument("--debug",         action="store_true",
                        help="Mostrar ventana con mascara de suelo")
    parser.add_argument("--fallback",      action="store_true",
                        help="Forzar metodo ExGR (ignorar modelo ONNX)")
    parser.add_argument("--no-display",    action="store_true",
                        help="Sin ventana de video (util con --output)")
    args = parser.parse_args()

    path = Path(args.video)
    if not path.exists():
        print(f"[ERROR] No encontrado: {path}"); sys.exit(1)

    cap   = cv2.VideoCapture(str(path))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video: {path.name}  {W}x{H}  {fps:.1f}fps  {total}f")

    # ── Cargar modelo ONNX ────────────────────────────────────────
    sess = inp_name = None
    if not args.fallback and args.model:
        if Path(args.model).exists():
            sess, inp_name = cargar_onnx(args.model, args.size)
        else:
            print(f"[WARN] Modelo no encontrado: {args.model}. Usando ExGR.")

    metodo = "CNN-MobileNetV3" if sess else "ExGR-clasico"
    print(f"[INFO] Metodo activo: {metodo}\n")

    # ── Writer ────────────────────────────────────────────────────
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc,
                                  fps / max(args.skip, 1), (W, H))
        print(f"[INFO] Guardando en: {args.output}")

    kalman = KalmanX(W / 2)
    fi = proc = 0

    # Para suavizado temporal de máscara (reduce parpadeo en CNN)
    mask_smooth = None
    MASK_ALPHA  = 0.4   # peso del frame actual vs historia

    t_inf_total = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fi += 1
        if (fi - 1) % args.skip != 0:
            continue
        proc += 1

        t0 = time.perf_counter()

        # ── Obtener máscara ───────────────────────────────────────
        if sess:
            mask_raw = inferir_mascara_onnx(sess, inp_name, frame, args.size)
        else:
            mask_raw = exgr_mask(frame)

        # Suavizado temporal de máscara
        if mask_smooth is None:
            mask_smooth = mask_raw.astype(np.float32)
        else:
            mask_smooth = MASK_ALPHA * mask_raw.astype(np.float32) \
                        + (1 - MASK_ALPHA) * mask_smooth
        mask_bin = (mask_smooth > 127).astype(np.uint8) * 255

        # ── Detectar posición del surco ───────────────────────────
        cx_raw, roi_y0 = detectar_linea_surco(mask_bin)
        cx = int(kalman.update(cx_raw))
        cx = int(np.clip(cx, 0, W - 1))

        inf_ms = (time.perf_counter() - t0) * 1000
        t_inf_total += inf_ms

        # ── Ángulo de corrección ──────────────────────────────────
        start_x = int(W * START_FRAC[0])
        start_y = int(H * START_FRAC[1])
        target_y = roi_y0 + int((H - roi_y0) * 0.12)
        dx = cx - start_x
        dy = start_y - target_y
        angulo = float(np.degrees(np.arctan2(dx, dy + 1e-6)))

        # ── Dibujar ───────────────────────────────────────────────
        frame_out = frame.copy()
        fps_str = f"{1000/inf_ms:.1f}fps" if inf_ms > 0 else ""
        dibujar_guia(frame_out, cx, roi_y0, metodo, angulo, fps_str)

        # Overlay semitransparente de la máscara (azul = suelo)
        if args.debug:
            overlay       = frame_out.copy()
            soil_overlay  = np.zeros_like(frame_out)
            soil_overlay[mask_bin > 0] = (180, 80, 0)   # azul oscuro = suelo
            frame_out = cv2.addWeighted(frame_out, 0.75, soil_overlay, 0.25, 0)

        # ── Mostrar / guardar ─────────────────────────────────────
        if not args.no_display:
            cv2.imshow("Surco CNN - Cana de Azucar", frame_out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Salida por usuario.")
                break

        if writer:
            writer.write(frame_out)

        if proc % 50 == 0:
            pct = fi / total * 100 if total else 0
            print(f"  f{fi}/{total} ({pct:.0f}%)  "
                  f"ang={angulo:+.1f}  x={cx}px  {inf_ms:.1f}ms/frame")

    cap.release()
    if writer:
        writer.release()
        print(f"\n[OK] Video guardado: {args.output}")
    cv2.destroyAllWindows()

    avg_inf = t_inf_total / max(proc, 1)
    print(f"[INFO] Frames procesados: {proc}")
    print(f"[INFO] Tiempo medio inferencia: {avg_inf:.1f} ms/frame  "
          f"({1000/avg_inf:.1f} FPS estimado)")
    print("[INFO] Fin.")


if __name__ == "__main__":
    main()
