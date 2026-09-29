
"""
4_navegacion_voz.py
═══════════════════

Sistema integrado de navegación autónoma para cultivo de caña de azúcar.

Integra:

1. Detección del centro del surco:
   - CNN MobileNetV3 mediante modelo ONNX.
   - Fallback automático a ExGR clásico.
   - Filtro Kalman para suavizar la posición.
   - Cálculo del ángulo de corrección.
   - Visualización de la línea guía.

2. Reconocimiento de comandos de voz:
   - Modelo KWS: superb/wav2vec2-base-superb-ks
   - GO / ON   -> iniciar o reanudar inferencia
   - OFF / DOWN -> pausar inferencia
   - STOP      -> detener completamente el programa

Estados:

    ESPERANDO
        |
        | GO / ON
        v
    ACTIVO <-------- GO / ON -------- PAUSA
        |
        | OFF / DOWN
        v
      PAUSA

    Desde ACTIVO o PAUSA:
        STOP -> TERMINAR

Uso:

    python 4_navegacion_voz.py video.mp4 --model modelo_surco.onnx

Opciones:

    --model modelo_surco.onnx
    --size 320
    --skip 1
    --output resultado.mp4
    --debug
    --fallback
    --no-display

Dependencias:

    pip install opencv-python numpy onnxruntime
    pip install sounddevice torch transformers
"""

import cv2
import numpy as np
import argparse
import sys
import time
import threading
from pathlib import Path

import sounddevice as sd
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


# ══════════════════════════════════════════════════════════════════
# CONFIGURACIÓN GENERAL
# ══════════════════════════════════════════════════════════════════

COLOR_LINE   = (0, 0, 255)
COLOR_START  = (0, 255, 255)
COLOR_TARGET = (0, 150, 255)

LINE_THICK = 3

START_FRAC = (0.50, 0.95)
ROI_TOP_FRAC = 0.35
MIN_SOIL_PX = 10

KALMAN_Q = 1e-2
KALMAN_R = 1e-1

# Normalización ImageNet
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DEL RECONOCIMIENTO DE VOZ
# ══════════════════════════════════════════════════════════════════

MODEL_NAME = "superb/wav2vec2-base-superb-ks"

SAMPLE_RATE = 16000

WINDOW_DURATION = 1.0
CHUNK_SIZE = int(SAMPLE_RATE * WINDOW_DURATION)

HOP_DURATION = 0.3
HOP_SIZE = int(SAMPLE_RATE * HOP_DURATION)

CONFIDENCE_THRESHOLD = 0.85
COOLDOWN_SECONDS = 1.2

COMMAND_MAP = {
    "go": "EMPEZAR",
    "on": "EMPEZAR",
    "off": "PAUSA",
    "down": "PAUSA",
    "stop": "DETENER"
}


# ══════════════════════════════════════════════════════════════════
# ESTADO GLOBAL DEL SISTEMA
# ══════════════════════════════════════════════════════════════════

estado = "ESPERANDO"

# Lock para proteger el acceso al estado desde:
# - hilo principal de video
# - callback de audio
estado_lock = threading.Lock()

# Evento para terminar el programa desde el hilo de voz
terminar_evento = threading.Event()

audio_buffer = np.zeros(CHUNK_SIZE, dtype=np.float32)
last_detection_time = 0.0


# ══════════════════════════════════════════════════════════════════
# FUNCIONES DE ESTADO
# ══════════════════════════════════════════════════════════════════

def obtener_estado():
    """Devuelve el estado actual de forma segura."""
    with estado_lock:
        return estado


def cambiar_estado(nuevo_estado):
    """Cambia el estado global de forma segura."""
    global estado

    with estado_lock:
        estado = nuevo_estado

    print(f"[ESTADO] >>> {nuevo_estado} <<<")


# ══════════════════════════════════════════════════════════════════
# KALMAN 1-D
# ══════════════════════════════════════════════════════════════════

class KalmanX:

    def __init__(self, x0):

        self.kf = cv2.KalmanFilter(2, 1)

        self.kf.transitionMatrix = np.array(
            [[1, 1],
             [0, 1]],
            np.float32
        )

        self.kf.measurementMatrix = np.array(
            [[1, 0]],
            np.float32
        )

        self.kf.processNoiseCov = (
            np.eye(2, dtype=np.float32) * KALMAN_Q
        )

        self.kf.measurementNoiseCov = (
            np.eye(1, dtype=np.float32) * KALMAN_R
        )

        self.kf.errorCovPost = np.eye(
            2,
            dtype=np.float32
        )

        self.kf.statePost = np.array(
            [[x0],
             [0]],
            np.float32
        )

    def update(self, x):

        self.kf.predict()

        state = self.kf.correct(
            np.array([[x]], np.float32)
        )

        return float(state[0])

    def predict_only(self):

        return float(
            self.kf.predict()[0]
        )


# ══════════════════════════════════════════════════════════════════
# FALLBACK ExGR
# ══════════════════════════════════════════════════════════════════

def exgr_mask(bgr):

    f = bgr.astype(np.float32)

    B = f[:, :, 0]
    G = f[:, :, 1]
    R = f[:, :, 2]

    tot = R + G + B + 1e-6

    r = R / tot
    g = G / tot
    b = B / tot

    exgr = (2 * g - r - b) - (1.3 * r - g)

    norm = cv2.normalize(
        exgr,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    _, veg = cv2.threshold(
        norm,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    soil = cv2.bitwise_not(veg)

    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7)
    )

    soil = cv2.morphologyEx(
        soil,
        cv2.MORPH_OPEN,
        k,
        iterations=2
    )

    soil = cv2.morphologyEx(
        soil,
        cv2.MORPH_CLOSE,
        k,
        iterations=3
    )

    return soil


# ══════════════════════════════════════════════════════════════════
# CARGAR MODELO ONNX
# ══════════════════════════════════════════════════════════════════

def cargar_onnx(model_path, inf_size):

    try:

        import onnxruntime as ort

    except ImportError:

        print(
            "[WARN] onnxruntime no instalado."
        )

        print(
            "       pip install onnxruntime"
        )

        return None, None

    sess_opts = ort.SessionOptions()

    sess_opts.intra_op_num_threads = 4

    sess_opts.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    sess = ort.InferenceSession(
        model_path,
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"]
    )

    inp_name = sess.get_inputs()[0].name

    print(
        f"[INFO] Modelo ONNX cargado: {model_path}"
    )

    print(
        f"[INFO] Input: {inp_name}"
    )

    print(
        f"[INFO] Resolución: {inf_size}x{inf_size}"
    )

    return sess, inp_name


# ══════════════════════════════════════════════════════════════════
# INFERENCIA CNN
# ══════════════════════════════════════════════════════════════════

def inferir_mascara_onnx(
    sess,
    inp_name,
    frame,
    inf_size
):

    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    resized = cv2.resize(
        rgb,
        (inf_size, inf_size)
    ).astype(np.float32) / 255.0

    tensor = (
        (resized - MEAN) / STD
    ).transpose(2, 0, 1)[np.newaxis]

    logits = sess.run(
        None,
        {inp_name: tensor}
    )[0]

    pred = np.argmax(
        logits[0],
        axis=0
    ).astype(np.uint8)

    mask = cv2.resize(
        pred * 255,
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    return mask


# ══════════════════════════════════════════════════════════════════
# DETECCIÓN DEL CENTRO DEL SURCO
# ══════════════════════════════════════════════════════════════════

def detectar_linea_surco(
    mask,
    roi_top_frac=ROI_TOP_FRAC
):

    h, w = mask.shape

    roi_y0 = int(
        h * roi_top_frac
    )

    roi = mask[roi_y0:, :]

    col_sum = (
        roi.sum(axis=0) / 255.0
    )

    valid = np.where(
        col_sum >= MIN_SOIL_PX
    )[0]

    if len(valid) == 0:

        return w // 2, roi_y0

    cx = int(
        np.average(
            valid,
            weights=col_sum[valid]
        )
    )

    return cx, roi_y0


# ══════════════════════════════════════════════════════════════════
# DIBUJAR INFORMACIÓN DE NAVEGACIÓN
# ══════════════════════════════════════════════════════════════════

def dibujar_guia(
    frame,
    cx_kalman,
    roi_y0,
    metodo,
    angulo,
    fps_str,
    estado_actual
):

    h, w = frame.shape[:2]

    start_x = int(
        w * START_FRAC[0]
    )

    start_y = int(
        h * START_FRAC[1]
    )

    target_x = cx_kalman

    target_y = (
        roi_y0
        + int((h - roi_y0) * 0.12)
    )

    # Línea horizontal de inicio del ROI
    cv2.line(
        frame,
        (0, roi_y0),
        (w, roi_y0),
        (60, 60, 60),
        1
    )

    # Línea guía
    cv2.line(
        frame,
        (start_x, start_y),
        (target_x, target_y),
        COLOR_LINE,
        LINE_THICK,
        cv2.LINE_AA
    )

    # Punto de inicio
    cv2.circle(
        frame,
        (start_x, start_y),
        9,
        COLOR_START,
        -1
    )

    # Punto objetivo
    cv2.circle(
        frame,
        (target_x, target_y),
        7,
        COLOR_TARGET,
        -1
    )

    # Dirección
    dir_txt = "RECTO"

    if angulo > 5:

        dir_txt = (
            f"DERECHA {angulo:.1f}deg"
        )

    elif angulo < -5:

        dir_txt = (
            f"IZQUIERDA {abs(angulo):.1f}deg"
        )

    cv2.putText(
        frame,
        f"Surco: {dir_txt}",
        (10, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        COLOR_LINE,
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"Metodo: {metodo}  X={cx_kalman}px  {fps_str}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1,
        cv2.LINE_AA
    )

    # Estado
    cv2.putText(
        frame,
        f"ESTADO: {estado_actual}",
        (10, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


# ══════════════════════════════════════════════════════════════════
# RECONOCIMIENTO DE VOZ
# ══════════════════════════════════════════════════════════════════

def cargar_modelo_voz():

    print(
        "\n[VOZ] Cargando modelo KWS..."
    )

    feature_extractor = (
        AutoFeatureExtractor.from_pretrained(
            MODEL_NAME
        )
    )

    model = (
        AutoModelForAudioClassification
        .from_pretrained(MODEL_NAME)
    )

    model.eval()

    print(
        "[VOZ] Modelo KWS cargado."
    )

    return feature_extractor, model


# ══════════════════════════════════════════════════════════════════
# CALLBACK DE AUDIO
# ══════════════════════════════════════════════════════════════════

def audio_callback(
    indata,
    frames,
    time_info,
    status
):

    global audio_buffer
    global last_detection_time

    if status:

        print(
            f"[VOZ] Estado de audio: {status}"
        )

    # Agregar audio al buffer
    n = len(indata)

    if n >= CHUNK_SIZE:

        audio_buffer[:] = (
            indata[-CHUNK_SIZE:, 0]
        )

    else:

        audio_buffer = np.roll(
            audio_buffer,
            -n
        )

        audio_buffer[-n:] = (
            indata[:, 0]
        )

    # VAD muy simple
    if np.max(
        np.abs(audio_buffer)
    ) < 0.03:

        return

    current_time = time.time()

    # Cooldown
    if (
        current_time - last_detection_time
        < COOLDOWN_SECONDS
    ):

        return

    try:

        inputs = feature_extractor(
            audio_buffer,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt"
        )

        with torch.no_grad():

            logits = model(
                **inputs
            ).logits

        probs = torch.softmax(
            logits,
            dim=-1
        )

        top_prob, top_class_id = (
            torch.max(
                probs,
                dim=-1
            )
        )

        label = model.config.id2label[
            top_class_id.item()
        ]

        confidence = top_prob.item()

    except Exception as e:

        print(
            f"[VOZ][ERROR] Inferencia: {e}"
        )

        return

    # Solo aceptar comandos conocidos
    if (
        confidence >= CONFIDENCE_THRESHOLD
        and label in COMMAND_MAP
    ):

        command = COMMAND_MAP[label]

        last_detection_time = current_time

        print(
            f"\n[VOZ] COMANDO: {label.upper()} "
            f"({confidence * 100:.1f}%)"
        )

        # ─────────────────────────────────────────────
        # GO / ON
        # ─────────────────────────────────────────────

        if command == "EMPEZAR":

            cambiar_estado("ACTIVO")

        # ─────────────────────────────────────────────
        # OFF / DOWN
        # ─────────────────────────────────────────────

        elif command == "PAUSA":

            cambiar_estado("PAUSA")

        # ─────────────────────────────────────────────
        # STOP
        # ─────────────────────────────────────────────

        elif command == "DETENER":

            cambiar_estado("DETENIDO")

            terminar_evento.set()


# ══════════════════════════════════════════════════════════════════
# PROGRAMA PRINCIPAL
# ══════════════════════════════════════════════════════════════════

def main():

    global feature_extractor
    global model

    parser = argparse.ArgumentParser(
        description=(
            "Navegación de surcos con "
            "control por comandos de voz"
        )
    )

    parser.add_argument(
        "video",
        help="Video de entrada"
    )

    parser.add_argument(
        "--model",
        help="Ruta al modelo ONNX",
        default=None
    )

    parser.add_argument(
        "--size",
        type=int,
        default=320,
        help="Resolución de inferencia"
    )

    parser.add_argument(
        "--output",
        help="Video de salida",
        default=None
    )

    parser.add_argument(
        "--skip",
        type=int,
        default=1,
        help="Procesar 1/N frames"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Mostrar máscara"
    )

    parser.add_argument(
        "--fallback",
        action="store_true",
        help="Forzar ExGR"
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Sin ventana"
    )

    args = parser.parse_args()

    # ═════════════════════════════════════════════════════════════
    # VALIDAR VIDEO
    # ═════════════════════════════════════════════════════════════

    path = Path(args.video)

    if not path.exists():

        print(
            f"[ERROR] Video no encontrado: {path}"
        )

        sys.exit(1)

    # ═════════════════════════════════════════════════════════════
    # ABRIR VIDEO
    # ═════════════════════════════════════════════════════════════

    cap = cv2.VideoCapture(
        str(path)
    )

    fps = (
        cap.get(cv2.CAP_PROP_FPS)
        or 30
    )

    W = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    H = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    print(
        f"[INFO] Video: {path.name} "
        f"{W}x{H} {fps:.1f}fps "
        f"{total} frames"
    )

    # ═════════════════════════════════════════════════════════════
    # MODELO ONNX
    # ═════════════════════════════════════════════════════════════

    sess = None
    inp_name = None

    if (
        not args.fallback
        and args.model
    ):

        if Path(args.model).exists():

            sess, inp_name = cargar_onnx(
                args.model,
                args.size
            )

        else:

            print(
                "[WARN] Modelo ONNX no encontrado."
            )

    metodo = (
        "CNN-MobileNetV3"
        if sess
        else
        "ExGR-clasico"
    )

    print(
        f"[INFO] Método de segmentación: {metodo}"
    )

    # ═════════════════════════════════════════════════════════════
    # MODELO DE VOZ
    # ═════════════════════════════════════════════════════════════

    feature_extractor, model = (
        cargar_modelo_voz()
    )

    # ═════════════════════════════════════════════════════════════
    # WRITER
    # ═════════════════════════════════════════════════════════════

    writer = None

    if args.output:

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            args.output,
            fourcc,
            fps,
            (W, H)
        )

        print(
            f"[INFO] Guardando video en: "
            f"{args.output}"
        )

    # ═════════════════════════════════════════════════════════════
    # KALMAN
    # ═════════════════════════════════════════════════════════════

    kalman = KalmanX(W / 2)

    # Suavizado temporal
    mask_smooth = None

    MASK_ALPHA = 0.4

    # ═════════════════════════════════════════════════════════════
    # STREAM DE AUDIO
    # ═════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)

    print(
        "SISTEMA DE NAVEGACIÓN AUTÓNOMA"
    )

    print("=" * 60)

    print(
        "Comandos:"
    )

    print(
        "  GO / ON       -> INICIAR / REANUDAR"
    )

    print(
        "  OFF / DOWN    -> PAUSAR"
    )

    print(
        "  STOP          -> DETENER"
    )

    print("=" * 60)

    print(
        "\nEstado inicial: ESPERANDO"
    )

    print(
        "Diga GO para comenzar...\n"
    )

    # ═════════════════════════════════════════════════════════════
    # BUCLE PRINCIPAL
    # ═════════════════════════════════════════════════════════════

    try:

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            callback=audio_callback,
            blocksize=HOP_SIZE
        ):

            while not terminar_evento.is_set():

                estado_actual = obtener_estado()

                # ─────────────────────────────────────────────────
                # STOP
                # ─────────────────────────────────────────────────

                if estado_actual == "DETENIDO":

                    break

                # ─────────────────────────────────────────────────
                # ESPERANDO / PAUSA
                # ─────────────────────────────────────────────────

                if (
                    estado_actual == "ESPERANDO"
                    or estado_actual == "PAUSA"
                ):

                    # No procesamos frames.
                    # El audio continúa funcionando.

                    if not args.no_display:

                        # Leer el frame actual únicamente
                        # para mostrar estado.
                        ret, frame = cap.read()

                        if not ret:

                            cap.set(
                                cv2.CAP_PROP_POS_FRAMES,
                                0
                            )

                            ret, frame = cap.read()

                        if ret:

                            texto = (
                                "ESPERANDO GO"
                                if estado_actual
                                == "ESPERANDO"
                                else
                                "PAUSA - diga GO"
                            )

                            cv2.putText(
                                frame,
                                texto,
                                (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.0,
                                (0, 255, 255),
                                2,
                                cv2.LINE_AA
                            )

                            cv2.putText(
                                frame,
                                "OFF=PAUSA  STOP=SALIR",
                                (20, 90),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.65,
                                (255, 255, 255),
                                2,
                                cv2.LINE_AA
                            )

                            cv2.imshow(
                                "Navegacion por voz - Surco",
                                frame
                            )

                            if (
                                cv2.waitKey(1)
                                & 0xFF
                                == ord("q")
                            ):

                                terminar_evento.set()
                                break

                    time.sleep(0.05)

                    continue

                # ─────────────────────────────────────────────────
                # ACTIVO
                # ─────────────────────────────────────────────────

                ret, frame = cap.read()

                if not ret:

                    print(
                        "[INFO] Fin del video."
                    )

                    break

                # Skip de frames
                frame_number = int(
                    cap.get(
                        cv2.CAP_PROP_POS_FRAMES
                    )
                )

                if (
                    args.skip > 1
                    and
                    frame_number % args.skip != 0
                ):

                    continue

                t0 = time.perf_counter()

                # ─────────────────────────────────────────────────
                # SEGMENTACIÓN
                # ─────────────────────────────────────────────────

                if sess:

                    mask_raw = (
                        inferir_mascara_onnx(
                            sess,
                            inp_name,
                            frame,
                            args.size
                        )
                    )

                else:

                    mask_raw = exgr_mask(
                        frame
                    )

                # ─────────────────────────────────────────────────
                # SUAVIZADO TEMPORAL
                # ─────────────────────────────────────────────────

                if mask_smooth is None:

                    mask_smooth = (
                        mask_raw.astype(
                            np.float32
                        )
                    )

                else:

                    mask_smooth = (
                        MASK_ALPHA
                        * mask_raw.astype(
                            np.float32
                        )
                        +
                        (1 - MASK_ALPHA)
                        * mask_smooth
                    )

                mask_bin = (
                    mask_smooth > 127
                ).astype(
                    np.uint8
                ) * 255

                # ─────────────────────────────────────────────────
                # CENTRO DEL SURCO
                # ─────────────────────────────────────────────────

                cx_raw, roi_y0 = (
                    detectar_linea_surco(
                        mask_bin
                    )
                )

                cx = int(
                    kalman.update(
                        cx_raw
                    )
                )

                cx = int(
                    np.clip(
                        cx,
                        0,
                        W - 1
                    )
                )

                # ─────────────────────────────────────────────────
                # TIEMPO DE INFERENCIA
                # ─────────────────────────────────────────────────

                inf_ms = (
                    time.perf_counter()
                    - t0
                ) * 1000

                # ─────────────────────────────────────────────────
                # ÁNGULO
                # ─────────────────────────────────────────────────

                start_x = int(
                    W * START_FRAC[0]
                )

                start_y = int(
                    H * START_FRAC[1]
                )

                target_y = (
                    roi_y0
                    + int(
                        (H - roi_y0)
                        * 0.12
                    )
                )

                dx = cx - start_x

                dy = (
                    start_y
                    - target_y
                )

                angulo = float(
                    np.degrees(
                        np.arctan2(
                            dx,
                            dy + 1e-6
                        )
                    )
                )

                # ─────────────────────────────────────────────────
                # VISUALIZACIÓN
                # ─────────────────────────────────────────────────

                frame_out = frame.copy()

                fps_str = (
                    f"{1000 / inf_ms:.1f} FPS"
                    if inf_ms > 0
                    else ""
                )

                dibujar_guia(
                    frame_out,
                    cx,
                    roi_y0,
                    metodo,
                    angulo,
                    fps_str,
                    estado_actual
                )

                # Debug
                if args.debug:

                    soil_overlay = (
                        np.zeros_like(
                            frame_out
                        )
                    )

                    soil_overlay[
                        mask_bin > 0
                    ] = (
                        180,
                        80,
                        0
                    )

                    frame_out = (
                        cv2.addWeighted(
                            frame_out,
                            0.75,
                            soil_overlay,
                            0.25,
                            0
                        )
                    )

                # ─────────────────────────────────────────────────
                # DISPLAY
                # ─────────────────────────────────────────────────

                if not args.no_display:

                    cv2.imshow(
                        "Navegacion por voz - Surco",
                        frame_out
                    )

                    key = (
                        cv2.waitKey(1)
                        & 0xFF
                    )

                    if key == ord("q"):

                        terminar_evento.set()

                        break

                # ─────────────────────────────────────────────────
                # GUARDAR
                # ─────────────────────────────────────────────────

                if writer:

                    writer.write(
                        frame_out
                    )

    except KeyboardInterrupt:

        print(
            "\n[INFO] Programa interrumpido."
        )

    except Exception as e:

        print(
            f"\n[ERROR] {e}"
        )

    finally:

        cambiar_estado(
            "DETENIDO"
        )

        terminar_evento.set()

        cap.release()

        if writer:

            writer.release()

        cv2.destroyAllWindows()

        print(
            "\n[INFO] Sistema finalizado."
        )


# ══════════════════════════════════════════════════════════════════
# EJECUCIÓN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    main()