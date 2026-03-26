import sys
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Configuración de umbrales
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    fft_fake_threshold: float = 2.6
    fft_real_threshold: float = 4.5
    fft_fake_penalty:   float = 40.0
    fft_real_bonus:     float = 30.0

    laplacian_edge_zone_pct:  float = 0.20
    laplacian_blur_threshold: float = 300.0
    laplacian_penalty:        float = 20.0

    motion_blur_threshold: float = 150.0
    motion_bonus:          float = 15.0

    static_bg_buffer_size: int   = 270
    static_bg_threshold:   float = 0.9
    static_bg_penalty:     float = 50.0

    score_authentic: float = 15.0
    score_deepfake:  float = 35.0


@dataclass
class FrameResult:
    fft_variance:      float = 0.0
    laplacian_score:   float = 0.0
    motion_blur_score: float = 0.0
    bg_variation:      float = 0.0
    total_score:       float = 0.0
    face_detected:     bool  = False
    label:             str   = "DESCONOCIDO"
    details:           list  = field(default_factory=list)

def analyze_fft(face_roi: np.ndarray, cfg: Config) -> tuple[float, float, str]:
    gray      = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if face_roi.ndim == 3 else face_roi
    fft_shift = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.log1p(np.abs(fft_shift))
    variance  = float(np.var(magnitude))

    if variance < cfg.fft_fake_threshold:
        return variance, cfg.fft_fake_penalty, f"FFT aplanada ({variance:.2f} < {cfg.fft_fake_threshold}) -> IA detectada  +{cfg.fft_fake_penalty:.0f} pts"
    elif variance > cfg.fft_real_threshold:
        return variance, -cfg.fft_real_bonus, f"FFT caotica ({variance:.2f} > {cfg.fft_real_threshold}) -> camara real  -{cfg.fft_real_bonus:.0f} pts"
    else:
        return variance, 0.0, f"FFT ambigua ({variance:.2f}) -> zona gris  0 pts"

def analyze_laplacian(face_roi: np.ndarray, cfg: Config) -> tuple[float, float, str]:
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if face_roi.ndim == 3 else face_roi
    h    = gray.shape[0]
    edge_start = int(h * (1.0 - cfg.laplacian_edge_zone_pct))
    edge_zone  = gray[edge_start:, :]

    if edge_zone.shape[0] < 4:
        return 0.0, 0.0, "Laplaciano: ROI demasiado pequeno  0 pts"

    variance = float(cv2.Laplacian(edge_zone, cv2.CV_64F).var())
    if variance < cfg.laplacian_blur_threshold:
        return variance, cfg.laplacian_penalty, f"Borde cuello difuso ({variance:.0f} < {cfg.laplacian_blur_threshold}) -> GAN  +{cfg.laplacian_penalty:.0f} pts"
    else:
        return variance, 0.0, f"Borde cuello nitido ({variance:.0f}) -> sin penalizacion  0 pts"

def analyze_motion(frame: np.ndarray, prev_frame: Optional[np.ndarray], bg_variation_buffer: list[float], has_face: bool, face_bbox: Optional[tuple], cfg: Config) -> tuple[float, float, float, str]:
    gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    delta      = 0.0
    msgs: list[str] = []

    if blur_score < cfg.motion_blur_threshold and has_face:
        delta -= cfg.motion_bonus
        msgs.append(f"Camara temblorosa ({blur_score:.0f}) + cara -> natural  -{cfg.motion_bonus:.0f} pts")

    bg_var = 0.0
    if prev_frame is not None and has_face and face_bbox is not None:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if prev_frame.ndim == 3 else prev_frame
        mask = np.ones(gray.shape, dtype=np.uint8) * 255
        x, y, w, h = face_bbox
        pad_x = int(w * 0.2);  pad_y = int(h * 0.2)
        x0 = max(0, x - pad_x);          y0 = max(0, y - pad_y)
        x1 = min(gray.shape[1], x+w+pad_x); y1 = min(gray.shape[0], y+h+pad_y)
        mask[y0:y1, x0:x1] = 0
        diff   = cv2.absdiff(gray, prev_gray)
        bg_var = float(np.mean(diff[mask > 0]))
        bg_variation_buffer.append(bg_var)

    if has_face and len(bg_variation_buffer) >= cfg.static_bg_buffer_size:
        avg_bg = float(np.mean(bg_variation_buffer[-cfg.static_bg_buffer_size:]))
        if avg_bg < cfg.static_bg_threshold:
            delta += cfg.static_bg_penalty
            msgs.append(f"Fondo congelado ({avg_bg:.3f} px) -> CLON-AVATAR  +{cfg.static_bg_penalty:.0f} pts")

    if not msgs: msgs.append("Motion blur normal")
    return blur_score, bg_var, delta, " | ".join(msgs)

def classify(score: float, cfg: Config) -> str:
    if score < cfg.score_authentic: return "AUTENTICO"
    elif score > cfg.score_deepfake: return "DEEPFAKE"
    return "SOSPECHOSO"

def detect_deepfake_frame(frame: np.ndarray, face_detector, prev_frame: Optional[np.ndarray], bg_variation_buffer: list[float], cfg: Config) -> FrameResult:
    result = FrameResult()
    score  = 0.0
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
    if len(faces) == 0:
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=2, minSize=(40, 40))
    if len(faces) == 0:
        faces = face_detector.detectMultiScale(cv2.equalizeHist(gray), scaleFactor=1.05, minNeighbors=2, minSize=(40, 40))

    has_face  = len(faces) > 0
    face_bbox = None

    if has_face:
        result.face_detected = True
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_bbox   = (x, y, w, h)
        face_roi    = frame[y:y+h, x:x+w]

        fft_var, d1, m1 = analyze_fft(face_roi, cfg)
        result.fft_variance = fft_var
        score += d1
        result.details.append(m1)

        lap_var, d2, m2 = analyze_laplacian(face_roi, cfg)
        result.laplacian_score = lap_var
        score += d2
        result.details.append(m2)
    else:
        result.details.append("Sin rostro — frame neutro (score=0)")

    blur_s, bg_var, d3, m3 = analyze_motion(frame, prev_frame, bg_variation_buffer, has_face, face_bbox, cfg)
    result.motion_blur_score = blur_s
    result.bg_variation      = bg_var
    score += d3
    result.details.append(m3)

    result.total_score = max(0.0, score)
    result.label       = classify(result.total_score, cfg)
    return result

if __name__ == "__main__":
    VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else "backend/data/video real 6.mp4"
    MAX_FRAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 200

    cfg      = Config()
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    cap        = cv2.VideoCapture(VIDEO_PATH)
    fps        = cap.get(cv2.CAP_PROP_FPS) or 30.0
    buf: list[float] = []
    prev       = None
    scores     = []
    fft_vals   = []
    lap_vals   = []
    frames_con_cara = 0
    frame_idx  = 0

    print(f"\n{'='*70}")
    print(f"DIAGNÓSTICO: {VIDEO_PATH}")
    print(f"FPS detectado: {fps:.1f}  |  Analizando hasta {MAX_FRAMES} frames")
    print(f"{'='*70}")
    print(f"{'Frame':>6} | {'Cara':>4} | {'Score':>6} | {'FFT var':>8} | {'Lap cuello':>10} | {'bg_var':>7} | Detalle")
    print("-"*90)

    while cap.isOpened() and frame_idx < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret: break

        result = detect_deepfake_frame(frame, detector, prev, buf, cfg)
        scores.append(result.total_score)
        if result.face_detected:
            frames_con_cara += 1
            fft_vals.append(result.fft_variance)
            lap_vals.append(result.laplacian_score)

        cara_str = "SI" if result.face_detected else "NO"
        detalle = result.details[0] if result.details else ""

        print(f"{frame_idx:>6} | {cara_str:>4} | {result.total_score:>6.1f} | "
              f"{result.fft_variance:>8.3f} | {result.laplacian_score:>10.0f} | "
              f"{result.bg_variation:>7.3f} | {detalle}")

        prev       = frame.copy()
        frame_idx += 1

    cap.release()

    scores_arr = np.array(scores)
    # Filter to only scores where a face was detected, because standard score is 0 and it ruins mean/median calculations
    scores_con_cara = np.array([scores[i] for i in range(len(scores)) if i < len(scores) and scores[i] != 0])
    if len(scores_con_cara) == 0:
        scores_con_cara = scores_arr
    
    print(f"\n{'─'*70}")
    print(f"RESUMEN  ({frame_idx} frames, {frames_con_cara} con cara)")
    print(f"{'─'*70}")
    print(f"Score  →  min={scores_arr.min():.1f}  media={scores_arr.mean():.1f}  "
          f"mediana={np.median(scores_arr):.1f}  p85={np.percentile(scores_arr,85):.1f}  "
          f"max={scores_arr.max():.1f}")

    if len(fft_vals):
        fft_arr = np.array(fft_vals)
        print(f"FFT    →  min={fft_arr.min():.3f}  media={fft_arr.mean():.3f}  "
              f"mediana={np.median(fft_arr):.3f}  max={fft_arr.max():.3f}")
        print(f"         Frames con FFT < 2.6 (IA):  {(fft_arr < 2.6).sum()} / {len(fft_arr)}")
        print(f"         Frames con FFT > 4.5 (real): {(fft_arr > 4.5).sum()} / {len(fft_arr)}")

    if len(lap_vals):
        lap_arr = np.array(lap_vals)
        print(f"Lap    →  min={lap_arr.min():.0f}  media={lap_arr.mean():.0f}  "
              f"mediana={np.median(lap_arr):.0f}  max={lap_arr.max():.0f}")
        print(f"         Frames con Lap < 300 (difuso): {(lap_arr < 300).sum()} / {len(lap_arr)}")

    print(f"\nDistribución de scores:")
    for umbral in [0, 10, 20, 35, 50, 60, 80, 100]:
        pct = (scores_arr >= umbral).sum()
        print(f"  >= {umbral:>3} pts : {pct:>4} frames  ({pct/max(1,frame_idx)*100:.1f}%)")
