"""
Comparacion forense: imagen sintetica IA vs foto real simulada.
Verifica que los umbrales recalibrados no generan falsos positivos.
"""
import sys, io, cv2, numpy as np
from PIL import Image

sys.path.insert(0, ".")
from models.image_detector import NueroscanEngineV5

e = object.__new__(NueroscanEngineV5)
e.config = {
    "grok_threshold": 25,
    "texture_thresholds": (12, 18),
    "noise_thresholds": (18, 28, 180),
    "fft_thresholds": (0.55, 0.70),
    "symmetry_threshold": 3.5,
    "rolloff_thresholds": (8.0, 12.0),
    "entropy_threshold": 100,
    "color_var_threshold": 500,
    "ela_threshold": 50,
    "srm_threshold": 3.0,
    "min_face_size": 50,
    "model_weights": {"detector1": 0.35, "detector2": 0.35, "clip": 0.30},
    "ensemble_weights": {"neural": 0.25, "forensic": 0.45, "grok": 0.30},
    "neural_unreliable_threshold": 60.0,
}

def analizar(nombre, arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=85)
    buf.seek(0)
    data = buf.read()
    img_pil = Image.open(io.BytesIO(data)).convert("RGB")
    img_cv  = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    gray    = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    metrics  = e._forensics(img_pil, img_cv, gray)
    rn       = e._residual_noise(gray)
    score, reasons = e._forensic_score(metrics, rn, [], gray, img_cv, has_exif_optical=False)
    print(f"\n{'='*55}")
    print(f"  {nombre}")
    print(f"  smooth_noise={metrics['smooth_noise']:.3f}  chan_corr={metrics['chan_corr']:.3f}")
    print(f"  grad_flat={metrics['grad_flat']:.3f}  grad_var={metrics['grad_var']:.1f}")
    print(f"  fft={metrics['fft_ratio']:.3f}  rn={rn:.1f}  texture={metrics['texture_score']:.1f}")
    print(f"  SCORE FORENSE: {score:.1f}")
    for r in reasons:
        print(f"    -> {r}")
    prob = 35.0 * 0.10 + score * 0.58 + 0.0 * 0.32
    if score >= 70: prob = max(prob, 72.0)
    elif score >= 55: prob = max(prob, 60.0)
    elif score >= 35: prob = max(prob, 45.0)
    print(f"  PROBABILIDAD ESTIMADA: {int(prob)}%")

# ── 1. Imagen IA pura: gradiente perfecto, sin ruido ─────────────────────────
ia_arr = np.zeros((300, 300, 3), dtype=np.uint8)
for i in range(300):
    ia_arr[i, :] = [i % 256, (i * 2) % 256, 255 - i % 256]
analizar("IA PERFECTA (gradiente sin ruido)", ia_arr)

# ── 2. Foto real simulada: ruido ISO, textura varia ──────────────────────────
rng = np.random.default_rng(42)
real_arr = rng.integers(80, 180, size=(300, 300, 3), dtype=np.uint8)
# Agregar variación natural de escena
real_arr[:150, :, 0] = np.clip(real_arr[:150, :, 0] + 40, 0, 255)  # cielo
real_arr[150:, :, 1] = np.clip(real_arr[150:, :, 1] + 30, 0, 255)  # verde pasto
analizar("FOTO REAL (ruido natural ISO)", real_arr.astype(np.uint8))

# ── 3. Foto real JPEG (con compresion agresiva) ───────────────────────────────
buf = io.BytesIO()
Image.fromarray(real_arr.astype(np.uint8)).save(buf, format="JPEG", quality=60)
buf.seek(0)
jpeg_arr = np.array(Image.open(buf))
analizar("FOTO REAL comprimida JPEG q=60 (sin EXIF)", jpeg_arr)

# ── 4. Imagen IA realista: suave, homogenea ───────────────────────────────────
ia2 = np.zeros((300, 300, 3), dtype=np.uint8)
for y in range(300):
    for x in range(300):
        ia2[y, x] = [
            int(128 + 60 * np.sin(y / 50)),
            int(128 + 40 * np.cos(x / 40)),
            int(200 - y // 3),
        ]
analizar("IA REALISTA (ondas suaves aurora-style)", ia2)
