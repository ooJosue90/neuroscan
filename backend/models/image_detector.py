import cv2
import numpy as np
import io
import torch
import time
import concurrent.futures
import logging
from PIL import Image
from PIL.ExifTags import TAGS
from transformers import pipeline
import open_clip
import warnings
import functools

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.WARNING, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class NueroscanEngineV5:
    def __init__(self, config=None):
        self.config = config or {
            "grok_threshold": 25,           # antes: 50 — más sensible a firma Aurora
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
            # Pesos del ensemble adaptativo
            "model_weights": {"detector1": 0.35, "detector2": 0.35, "clip": 0.30},
            # Pesos del ensemble final (neural, forensic, grok)
            "ensemble_weights": {"neural": 0.25, "forensic": 0.45, "grok": 0.30},
            # Umbral bajo el que los neural models se consideran «poco fiables»
            "neural_unreliable_threshold": 60.0,
        }

        # Optimización GPU: fp16 ahorra ~50% VRAM
        self.device = 0 if torch.cuda.is_available() else -1
        self.torch_dtype = torch.float16 if self.device == 0 else torch.float32
        self._models_cache = {}

        print(">>> Iniciando Nueroscan Engine V5 (Optimized Enterprise Edition) ...")

        self.detector1 = self._load_model("Organika/sdxl-detector")
        self.detector2 = self._load_model("umm-maybe/AI-image-detector")
        self.detector3 = self._load_model("Falcons/ai-vs-real-image-classifier")

        self._load_clip()

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        print(">>> Sistema V5 listo y optimizado.")

    # ─────────────────────────────────────────────
    # CARGA DE MODELOS (OPTIMIZADA)
    # ─────────────────────────────────────────────

    def _load_model(self, model_path):
        try:
            return pipeline(
                "image-classification",
                model=model_path,
                device=self.device,
                torch_dtype=self.torch_dtype,  # ~50% menos VRAM en GPU
            )
        except Exception as e:
            logger.error(f"Error cargando {model_path}: {e}")
            return None

    def _load_clip(self):
        try:
            precision = "fp16" if self.device == 0 else "fp32"
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k", precision=precision
            )
            if self.device == 0:
                self.clip_model = self.clip_model.to("cuda")
            self.clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
            print(">>> Módulo CLIP cargado en alta eficiencia.")
        except Exception as e:
            logger.error(f"Error cargando CLIP: {e}")
            self.clip_model = None

    # ─────────────────────────────────────────────
    # METADATA / EXIF
    # ─────────────────────────────────────────────

    def _check_exif(self, img_pil):
        has_real = False
        has_ai = False
        try:
            exif_data = (
                img_pil.getexif()
                if hasattr(img_pil, "getexif")
                else getattr(img_pil, "_getexif", lambda: None)()
            )
            if not exif_data:
                return False, False

            optical_tags = {
                "ExposureTime", "FNumber", "ISOSpeedRatings",
                "FocalLength", "LensModel", "ShutterSpeedValue",
                "ApertureValue", "MeteringMode", "Flash",
            }
            # ── Grok/Aurora añade "xai" o "aurora" en Software/Comment ──
            ai_keywords = [
                "midjourney", "stable diffusion", "dall-e",
                "ai generated", "comfyui", "firefly", "imagen",
                "artificial intelligence", "generative",
                "xai", "aurora", "grok", "flux", "ideogram",
            ]

            optical_count = 0
            for tag, value in exif_data.items():
                decoded = TAGS.get(tag, tag)
                val = str(value).lower()
                if decoded in optical_tags:
                    optical_count += 1
                if any(x in val for x in ai_keywords):
                    has_ai = True

            if optical_count >= 3:
                has_real = True

        except Exception:
            pass

        return has_real, has_ai

    # ─────────────────────────────────────────────
    # ANÁLISIS FORENSE GENERAL
    # ─────────────────────────────────────────────

    # ─────────────────────────────────────────────
    # FILTRO SRM (Spatial Rich Model)
    # ─────────────────────────────────────────────

    def _srm_filter(self, gray):
        """
        Filtro Spatial Rich Model (SRM).
        Extrae el mapa de ruido de alta frecuencia residual.
        Las IAs generativas tienden a producir un ruido estadísticamente
        demasiado uniforme / bajo en varianza comparado con fotos reales.
        """
        kernel = np.array([
            [-1,  2, -1],
            [ 2, -4,  2],
            [-1,  2, -1],
        ], dtype=np.float32) / 4.0
        filtered = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        return float(np.var(filtered))

    def _gradient_flatness(self, gray):
        """
        Métrica de alto impacto: Las IAs generan gradientes demasiado suaves.
        Medimos cuántos píxeles tienen gradiente prácticamente cero en zonas
        que deberían tener variación (piel, cielo, tela, etc.)
        Real = muchas micro-variaciones. IA = áreas planas perfectas.
        """
        try:
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            # Porcentaje de píxeles con gradiente casi nulo (<2.0)
            flat_ratio = float(np.mean(magnitude < 2.0))
            # Varianza del campo de gradientes (baja = demasiado uniforme)
            grad_var = float(np.var(magnitude))
            return flat_ratio, grad_var
        except Exception:
            return 0.0, 9999.0

    def _smooth_area_noise_floor(self, gray):
        """
        Métrica crítica: En zonas uniformes (cielo, piel lisa),
        una foto real tiene ruido ISO. Una imagen IA tiene cero ruido
        porque el VAE interpola perfectamente.
        """
        try:
            # Detectar zonas muy uniformes (bajo gradiente local)
            blur = cv2.GaussianBlur(gray, (15, 15), 0)
            diff_from_blur = np.abs(gray.astype(np.float32) - blur.astype(np.float32))
            smooth_mask = diff_from_blur < 3.0  # zona muy uniform
            if smooth_mask.sum() < 100:          # no hay zonas suaves
                return 50.0                      # valor neutral
            noise_in_smooth = float(np.std(gray[smooth_mask].astype(np.float32)))
            return noise_in_smooth               # Real > 2.5, IA < 1.5
        except Exception:
            return 50.0

    def _color_channel_independence(self, img_cv):
        """
        En fotos reales cada canal tiene ruido independiente del sensor.
        En IA los canales son hiperbolicamente correlacionados.
        Retorna la media de correlación cruzada de canales (Real: <0.85, IA: >0.92)
        """
        try:
            b, g, r = [c.astype(np.float32).flatten() for c in cv2.split(img_cv)]
            corr_rg = float(np.corrcoef(r, g)[0, 1])
            corr_rb = float(np.corrcoef(r, b)[0, 1])
            corr_gb = float(np.corrcoef(g, b)[0, 1])
            return (corr_rg + corr_rb + corr_gb) / 3.0
        except Exception:
            return 0.0

    def _forensics(self, img_pil, img_cv, gray):
        noise = float(np.var(cv2.Laplacian(gray, cv2.CV_64F)))
        edges = cv2.Canny(gray, 100, 200)
        edge_density = float(np.sum(edges) / (edges.size + 1e-7))

        f = np.fft.fft2(gray)
        mag = np.abs(np.fft.fftshift(f))
        h, w = gray.shape
        center = mag[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        outer = mag.copy()
        outer[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 0
        fft_ratio = float(np.sum(outer) / (np.sum(center) + 1e-7))

        blockiness  = self._jpeg_blockiness(gray)
        ela_var     = self._compute_ela(img_pil, img_cv)
        color_var   = self._color_variance(img_cv)
        texture_score = self._texture_uniformity(gray)
        rolloff     = self._high_freq_rolloff(gray)
        entropy_var = self._local_entropy_variance(gray)
        srm_noise   = self._srm_filter(gray)
        grad_flat, grad_var  = self._gradient_flatness(gray)
        smooth_noise = self._smooth_area_noise_floor(gray)
        chan_corr    = self._color_channel_independence(img_cv)

        return {
            "noise": noise,
            "edges": edge_density,
            "fft_ratio": fft_ratio,
            "blockiness": blockiness,
            "ela_var": ela_var,
            "color_var": color_var,
            "texture_score": texture_score,
            "rolloff": rolloff,
            "entropy_var": entropy_var,
            "srm_noise": srm_noise,
            "grad_flat": grad_flat,
            "grad_var": grad_var,
            "smooth_noise": smooth_noise,
            "chan_corr": chan_corr,
        }

    def _high_freq_rolloff(self, gray):
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        mag = np.abs(fshift) + 1e-7
        h, w = gray.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        max_r = min(h, w) / 2
        mask_mid = (r > max_r * 0.3) & (r <= max_r * 0.6)
        mask_high = (r > max_r * 0.8) & (r <= max_r * 0.95)
        energy_mid = float(np.mean(mag[mask_mid])) if np.any(mask_mid) else 1.0
        energy_high = float(np.mean(mag[mask_high])) if np.any(mask_high) else 1.0
        return energy_mid / (energy_high + 1e-5)

    def _local_entropy_variance(self, gray):
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        local_var = cv2.Laplacian(blur, cv2.CV_64F) ** 2
        return float(np.var(local_var))

    def _jpeg_blockiness(self, gray):
        h, w = gray.shape
        block = 8
        diff = 0
        gray_int = gray.astype(np.int32)
        for i in range(block, h, block):
            diff += np.sum(np.abs(gray_int[i, :] - gray_int[i - 1, :]))
        for j in range(block, w, block):
            diff += np.sum(np.abs(gray_int[:, j] - gray_int[:, j - 1]))
        return float(diff / (h * w + 1e-7))

    def _compute_ela(self, img_pil, img_cv):
        try:
            buffer = io.BytesIO()
            img_pil.save(buffer, format="JPEG", quality=90)
            buffer.seek(0)
            img_c = Image.open(buffer).convert("RGB")
            img_c_cv = cv2.cvtColor(np.array(img_c), cv2.COLOR_RGB2BGR)
            diff = cv2.absdiff(img_cv, img_c_cv)
            max_d = np.max(diff)
            if max_d == 0:
                return 0.0
            return float(np.var(diff * (255.0 / max_d)))
        except Exception:
            return 0.0

    def _color_variance(self, img_cv):
        try:
            hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
            return float(np.var(hsv[:, :, 1]))
        except Exception:
            return 0.0

    def _texture_uniformity(self, gray):
        kernel = np.ones((8, 8), np.float32) / 64
        mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        sq_mean = cv2.filter2D((gray.astype(np.float32)) ** 2, -1, kernel)
        local_var = sq_mean - mean ** 2
        return float(np.mean(np.sqrt(np.maximum(local_var, 0))))

    def _residual_noise(self, gray):
        blur = cv2.bilateralFilter(gray, 9, 75, 75).astype(np.float32)
        residual = np.abs(gray.astype(np.float32) - blur)
        return float(np.var(residual))

    # ─────────────────────────────────────────────
    # FIRMA ESPECÍFICA GROK AURORA  ██████
    # ─────────────────────────────────────────────

    def _grok_aurora_signature(self, img_pil, img_cv, gray):
        """
        Detecta la firma del VAE de xAI (Grok Aurora).
        Aurora usa un espacio latente altamente comprimido que deja rastros 
        matemáticos de periodicidad de 2x2 píxeles.
        """
        score = 0.0
        reasons = []

        try:
            h, w = gray.shape
            min_dim = min(h, w)
            
            if min_dim < 64:
                reasons.append("Imagen muy pequeña para análisis forense completo")
                return 0.0, reasons

            cb_score = self._vae_checkerboard(gray)
            if cb_score > 0.08:            # antes: 0.15
                score += 30
                reasons.append(f"Artefacto de rejilla VAE detectado (score={cb_score:.2f})")
            elif cb_score > 0.05:
                score += 12
                reasons.append(f"Rejilla VAE leve detectada (score={cb_score:.2f})")

            rgb_corr = self._rgb_interchannel_correlation(img_cv)
            if rgb_corr > 0.95:            # antes: 0.98
                score += 25
                reasons.append(f"Correlación RGB sintética ({rgb_corr:.3f})")
            elif rgb_corr > 0.90:
                score += 10
                reasons.append(f"Correlación RGB elevada ({rgb_corr:.3f})")

            chroma_ratio = self._chroma_blur_ratio(img_cv)
            if chroma_ratio > 2.5:         # antes: 3.8
                score += 20
                reasons.append(f"Chroma Bleeding detectado (ratio={chroma_ratio:.1f})")
            elif chroma_ratio > 1.8:
                score += 8
                reasons.append(f"Chroma Blur leve (ratio={chroma_ratio:.1f})")

            dwt_score = self._dwt_artifact_detection(gray)
            if dwt_score > 0.08:           # antes: 0.12
                score += 15
                reasons.append(f"Firma DWT detectada (score={dwt_score:.2f})")
            elif dwt_score > 0.05:
                score += 6
                reasons.append(f"Firma DWT leve (score={dwt_score:.2f})")

        except Exception as e:
            print(f"Error en firma Grok: {e}")

        return min(score, 100.0), reasons

    @functools.lru_cache(maxsize=32)
    def _dwt_artifact_detection(self, gray):
        """Detecta artefactos de compresión DWT typical de VAEs."""
        try:
            import pywt
            coeffs = pywt.wavedec2(gray, 'haar', level=2)
            cH = np.abs(coeffs[1][0])
            cV = np.abs(coeffs[1][1])
            cD = np.abs(coeffs[1][2])
            
            energy = np.mean(cH) + np.mean(cV) + np.mean(cD)
            detail_energy = np.std(cH) + np.std(cV) + np.std(cD)
            
            return float(detail_energy / (energy + 1e-7))
        except ImportError:
            return 0.0
        except Exception:
            return 0.0

    # ─────────────────────────────────────────────
    # FUNCIONES MATEMÁTICAS AVANZADAS
    # ─────────────────────────────────────────────

    def _vae_checkerboard(self, gray):
        """Calcula la energía en las esquinas del espectro (periodicidad 2px)."""
        f = np.fft.fft2(gray.astype(np.float32))
        fshift = np.fft.fftshift(f)
        mag = np.abs(fshift)
        h, w = mag.shape
        # Miramos la energía en los bordes del espectro (alta frecuencia pura)
        corners = (mag[0:5, 0:5].mean() + mag[-5:, -5:].mean()) / 2
        center = mag[h//2-5:h//2+5, w//2-5:w//2+5].mean()
        return float(corners / (center + 1e-7))

    def _rgb_interchannel_correlation(self, img_cv):
        """Mide qué tan 'pegados' están los colores."""
        b, g, r = cv2.split(img_cv.astype(np.float32))
        corr_rg = np.corrcoef(r.flatten(), g.flatten())[0,1]
        corr_rb = np.corrcoef(r.flatten(), b.flatten())[0,1]
        return (corr_rg + corr_rb) / 2

    def _chroma_blur_ratio(self, img_cv):
        """Diferencia de nitidez entre blanco y negro vs color."""
        ycrcb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y_lap = cv2.Laplacian(y, cv2.CV_64F).var()
        c_lap = (cv2.Laplacian(cr, cv2.CV_64F).var() + cv2.Laplacian(cb, cv2.CV_64F).var()) / 2
        return float(y_lap / (c_lap + 1e-7))

    # ─────────────────────────────────────────────
    # DETECCIÓN DE ROSTROS
    # ─────────────────────────────────────────────

    def _detect_faces(self, gray):
        min_size = self.config.get("min_face_size", 50)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(min_size, min_size))
        return faces

    def _face_symmetry(self, gray, faces):
        sym_scores = []
        min_size = self.config.get("min_face_size", 50)
        for x, y, w, h in faces:
            if w < min_size or h < min_size:
                continue
            face = gray[y : y + h, x : x + w]
            flip = cv2.flip(face, 1)
            diff = np.mean(np.abs(face.astype(np.float32) - flip.astype(np.float32)))
            sym_scores.append(diff)
        
        if not sym_scores:
            return None
        return float(np.mean(sym_scores))

    def _face_liveness(self, img_cv, faces):
        """Detecta señales de liveness en rostros (reflejos, textura)."""
        if len(faces) == 0:
            return 0.0
        
        liveness_scores = []
        for x, y, w, h in faces:
            if w < 50 or h < 50:
                continue
            
            face_bgr = img_cv[y:y+h, x:x+w]
            hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
            
            v_channel = hsv[:, :, 2]
            skin_tone_var = np.var(v_channel)
            
            eye_regions = [
                face_bgr[int(h*0.2):int(h*0.35), int(w*0.25):int(w*0.4)],
                face_bgr[int(h*0.2):int(h*0.35), int(w*0.6):int(w*0.75)],
            ]
            
            has_eyes = all(r.shape[0] > 5 and r.shape[1] > 5 for r in eye_regions)
            if has_eyes:
                pupil_dark = sum(np.mean(r[:, :, 0]) < 80 for r in eye_regions if r.size > 0)
                liveness_scores.append(pupil_dark / 2)
        
        return float(np.mean(liveness_scores)) if liveness_scores else 0.0

    # ─────────────────────────────────────────────
    # SCORING DE MODELOS NEURONALES
    # ─────────────────────────────────────────────

    def _score_model(self, model, img):
        if not model:
            return None
        try:
            res = model(img)
            if isinstance(res, dict):
                res = [res]
            # dict comprehension más eficiente
            scores = {str(r["label"]).lower(): float(r["score"]) for r in res}

            ai_patterns   = ["fake", "ai", "artificial", "synthetic", "generated", "non-photo"]
            real_patterns = ["real", "human", "natural", "photo", "authentic", "genuine"]

            for key, sc in scores.items():
                if any(x in key for x in ai_patterns):
                    return sc * 100
                if any(x in key for x in real_patterns):
                    return (1.0 - sc) * 100

            if len(scores) == 1:
                return list(scores.values())[0] * 100
            return None
        except Exception as e:
            logger.debug(f"Modelo falló en predicción: {e}")
            return None

    def _score_model_safe(self, model_name, img):
        cache_key = f"{model_name}_{id(img)}"
        if cache_key in self._models_cache:
            return self._models_cache[cache_key]
        
        model = getattr(self, model_name, None)
        result = self._score_model(model, img)
        
        if len(self._models_cache) > 100:
            self._models_cache.clear()
        self._models_cache[cache_key] = result
        return result

    def _clip_score(self, img_pil):
        """
        Prompts específicos de Aurora/Grok para discriminación semántica.
        Usa torch.inference_mode() (más rápido que no_grad en PyTorch moderno)
        y mueve tensores al dispositivo correcto.
        """
        if self.clip_model is None:
            return None
        try:
            dev = "cuda" if self.device == 0 else "cpu"
            image = self.clip_preprocess(img_pil).unsqueeze(0).to(dev)

            prompts = [
                # ── Reales ──
                "an authentic photograph taken with a DSLR camera with natural sensor noise, "
                "genuine optical lens aberrations, and physically accurate lighting",
                "a candid real-world photo with authentic film grain, imperfect focus, "
                "random micro-textures and genuine color fringing from a real lens",
                # ── IA genérica ──
                "an AI generated image with synthetic over-smoothed textures and unnatural "
                "perfect lighting created by a latent diffusion model",
                # ── Grok Aurora específico ──
                "a hyper-realistic image generated by Grok Aurora from xAI with perfect skin "
                "texture, artificially smooth gradients, and VAE checkerboard artifacts",
                "a synthetic portrait generated by the Aurora diffusion model with impossibly "
                "smooth skin, perfect symmetry, and latent space color bleeding",
                # ── Flux/Aurora VAE ──
                "an image produced by a Flux or Aurora VAE decoder with chroma blur, "
                "bimodal sharpness distribution and over-saturated local colors",
            ]
            text = self.clip_tokenizer(prompts).to(dev)

            with torch.inference_mode():  # más rápido que no_grad en PyTorch >= 1.9
                img_f = self.clip_model.encode_image(image)
                txt_f = self.clip_model.encode_text(text)
                img_f /= img_f.norm(dim=-1, keepdim=True)
                txt_f /= txt_f.norm(dim=-1, keepdim=True)
                sim = (img_f @ txt_f.T).softmax(dim=-1)[0]

            real_score = float(sim[0] + sim[1])
            ai_score   = float(sim[2] + sim[3] + sim[4] + sim[5])

            total = real_score + ai_score
            if total == 0:
                return None
            return (ai_score / total) * 100
        except Exception as e:
            logger.debug(f"CLIP falló en predicción: {e}")
            return None

    # ─────────────────────────────────────────────
    # PUNTUACIÓN FORENSE CALIBRADA
    # ─────────────────────────────────────────────

    def _forensic_score(self, f_data, residual_noise, faces, gray, img_cv, has_exif_optical):
        """
        Sistema de puntuación forense recalibrado.
        Cada métrica tiene el doble de peso que antes.
        Se añade penalización por ausencia de EXIF óptico (la señal más barata y confiable).
        """
        score = 0.0
        reasons = []
        cfg = self.config

        # ─── NOTA EXIF ───────────────────────────────────────────────────────
        # NO añadimos puntos por ausencia de EXIF porque fotos reales enviadas
        # por WhatsApp, Instagram, Twitter, etc. siempre pierden su EXIF.
        # Solo usamos el EXIF como evidencia positiva (ya manejado en process_image).

        # ─── GRADIENTE PLANO ─────────────────────────────────────────────────
        # Fotos reales: fondos uniformes pueden tener 50-70% de gradiente~0.
        # IA tipo Aurora: 80-95% de gradiente~0 (todo es demasiado suave).
        grad_flat  = f_data.get("grad_flat", 0.0)
        grad_var   = f_data.get("grad_var", 9999.0)
        if grad_flat > 0.82:              # umbral alto: solo dispara en IA extrema
            score += 28
            reasons.append(f"Gradientes extremadamente planos ({grad_flat*100:.0f}%) → IA")
        elif grad_flat > 0.72:
            score += 12
            reasons.append(f"Gradientes muy uniformes ({grad_flat*100:.0f}%)")
        if grad_var < 200:                # varianza muy baja = demasiado uniforme
            score += 18
            reasons.append(f"Campo de gradiente casi constante (var={grad_var:.0f})")
        elif grad_var < 600:
            score += 7
            reasons.append(f"Campo de gradiente poco variable (var={grad_var:.0f})")

        # ─── RUIDO EN ZONAS LISAS ────────────────────────────────────────────
        # NOTA CRÍTICA: JPEG destruye el ruido ISO en zonas lisas. Por eso
        # incluso fotos reales pueden tener σ < 2.0 después de compresión JPEG.
        # Solo flagear si el ruido es PRÁCTICAMENTE CERO (σ < 0.5),
        # que indica generación perfecta por VAE sin ningún ruido digital.
        smooth_noise = f_data.get("smooth_noise", 50.0)
        if smooth_noise < 0.4:            # esencialmente cero = VAE generativo
            score += 30
            reasons.append(f"Zonas lisas perfectas sin ruido (σ={smooth_noise:.2f}) → VAE")
        elif smooth_noise < 0.8:
            score += 14
            reasons.append(f"Ruido muy bajo en zonas lisas (σ={smooth_noise:.2f})")

        # ─── CORRELACIÓN DE CANALES ──────────────────────────────────────────
        # Fotos reales pueden tener correlación 0.70-0.93 (depende del sujeto).
        # JPEG además mezcla canales. Solo flagear correlación EXTREMA (>0.97).
        chan_corr = f_data.get("chan_corr", 0.0)
        if chan_corr > 0.97:
            score += 22
            reasons.append(f"Canales RGB perfectamente correlacionados ({chan_corr:.3f}) → IA")
        elif chan_corr > 0.94:
            score += 9
            reasons.append(f"Alta correlacion de canales ({chan_corr:.3f})")

        # ─── TEXTURA ──────────────────────────────────────────────────
        ts = f_data["texture_score"]
        if ts < cfg["texture_thresholds"][0]:
            score += 30
            reasons.append(f"Textura sintética uniforme (score={ts:.1f})")
        elif ts < cfg["texture_thresholds"][1]:
            score += 14
            reasons.append(f"Textura ligeramente sintética (score={ts:.1f})")

        # ─── RUIDO RESIDUAL ─────────────────────────────────────────────
        rn = residual_noise
        if rn < cfg["noise_thresholds"][0]:
            score += 25
            reasons.append(f"Ruido residual casi nulo (rn={rn:.1f}) → motor difusión")
        elif rn > cfg["noise_thresholds"][2]:
            score += 18
            reasons.append(f"Ruido residual excesivo/plano (rn={rn:.1f})")
        elif rn < cfg["noise_thresholds"][1]:
            score += 12
            reasons.append(f"Ruido residual bajo (rn={rn:.1f})")

        # ─── FFT ────────────────────────────────────────────────────
        fft = f_data["fft_ratio"]
        if fft < cfg["fft_thresholds"][0]:
            score += 25
            reasons.append(f"FFT aplanado (ratio={fft:.2f})")
        elif fft < cfg["fft_thresholds"][1]:
            score += 10
            reasons.append(f"FFT levemente aplanado (ratio={fft:.2f})")

        # ─── COLOR ────────────────────────────────────────────────────
        cv = f_data["color_var"]
        if cv < cfg["color_var_threshold"]:
            score += 12
            reasons.append(f"Saturación cromática anómala (var={cv:.0f})")

        # ─── ELA ─────────────────────────────────────────────────────
        ela = f_data["ela_var"]
        if ela < cfg["ela_threshold"]:
            score += 10
            reasons.append("ELA plano → imagen no comprimida orgánicamente")

        # ─── SIMETRÍA FACIAL ───────────────────────────────────────────
        if len(faces) > 0:
            sym = self._face_symmetry(gray, faces)
            if sym is not None and sym < cfg["symmetry_threshold"]:
                score += 20
                reasons.append(f"Simetría facial casi perfecta (diff={sym:.2f})")
            liveness = self._face_liveness(img_cv, faces)
            if liveness > 0.5:
                score -= 8
                reasons.append(f"Posible rostro real (liveness={liveness:.2f})")

        # ─── ROLLOFF ESPECTRAL ───────────────────────────────────────
        rolloff = f_data.get("rolloff", 1.0)
        if rolloff > cfg["rolloff_thresholds"][1]:
            score += 18
            reasons.append(f"Decaimiento espectral anómalo (Firma VAE, ratio={rolloff:.1f})")
        elif rolloff > cfg["rolloff_thresholds"][0]:
            score += 9
            reasons.append(f"Frecuencias ultra-altas ausentes (ratio={rolloff:.1f})")

        # ─── ENTROPÍA LOCAL ───────────────────────────────────────────
        ent = f_data.get("entropy_var", 5000)
        if ent < cfg["entropy_threshold"]:
            score += 15
            reasons.append(f"Micro-textura con over-smoothing (var={ent:.0f})")

        # ─── SRM ───────────────────────────────────────────────────────
        srm = f_data.get("srm_noise", 9999.0)
        srm_thresh = cfg.get("srm_threshold", 3.0)
        if srm < srm_thresh:
            score += 18
            reasons.append(f"SRM residual bajo → motor generativo (var={srm:.2f})")
        elif srm < srm_thresh * 2:
            score += 8
            reasons.append(f"SRM residual levemente bajo (var={srm:.2f})")

        # ─── BLOCKINESS ────────────────────────────────────────────────
        blockiness = f_data.get("blockiness", 0)
        if blockiness > 5.0:
            score += 6
            reasons.append(f"Blockiness JPEG elevado ({blockiness:.2f})")

        return min(score, 100.0), reasons

    # ---------------- MAIN ----------------

    def process_image(self, data, verbose=False):
        try:
            img_pil = Image.open(io.BytesIO(data)).convert("RGB")
            img_cv  = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            gray    = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

            h, w = gray.shape
            if h < 32 or w < 32:
                return {"status": "error", "error": "Imagen muy pequeña"}

            # -- 1. EXIF -------------------------------------------------
            has_real, has_ai = self._check_exif(img_pil)
            if has_ai:
                return {"status": "success", "probabilidad": 99,
                        "nota": "Firma IA en EXIF", "fuente": "exif"}
            if has_real:
                return {"status": "success", "probabilidad": 5,
                        "nota": "EXIF optico detectado (Probable Real)", "fuente": "exif"}

            # -- 2. MODELOS NEURALES ------------------------------------
            s1   = self._score_model(self.detector1, img_pil)
            s2   = self._score_model(self.detector2, img_pil)
            s3   = self._score_model(self.detector3, img_pil)
            clip = self._clip_score(img_pil)

            s1_raw, s2_raw, s3_raw = s1, s2, s3

            s1   = s1   if s1   is not None else 50.0
            s2   = s2   if s2   is not None else 50.0
            s3   = s3   if s3   is not None else 50.0
            clip = clip if clip is not None else 50.0

            weights    = self.config["model_weights"]
            neural_avg = (
                s1 * weights["detector1"]
                + s2 * weights["detector2"]
                + s3 * 0.10
                + clip * weights["clip"]
            )

            # -- 3. FORENSE + GROK -------------------------------------
            residual_noise = self._residual_noise(gray)
            faces  = self._detect_faces(gray)
            f_data = self._forensics(img_pil, img_cv, gray)

            forensic_score, forensic_reasons = self._forensic_score(
                f_data, residual_noise, faces, gray, img_cv,
                has_exif_optical=False,   # ya salimos arriba si tenia EXIF real
            )
            grok_score, grok_reasons = self._grok_aurora_signature(
                img_pil, img_cv, gray
            )

            # -- 4. ENSEMBLE ADAPTATIVO ---------------------------------
            # Regla 1: neural tiene peso bajo (25%), no conoce Aurora/Grok.
            # Regla 2: si ningun modelo > 60% -> peso neural se reduce a 10%.
            # Regla 3: pisos minimos garantizados por forensic/grok.
            unr = self.config.get("neural_unreliable_threshold", 60.0)
            neural_raw_scores = [v for v in [s1_raw, s2_raw, s3_raw] if v is not None]
            neural_uncertain  = (not neural_raw_scores
                                 or max(neural_raw_scores) < unr)

            if neural_uncertain:
                w_n, w_f, w_g = 0.10, 0.58, 0.32
                mode = "forense-dominante"
            else:
                ew = self.config.get("ensemble_weights",
                                     {"neural": 0.25, "forensic": 0.45, "grok": 0.30})
                w_n, w_f, w_g = ew["neural"], ew["forensic"], ew["grok"]
                mode = "equilibrado"

            prob = neural_avg * w_n + forensic_score * w_f + grok_score * w_g

            # Pisos minimos garantizados
            if forensic_score >= 70:
                prob = max(prob, 72.0)
                forensic_reasons.append(
                    f"Nivel forense critico -> piso=72 (score={forensic_score:.0f})"
                )
            elif forensic_score >= 55:
                prob = max(prob, 60.0)
                forensic_reasons.append(
                    f"Nivel forense alto -> piso=60 (score={forensic_score:.0f})"
                )
            elif forensic_score >= 35:
                prob = max(prob, 45.0)
                forensic_reasons.append(
                    f"Evidencia forense moderada -> piso=45 (score={forensic_score:.0f})"
                )

            if grok_score > self.config["grok_threshold"]:
                prob = max(prob, grok_score * 0.9)
                grok_reasons.append(
                    f"Firma Aurora prevalece (score={grok_score:.1f})"
                )

            # Refuerzo cruzado
            if not neural_uncertain and neural_avg > 60 and forensic_score > 40:
                prob = min(prob * 1.15, 99)
                mode += "+refuerzo"

            final = int(np.clip(prob, 0, 100))

            # -- 5. RESPUESTA -------------------------------------------
            all_reasons = []
            all_reasons.extend(forensic_reasons)
            all_reasons.extend(grok_reasons)

            result = {
                "status": "success",
                "probabilidad": final,
                "nota": " | ".join(all_reasons) if all_reasons else "Organico",
                "detalles": {
                    "neural": {
                        "detector1": round(s1, 1),
                        "detector2": round(s2, 1),
                        "detector3": round(s3, 1),
                        "clip": round(clip, 1),
                        "promedio": round(neural_avg, 1),
                        "incierto": neural_uncertain,
                    },
                    "forense": round(forensic_score, 1),
                    "grok_signature": round(grok_score, 1),
                    "modo": mode,
                    "prob_bruta": round(prob, 1),
                },
            }

            if verbose:
                result["detalles"]["forense_raw"] = {
                    k: round(v, 4) if isinstance(v, float) else v
                    for k, v in f_data.items()
                }

            return result

        except Exception as e:
            logger.error(f"Error en process_image: {e}")
            return {"status": "error", "error": str(e)}

    # ---------------------------------------------------------

    # BATCH PROCESSING
    # ─────────────────────────────────────────────

    def process_batch(
        self,
        data_list,
        max_workers: int = 4,
        progress_callback=None,
    ):
        """
        Analiza múltiples imágenes en paralelo.

        Args:
            data_list  : list[bytes | dict]  – bytes crudos O dicts con clave
                         ``data`` (bytes) y opcionalmente ``id`` (str).
            max_workers: int – hilos paralelos (default 4).
            progress_callback: callable(done: int, total: int, result: dict) | None

        Returns:
            dict con:
                ``results``  – lista de resultados en el mismo orden que data_list
                ``summary``  – estadísticas agregadas del batch
                ``elapsed_s``– segundos totales del batch
        """
        total = len(data_list)
        if total == 0:
            return {
                "results": [],
                "summary": self._batch_summary([]),
                "elapsed_s": 0.0,
            }

        # Normalizar entradas → siempre trabajamos con (idx, id, bytes)
        tasks = []
        for idx, item in enumerate(data_list):
            if isinstance(item, (bytes, bytearray)):
                tasks.append((idx, str(idx), bytes(item)))
            elif isinstance(item, dict):
                img_id = str(item.get("id", idx))
                img_bytes = item.get("data", b"")
                tasks.append((idx, img_id, img_bytes))
            else:
                tasks.append((idx, str(idx), b""))

        results_ordered = [None] * total
        done_count = 0
        t_start = time.perf_counter()

        def _worker(task):
            idx, img_id, img_bytes = task
            if not img_bytes:
                return idx, {
                    "id": img_id,
                    "status": "error",
                    "error": "Sin datos de imagen",
                }
            try:
                res = self.process_image(img_bytes)
                res["id"] = img_id
                return idx, res
            except Exception as exc:
                return idx, {"id": img_id, "status": "error", "error": str(exc)}

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_worker, task): task[0] for task in tasks}
            for future in concurrent.futures.as_completed(futures):
                idx, res = future.result()
                results_ordered[idx] = res
                done_count += 1
                if progress_callback is not None:
                    try:
                        progress_callback(done_count, total, res)
                    except Exception:
                        pass

        elapsed = time.perf_counter() - t_start
        return {
            "results": results_ordered,
            "summary": self._batch_summary(results_ordered),
            "elapsed_s": round(elapsed, 3),
        }

    @staticmethod
    def _batch_summary(results):
        """Genera estadísticas agregadas de un batch de resultados."""
        total = len(results)
        if total == 0:
            return {"total": 0}

        ok = [r for r in results if r and r.get("status") == "success"]
        errors = total - len(ok)
        probs = [r["probabilidad"] for r in ok if "probabilidad" in r]

        if not probs:
            return {
                "total": total,
                "ok": 0,
                "errors": errors,
            }

        ai_count    = sum(1 for p in probs if p >= 70)
        real_count  = sum(1 for p in probs if p <  40)
        doubt_count = total - ai_count - real_count - errors

        return {
            "total"        : total,
            "ok"           : len(ok),
            "errors"       : errors,
            "ai_detected"  : ai_count,
            "real_detected": real_count,
            "doubtful"     : doubt_count,
            "prob_mean"    : round(sum(probs) / len(probs), 1),
            "prob_max"     : max(probs),
            "prob_min"     : min(probs),
        }


# ─────────────────────────────────────────────
# SINGLETON
# ─────────────────────────────────────────────

_engine = None


def get_engine(config=None):
    global _engine
    if _engine is None:
        _engine = NueroscanEngineV5(config)
    return _engine


def analyze_image(data, config=None):
    engine = get_engine(config)
    return engine.process_image(data)


def reset_engine():
    global _engine
    if _engine:
        _engine._models_cache.clear()
        _engine = None


def analyze_batch(
    data_list,
    config=None,
    max_workers: int = 4,
    progress_callback=None,
):
    """
    Analiza un lote de imágenes usando el engine singleton.

    Args:
        data_list       : list[bytes | dict]  – ver ``process_batch`` para formato.
        config          : dict | None  – configuración del engine (sólo se aplica
                          si el engine aún no ha sido inicializado).
        max_workers     : int – hilos paralelos.
        progress_callback: callable(done, total, result) | None

    Returns:
        dict  – ``results``, ``summary`` y ``elapsed_s``.

    Ejemplo de uso básico::

        with open("img1.jpg", "rb") as f1, open("img2.jpg", "rb") as f2:
            batch = [
                {"id": "foto_1", "data": f1.read()},
                {"id": "foto_2", "data": f2.read()},
            ]
        resultado = analyze_batch(batch, max_workers=2)
        print(resultado["summary"])

    Ejemplo con callback de progreso::

        def on_progress(done, total, res):
            pct = int(done / total * 100)
            print(f"[{pct}%] {done}/{total} – id={res.get('id')} prob={res.get('probabilidad')}")

        resultado = analyze_batch(batch, progress_callback=on_progress)
    """
    engine = get_engine(config)
    return engine.process_batch(
        data_list,
        max_workers=max_workers,
        progress_callback=progress_callback,
    )