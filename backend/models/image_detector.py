import base64
import io
import json
import hashlib
import logging
import warnings
import concurrent.futures
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import cv2
from PIL import Image
from PIL.ExifTags import TAGS
from transformers import pipeline

try:
    import open_clip
except ImportError:
    open_clip = None

# [V10.3] Importación de Hive
from models.modules.hive_analyzer import HiveAnalyzer

warnings.filterwarnings("ignore")
import os
os.environ["HF_HUB_READ_TIMEOUT"] = "60"  # [PROD] V10.3: Aumentar timeout para evitar crash en arranque
os.environ["TRANSFORMERS_OFFLINE"] = "0"   # Permitir descarga si es necesario, pero priorizar cache
logger = logging.getLogger("TalosV10")

# Silenciar logs ruidosos de bibliotecas externas
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

import transformers
transformers.utils.logging.set_verbosity_warning()

_MAX_IMAGE_BYTES  = 50 * 1024 * 1024   # 50 MB raw bytes
_MAX_IMAGE_PIXELS = 4096 * 4096        # ~16 Mpx

# Checkpoint rama híbrida (CLIP + EfficientNet + FFT + residual), opcional
_HYBRID_CKPT = Path(__file__).resolve().parent / "checkpoints" / "ai_image_detector_quick.pt"

# ═══════════════════════════════════════════════════════════════
# RESOLUCIONES NATIVAS DE GENERADORES IA
# Clave para identificar imágenes sin metadata: los generadores
# producen dimensiones fijas o múltiplos exactos de sus latent spaces.
# ═══════════════════════════════════════════════════════════════
_AI_NATIVE_RESOLUTIONS: List[Tuple[int, int]] = [
    # SD 1.x / 2.x (latent 64px)
    (512, 512), (512, 768), (768, 512), (768, 768),
    (512, 896), (896, 512), (640, 640),
    # SDXL / Flux (latent 128px)
    (1024, 1024), (1024, 768), (768, 1024),
    (1344, 768), (768, 1344), (1152, 896), (896, 1152),
    (1216, 832), (832, 1216), (1344, 704), (704, 1344),
    (1536, 640), (640, 1536),
    # DALL-E 3
    (1024, 1024), (1792, 1024), (1024, 1792),
    # Midjourney (múltiplos de 64)
    (1456, 816), (816, 1456), (1232, 928),
    # Grok/Aurora (V1 + V2 + Aurora 2)
    (1024, 1024), (1366, 768), (1280, 720),
    (2048, 2048), (1536, 1024), (1024, 1536),
    (1536, 1536), (2048, 1024), (1024, 2048),
    (1280, 1280), (1920, 1080), (1080, 1920),
    # Gemini Imagen 3
    (1536, 1536), (2048, 2048), (1024, 1024),
    # Ideogram V2/V3 / Playground V3
    (1024, 1024), (1080, 1080), (1344, 768), (768, 1344),
]
_AI_NATIVE_RESOLUTIONS_SET = set(_AI_NATIVE_RESOLUTIONS)


# ═══════════════════════════════════════════════════════════════
# CONFIGURACIÓN CENTRALIZADA
# ═══════════════════════════════════════════════════════════════
@dataclass
class ThresholdConfig:
    # ── Veredicto ────────────────────────────────────────────────
    prior_bias: float              = 1.0
    verdict_ai_threshold: float    = 60.0   # V10.3 Standard Pulzo
    verdict_real_threshold: float  = 20.0   

    # ── Laplaciano ───────────────────────────────────────────────
    laplacian_ai_max: float           = 80.0
    laplacian_professional_min: float = 800.0
    laplacian_weight: float           = 30.0

    # ── Ruido ────────────────────────────────────────────────────
    noise_ai_max: float   = 1.2
    noise_weight: float   = 30.0

    # ── ELA ──────────────────────────────────────────────────────
    ela_quality: int             = 95
    ela_region_std_ai_max: float = 4.5
    ela_mean_ai_min: float       = 8.0
    ela_weight: float            = 40.0

    # ── Color (Grok / DALL-E) ────────────────────────────────────
    grok_sat_max: float  = 180.0
    grok_val_max: float  = 230.0
    grok_weight: float   = 25.0

    # ── Grok/Aurora — detector específico ────────────────────────
    # V10.1: re-calibrados — más abiertos que V9 pero protegidos por
    # el requisito mandatorio de ausencia de EXIF
    grok_noise_max: float              = 1.8
    grok_shadow_entropy_max: float     = 3.5
    grok_gradient_smoothness_max: float = 10.0
    grok_dct_uniformity_max: float     = 30.0
    grok_min_conditions: int           = 3     # V10.2: protegido por EXIF shield, seguro en 3

    # ── Shadow Entropy ───────────────────────────────────────────
    shadow_entropy_ai_max: float   = 1.1
    shadow_entropy_real_min: float = 2.5
    shadow_entropy_weight: float   = 35.0

    # ── FFT ──────────────────────────────────────────────────────
    fft_ai_min: float  = 0.32
    fft_weight: float  = 35.0

    # ── Aberración cromática (señal de cámara real) ───────────────
    ca_ai_max: float   = 0.35
    ca_weight: float   = 25.0

    # ── Entropía local de textura ─────────────────────────────────
    local_entropy_ai_max: float = 4.2
    local_entropy_weight: float = 20.0

    # ── V10: Simetría bilateral (IA produce simetría antinatural) ─
    bilateral_symmetry_ai_min: float = 0.85
    bilateral_symmetry_weight: float = 20.0

    # ── V10: Clustering de paleta de color ────────────────────────
    # Grok/Flux usan paletas estrechas con pocos clusters dominantes
    color_cluster_ai_max: int   = 6      # ≤6 clusters dominantes → IA
    color_cluster_weight: float = 20.0

    # ── V10: FFT mid-band ringing (upsampler artifacts) ──────────
    fft_midband_ring_min: float = 1.5
    fft_midband_ring_weight: float = 25.0

    # ── V10: Varianza de bordes por cuadrante ─────────────────────
    edge_variance_ratio_ai_max: float = 1.3  # IA: varianza uniforme
    edge_variance_weight: float = 15.0

    # ── Metadata ─────────────────────────────────────────────────
    metadata_confirmed_ai_weight: float  = 95.0
    metadata_c2pa_unsigned_weight: float = 20.0
    metadata_real_camera_weight: float   = -75.0

    # ── THE HIVE V3 ──────────────────────────────────────────────
    enable_hive: bool = True
    timeout_hive: float = 30.0
    hive_ai_threshold: float = 80.0  # Umbral para refuerzo agresivo

    # ── Rama híbrida (ai_image_detector + checkpoint local) ─────────
    # Fusiona con el score neural HF antes del MetaClassifier.
    enable_hybrid_image_branch: bool = True
    hybrid_blend_weight: float = 0.40  # peso del híbrido en la señal neural previa al MetaClassifier
    # Ajuste tardío asimétrico sobre prob. final (solo si el híbrido está seguro)
    hybrid_final_nudge_weight: float = 0.18
    hybrid_nudge_high_threshold: float = 0.55  # por encima: reforzar IA
    hybrid_nudge_low_threshold: float = 0.42   # por debajo: suavizar hacia REAL


# ═══════════════════════════════════════════════════════════════
# CACHÉ LRU THREAD-SAFE
# ═══════════════════════════════════════════════════════════════
class _LRUCache:
    def __init__(self, maxsize: int = 200):
        self._data: OrderedDict = OrderedDict()
        self._maxsize = maxsize
        self._lock = Lock()

    def get(self, key):
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def set(self, key, value):
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = value
            if len(self._data) > self._maxsize:
                self._data.popitem(last=False)

    def clear(self):
        with self._lock:
            self._data.clear()


# ═══════════════════════════════════════════════════════════════
# MÓDULO 1: EXTRACCIÓN DE SEÑALES UNIVERSALES
# ═══════════════════════════════════════════════════════════════
class UniversalFeatureExtractor:
    """
    Extrae 12 biometrías forenses resistentes a compresión JPEG.

    Nuevas en V9:
      10. Aberración cromática lateral (CA) — cámaras reales la tienen, IA no.
      11. Entropía local de textura — IA produce texturas demasiado uniformes.
      12. Varianza de gradiente por canal — detecta la firma de suavizado de IA.
    """

    @staticmethod
    def extract(gray: np.ndarray, img_cv: np.ndarray) -> Dict[str, Any]:
        features: Dict[str, Any] = {}
        h, w = gray.shape

        # 1. Ruido Residual
        try:
            blur = cv2.GaussianBlur(gray.astype(np.float32), (7, 7), 0)
            diff = np.abs(gray.astype(np.float32) - blur)
            mask = diff < 5.0
            features["noise_level"] = float(np.std(diff[mask])) if mask.sum() > 100 else None
        except Exception as e:
            logger.debug(f"Fallo al calcular noise_level: {e}")
            features["noise_level"] = None

        # 2. Varianza Laplaciana
        try:
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            features["laplacian_var"] = float(np.var(lap))
        except Exception as e:
            logger.debug(f"Fallo al calcular laplacian_var: {e}")
            features["laplacian_var"] = None

        # 3. FFT High-Frequency Ratio
        try:
            mag = np.abs(np.fft.fftshift(np.fft.fft2(gray.astype(float))))
            cy, cx = h // 2, w // 2
            r = np.sqrt(
                (np.ogrid[:h, :w][0] - cy) ** 2 + (np.ogrid[:h, :w][1] - cx) ** 2
            )
            mid_freq  = np.mean(mag[(r > min(h, w) * 0.10) & (r < min(h, w) * 0.30)])
            high_freq = np.mean(mag[(r > min(h, w) * 0.40) & (r < min(h, w) * 0.48)])
            features["fft_ratio"] = float(high_freq / (mid_freq + 1e-5))
        except Exception as e:
            logger.debug(f"Fallo al calcular fft_ratio: {e}")
            features["fft_ratio"] = None

        # 4. Homogeneidad de color
        try:
            hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
            features["saturation_var"]  = float(np.var(hsv[:, :, 1]))
            features["value_var"]       = float(np.var(hsv[:, :, 2]))
            features["saturation_mean"] = float(np.mean(hsv[:, :, 1]))
        except Exception as e:
            logger.debug(f"Fallo al calcular homogeneidad de color: {e}")
            features["saturation_var"]  = None
            features["value_var"]       = None
            features["saturation_mean"] = None

        # 5. Consistencia global por bloques
        try:
            step_h, step_w = max(1, h // 4), max(1, w // 4)
            blocks = [
                np.var(gray[i:min(i + step_h, h), j:min(j + step_w, w)])
                for i in range(0, h, step_h)
                for j in range(0, w, step_w)
            ]
            features["block_consistency_std"] = float(np.std(blocks)) if blocks else None
        except Exception as e:
            logger.debug(f"Fallo al calcular block_consistency_std: {e}")
            features["block_consistency_std"] = None

        # 6. Distribución de histograma
        try:
            hist, _ = np.histogram(gray, bins=256, range=(0, 255))
            features["hist_smoothness"] = float(np.std(hist))
        except Exception as e:
            logger.debug(f"Fallo al calcular hist_smoothness: {e}")
            features["hist_smoothness"] = None

        # 7. Shadow Entropy
        try:
            shadow_mask = gray < 30
            if shadow_mask.sum() > 50:
                hist_s, _ = np.histogram(gray[shadow_mask], bins=30, range=(0, 30))
                p_s = hist_s / (hist_s.sum() + 1e-8)
                p_s = p_s[p_s > 0]
                features["shadow_entropy"] = float(-np.sum(p_s * np.log2(p_s)))
            else:
                features["shadow_entropy"] = None
        except Exception as e:
            logger.debug(f"Fallo al calcular shadow_entropy: {e}")
            features["shadow_entropy"] = None

        # 8. Gradiente sintético (Grok/Aurora)
        try:
            bh_g, bw_g = max(1, h // 8), max(1, w // 8)
            grad_scores = []
            for i in range(8):
                for j in range(8):
                    blk = gray[i*bh_g:(i+1)*bh_g, j*bw_g:(j+1)*bw_g].astype(float)
                    if blk.size < 16:
                        continue
                    dx = np.diff(blk, axis=1)
                    dy = np.diff(blk, axis=0)
                    grad_scores.append(float(np.var(dx) + np.var(dy)))
            features["gradient_smoothness"] = float(np.median(grad_scores)) if grad_scores else None
        except Exception as e:
            logger.debug(f"Fallo al calcular gradient_smoothness: {e}")
            features["gradient_smoothness"] = None

        # 9. Uniformidad DCT (Grok primera generación JPEG)
        try:
            bms = []
            for i in range(0, h - 8, 8):
                for j in range(0, w - 8, 8):
                    bms.append(float(np.std(gray[i:i+8, j:j+8].astype(float))))
            features["dct_block_uniformity"] = float(np.std(bms)) if bms else None
        except Exception as e:
            logger.debug(f"Fallo al calcular dct_block_uniformity: {e}")
            features["dct_block_uniformity"] = None

        # 10. ABERRACIÓN CROMÁTICA LATERAL
        try:
            b_ch, g_ch, r_ch = cv2.split(img_cv)
            edges_r = cv2.Canny(r_ch, 40, 120).astype(np.float32)
            edges_b = cv2.Canny(b_ch, 40, 120).astype(np.float32)
            if edges_r.sum() > 100 and edges_b.sum() > 100:
                shifted = np.roll(edges_b, 1, axis=1)
                diff_rb = float(np.mean(np.abs(edges_r - shifted)))
                orig_rb = float(np.mean(np.abs(edges_r - edges_b)))
                ca_score = diff_rb / (orig_rb + 1e-5)
                features["chromatic_aberration"] = float(np.clip(ca_score, 0.0, 5.0))
            else:
                features["chromatic_aberration"] = None
        except Exception as e:
            logger.debug(f"Fallo al calcular aberración cromática: {e}")
            features["chromatic_aberration"] = None

        # 11. ENTROPÍA LOCAL DE TEXTURA
        try:
            patch_entropies = []
            ph, pw = 8, 8
            for i in range(0, h - ph, ph * 2):
                for j in range(0, w - pw, pw * 2):
                    patch = gray[i:min(i+ph, h), j:min(j+pw, w)]
                    hist_p, _ = np.histogram(patch, bins=16, range=(0, 255))
                    total = hist_p.sum()
                    if total > 0:
                        p = hist_p[hist_p > 0] / total
                        patch_entropies.append(float(-np.sum(p * np.log2(p))))
            features["local_texture_entropy"] = float(np.mean(patch_entropies)) if patch_entropies else None
        except Exception as e:
            logger.debug(f"Fallo al calcular local_texture_entropy: {e}")
            features["local_texture_entropy"] = None

        # 12. VARIANZA INTER-CANAL
        try:
            b_ch, g_ch, r_ch = cv2.split(img_cv)
            var_r = float(np.var(cv2.Laplacian(r_ch, cv2.CV_64F)))
            var_g = float(np.var(cv2.Laplacian(g_ch, cv2.CV_64F)))
            var_b = float(np.var(cv2.Laplacian(b_ch, cv2.CV_64F)))
            g_dominance = var_g / (((var_r + var_b) / 2.0) + 1e-5)
            features["green_channel_dominance"] = float(np.clip(g_dominance, 0.0, 10.0))
        except Exception as e:
            logger.debug(f"Fallo al calcular green_channel_dominance: {e}")
            features["green_channel_dominance"] = None

        # 13. SIMETRÍA BILATERAL
        try:
            left_half  = gray[:, :w // 2]
            right_half = np.fliplr(gray[:, w // 2: w // 2 * 2])
            min_w = min(left_half.shape[1], right_half.shape[1])
            if min_w > 16:
                left_flat  = left_half[:, :min_w].ravel().astype(np.float64)
                right_flat = right_half[:, :min_w].ravel().astype(np.float64)
                lm, rm = left_flat.mean(), right_flat.mean()
                ld, rd = left_flat - lm, right_flat - rm
                denom = (np.sqrt(np.sum(ld**2)) * np.sqrt(np.sum(rd**2))) + 1e-8
                corr = float(np.sum(ld * rd) / denom)
                features["bilateral_symmetry"] = float(np.clip(corr, -1.0, 1.0))
            else:
                features["bilateral_symmetry"] = None
        except Exception as e:
            logger.debug(f"Fallo al calcular bilateral_symmetry: {e}")
            features["bilateral_symmetry"] = None

        # 14. CLUSTERING DE PALETA DE COLOR
        try:
            hsv_flat = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
            n_samples = min(len(hsv_flat), 10000)
            rng = np.random.default_rng(42)
            sample = hsv_flat[rng.choice(len(hsv_flat), n_samples, replace=False)]
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            K = 12
            _, labels, centers = cv2.kmeans(sample, K, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
            counts = np.bincount(labels.ravel(), minlength=K)
            dominant = int(np.sum(counts > n_samples * 0.05))
            features["color_cluster_count"] = dominant
            hue_range = float(np.ptp(centers[:, 0]))
            features["color_hue_spread"] = hue_range
        except Exception as e:
            logger.debug(f"Fallo al calcular color_clustering: {e}")
            features["color_cluster_count"] = None
            features["color_hue_spread"] = None

        # 15. FFT MID-BAND RINGING
        try:
            mag = np.abs(np.fft.fftshift(np.fft.fft2(gray.astype(float))))
            cy, cx = h // 2, w // 2
            r = np.sqrt(
                (np.arange(h)[:, None] - cy) ** 2 +
                (np.arange(w)[None, :] - cx) ** 2
            )
            r_max = min(h, w) / 2.0
            mid_mask = (r > r_max * 0.15) & (r < r_max * 0.35)
            mid_vals = mag[mid_mask]
            if mid_vals.size > 100:
                mid_peak = float(np.percentile(mid_vals, 98))
                mid_mean = float(np.mean(mid_vals))
                features["fft_midband_ring"] = float(mid_peak / (mid_mean + 1e-5))
            else:
                features["fft_midband_ring"] = None
        except Exception as e:
            logger.debug(f"Fallo al calcular fft_midband_ring: {e}")
            features["fft_midband_ring"] = None

        # 16. VARIANZA DE BORDES POR CUADRANTE
        try:
            edges_full = cv2.Canny(gray, 30, 100).astype(np.float32)
            qh, qw = h // 2, w // 2
            quad_densities = [
                float(edges_full[:qh, :qw].mean()) if edges_full[:qh, :qw].size > 0 else 0.0,
                float(edges_full[:qh, qw:].mean()) if edges_full[:qh, qw:].size > 0 else 0.0,
                float(edges_full[qh:, :qw].mean()) if edges_full[qh:, :qw].size > 0 else 0.0,
                float(edges_full[qh:, qw:].mean()) if edges_full[qh:, qw:].size > 0 else 0.0,
            ]
            q_mean = np.mean(quad_densities) + 1e-5
            q_std = np.std(quad_densities)
            features["edge_quad_ratio"] = float(q_std / q_mean)
        except Exception as e:
            logger.debug(f"Fallo al calcular edge_quad_ratio: {e}")
            features["edge_quad_ratio"] = None

        return features


# ═══════════════════════════════════════════════════════════════
# MÓDULO 3: ANALIZADOR DE METADATA
# ═══════════════════════════════════════════════════════════════
class MetadataAnalyzer:
    """
    Extrae señales de procedencia desde metadatos embebidos.
    Cobertura: EXIF, PNG chunks, XMP, C2PA/JUMBF, escaneo binario.
    """

    _AI_SIGNATURES: Dict[str, List[str]] = {
        # ── Grandes plataformas ──────────────────────────────────────────
        "Grok/Aurora":          [
            "aurora", "xai", "grok", "x.ai", "grok-aurora", "aurora-model",
            "xai-image", "grok image", "generated by grok", "aurora diffusion",
            "xai aurora", "image created by x", "created with grok",
        ],
        "Gemini/Imagen":        [
            "google imagen", "synthid", "gemini", "google deepmind", "imagegeneration",
            "google llc", "google ai", "imagen2", "imagen3",
            "parti", "muse model", "phenaki",
        ],
        "DALL-E":               [
            "dall-e", "dall·e", "openai", "dall_e", "openai api", "dalle3", "dalle2",
        ],
        "Midjourney":           [
            "midjourney", "nijijourney", "mj v", "mjv",
            "midjourney v5", "midjourney v6", "niji 6",
        ],
        "Sora":                 ["sora", "openai sora", "openai video"],
        "Adobe Firefly":        [
            "firefly", "adobe firefly", "adobe generative", "content credentials",
            "adobe stock ai", "firefly 2", "firefly 3", "adobe express ai",
            "adobe photoshop generative",
        ],
        "Microsoft/Copilot":    [
            "microsoft designer", "bing image creator", "copilot image",
            "microsoft bing", "designer ai", "azure openai", "bing create",
        ],
        "Canva AI":             ["canva", "canva ai", "text to image canva"],
        "Pika":                 ["pika labs", "pikalabs", "pika 1.0", "pika 2.0", "pika 2.2"],
        "Runway":               [
            "runway", "runwayml", "gen-1", "gen-2", "gen-3",
            "runway gen", "runwayml.com", "runway alpha",
        ],
        "Kling":                ["kling", "kling ai", "kuaishou", "kwai-kolors"],
        "Luma AI":              ["luma ai", "lumalabs", "dream machine", "luma dream"],
        "Hailuo/MiniMax":       ["hailuo", "minimax", "hailuoai", "minimax video"],
        "Haiper":               ["haiper", "haiper ai"],
        "Wan/Alibaba":          ["wan ai", "alibaba wan", "tongyi", "wanx", "alibaba cloud ai"],
        # ── Stable Diffusion y derivados ────────────────────────────────
        "Stable Diffusion":     [
            "stable diffusion", "automatic1111", "a1111", "comfyui", "comfy ui",
            "invokeai", "diffusers", "safetensors", "dreambooth", "novelai",
            "stable-diffusion", "sd webui", "sdwebui", "stable diffusion webui",
            "vladmandic", "fooocus", "sd-next",
        ],
        "SDXL/Turbo/Lightning": [
            "sdxl", "stable diffusion xl", "sdxl-turbo", "sdxl lightning",
            "sdxl-lightning", "lcm", "latent consistency", "hyper-sd", "hyper sd",
        ],
        "Stable Cascade":       ["stable cascade", "würstchen", "wuerstchen"],
        "Flux":                 [
            "flux", "black forest labs", "bfl", "flux.1",
            "flux-dev", "flux-schnell", "flux-pro", "bfl.ml",
        ],
        "PixArt":               ["pixart", "pixart-alpha", "pixart-sigma"],
        "DeepFloyd IF":         ["deepfloyd", "deep floyd", "if-i", "if-ii"],
        "Kandinsky":            ["kandinsky", "ai-forever", "sber"],
        "Kolors":               ["kolors", "kwai-kolors"],
        "AuraFlow":             ["auraflow", "aura flow"],
        "CogView":              ["cogview", "cogview2", "cogview3", "thudm cogview"],
        "HunyuanDiT":          ["hunyuan", "hunyuandit", "tencent hunyuan"],
        "Janus/DeepSeek":       ["janus", "deepseek", "janus-pro", "deepseek janus"],
        "AnimateDiff":          ["animatediff", "animate diff", "motion module"],
        # ── Plataformas de consumo ───────────────────────────────────────
        "Leonardo.ai":          ["leonardo.ai", "leonardo ai", "leonardo creative"],
        "Ideogram":             ["ideogram", "ideogram ai", "ideogram v1", "ideogram v2", "ideogram v3"],
        "Playground":           ["playground ai", "playgroundai", "playground v2", "playground v3"],
        "Lexica":               ["lexica", "lexica.art", "lexica aperture"],
        "Dreamlike":            ["dreamlike", "dreamlike.art", "dreamlike diffusion"],
        "Artbreeder":           ["artbreeder", "artbreeder collage"],
        "StarryAI":             ["starryai", "starry ai"],
        "Wombo":                ["wombo", "dream by wombo", "wombo dream"],
        "Craiyon":              ["craiyon", "dalle-mini", "dall-e mini", "mini dalle"],
        "NightCafe":            ["nightcafe", "night cafe", "nightcafe creator"],
        "Gencraft":             ["gencraft"],
        "NovelAI":              ["novelai", "novel ai", "novelai diffusion"],
        "Mage.space":           ["mage.space", "mage space", "mage ai"],
        "Dezgo":                ["dezgo", "dezgo api"],
        "Getimg.ai":            ["getimg", "getimg.ai"],
        "Segmind":              ["segmind", "segmind api"],
        "Venice.ai":            ["venice.ai", "venice ai"],
        "SeaArt":               ["seaart", "sea art", "seaart.ai"],
        "Tensor.art":           ["tensor.art", "tensorart", "tensor art"],
        "Civitai":              ["civitai", "civit ai"],
        # ── Edición con IA ──────────────────────────────────────────────
        "Clipdrop/Stability":   ["clipdrop", "clip drop", "stability ai api", "stability.ai", "stabilityai"],
        "Remini":               ["remini", "remini ai"],
        "Luminar AI":           ["luminar ai", "luminar neo", "skylum"],
        "Topaz":                ["topaz", "topaz labs", "gigapixel ai", "topaz photo ai"],
        "PicsArt AI":           ["picsart", "picsart ai"],
        "Pixlr AI":             ["pixlr", "pixlr ai"],
        "Fotor AI":             ["fotor", "fotor ai"],
        "Hotpot.ai":            ["hotpot.ai", "hotpot ai"],
        "PhotoRoom":            ["photoroom", "photo room ai"],
        "Deep Dream":           ["deep dream", "deepdream", "google deepdream"],
        "FaceApp":              ["faceapp", "face app", "wireless lab"],
        "Meitu":                ["meitu", "meipai", "meitu ai"],
        "Kaiber":               ["kaiber", "kaiber ai"],
        "Prodia":               ["prodia", "prodia api"],
        "Replicate":            ["replicate", "replicate.com", "cog model"],
        "RunDiffusion":         ["rundiffusion", "run diffusion"],
        "PixAI":                ["pixai", "pixai.art"],
        "Perchance":            ["perchance", "perchance.org ai"],
        "Stablecog":            ["stablecog", "stable cog"],
        "CF Spark":             ["cf spark", "creative fabrica spark"],
        "Jasper Art":           ["jasper art", "jasperart"],
        "Kapwing AI":           ["kapwing", "kapwing ai"],
    }

    _REAL_CAMERA_MAKERS: List[str] = [
        # Cámaras profesionales / mirrorless / DSLR
        "canon", "nikon", "sony", "fujifilm", "olympus", "panasonic",
        "leica", "hasselblad", "phase one", "pentax", "sigma", "ricoh",
        "mamiya", "gopro", "dji", "insta360", "blackmagic",
        # Móviles — marcas principales
        "apple", "samsung", "huawei", "xiaomi", "oppo", "motorola",
        "google", "oneplus", "vivo", "realme", "honor", "poco",
        "nothing", "asus", "lg", "htc", "nokia", "zte", "tcl",
        "infinix", "tecno", "itel", "fairphone", "cat",
        # Otros dispositivos con cámara
        "microsoft", "amazon",
    ]

    _AI_HARDWARE_WORDS: List[str] = [
        "ai model", "gemini", "generator ai", "synthid", "diffusion",
    ]

    _PNG_AI_CHUNK_KEYS = [
        "parameters", "prompt", "Comment", "Description", "workflow",
        "negative_prompt", "model", "Software", "invokeai_metadata",
        "invokeai_graph", "comfy_prompt", "sd-metadata", "generation_data",
        "image_generation_data", "ai_generated", "generator", "model_hash",
        "vae", "cfg_scale", "sampler", "steps", "seed", "clip_skip",
        "lora_hashes", "hashes", "naidata", "source", "title",
        "fooocus_scheme", "fooocus_task", "kling_info", "video_frame",
    ]

    _KNOWN_SD_MODELS = [
        "realisticvision", "realistic vision", "epicrealism", "deliberate",
        "dreamshaper", "revanimated", "rev animated", "aom3", "orangemix",
        "counterfeit", "anything", "chilloutmix", "ghostmix", "cyberrealistic",
        "absolute reality", "absolutereality", "juggernautxl", "juggernaut xl",
        "realvisxl", "dreamshaper xl", "playground v2", "copax", "ponyxl",
        "pony diffusion", "novelai", "nai diffusion", "animefull", "anything v",
        "holodayo", "waifu diffusion", "toon crafter",
    ]

    _AI_EDITING_SOFTWARE = [
        "luminar ai", "luminar neo", "topaz", "gigapixel", "remini",
        "faceapp", "meitu", "picsart", "photoroom", "remove.bg", "clipdrop",
        "photoshop generative", "firefly", "lightroom ai", "denoise ai",
    ]

    # Firmas binarias de Grok/Aurora embebidas en el archivo
    _GROK_BINARY_SIGNATURES = [
        b"xai", b"aurora", b"grok", b"x.ai", b"xai\x00",
        b"generated by x", b"grok-image",
    ]

    _C2PA_XMP_MARKER      = "c2pa"
    _C2PA_JUMBF_SIGNATURE = b"jumb"
    _WEBP_XMP_SIGNATURE   = b"XMP "
    _JUMBF_SCAN_LIMIT     = 262144
    _XMP_EXTRACT_LIMIT    = 262144

    def analyze(self, data: bytes, img_pil: Image.Image) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "confirmed_ai":          False,
            "generator":             "",
            "real_camera":           False,
            "real_camera_field_count": 0,
            "c2pa_present":          False,
            "c2pa_ai_signed":        False,
            "raw_fields":            {},
            "signals":               [],
            "has_any_exif":          False,
            "is_social_media":      False,
        }
        text_pool: List[str] = []

        # -- Detectar Social Media por patrones de archivo (si el host lo provee en metadata o bytes) --
        # Nota: En sistemas reales, a veces el filename viene en un chunk del archivo o se infiere.
        # Aquí buscamos firmas binarias comunes o ausencia total de EXIF en formatos típicos.
        # -- Detectar Social Media por firmas binarias (WhatsApp/Telegram/Instagram) --
        scan_head = data[:8192].lower()
        social_signatures = [b"whatsapp", b"telegram", b"instagram", b"screenshot", b"webui"]
        if any(sig in scan_head for sig in social_signatures):
            result["is_social_media"] = True
            if b"webui" not in scan_head: # webui no es social media
                result["signals"].append("Origen detectado: Firma binaria de Red Social/Captura")
            else:
                result["is_social_media"] = False
        
        # Escaneo de firmas de WhatsApp/Telegram en chunks de texto (si existen)
        scan_head = data[:4096].lower()
        if b"whatsapp" in scan_head or b"telegram" in scan_head or b"instagram" in scan_head:
            result["is_social_media"] = True
            result["signals"].append("Origen detectado: Red Social (compresión probable)")

        # ── EXIF ─────────────────────────────────────────────────
        try:
            exif_obj = img_pil.getexif()
            if exif_obj:
                exif = {TAGS.get(k, str(k)): str(v) for k, v in exif_obj.items()}
                result["raw_fields"]["exif"] = exif
                result["has_any_exif"] = bool(exif)
                text_pool.extend(exif.values())

                make      = exif.get("Make", "").lower()
                model_tag = exif.get("Model", "").lower()
                software  = exif.get("Software", "").lower()
                has_gps   = "GPSInfo" in exif
                has_focal = "FocalLength" in exif

                is_real_brand   = any(m in make for m in self._REAL_CAMERA_MAKERS)
                is_google_pixel = "google" in make and "pixel" in model_tag
                has_ai_words    = any(
                    w in model_tag or w in software
                    for w in self._AI_HARDWARE_WORDS
                )

                if (is_real_brand or is_google_pixel) and not has_ai_words:
                    result["real_camera"] = True
                elif not is_real_brand and not is_google_pixel and not has_ai_words:
                    # V10.1: Detección genérica — si el EXIF tiene campos
                    # típicos de cámara, tratar como cámara real.
                    has_photo_fields = (
                        exif.get("DateTimeOriginal") or
                        exif.get("FNumber") or
                        exif.get("ExposureTime") or
                        exif.get("ISOSpeedRatings") or
                        exif.get("PhotographicSensitivity") or
                        has_focal
                    )
                    if has_photo_fields:
                        result["real_camera"] = True

                # Contar campos EXIF para TODAS las cámaras reales detectadas
                if result["real_camera"]:
                    cam_str = f"{exif.get('Make', '')} {exif.get('Model', '')}".strip()
                    real_fields = []
                    if exif.get("ExposureTime"):  real_fields.append("ExposureTime")
                    if exif.get("FNumber"):        real_fields.append("FNumber")
                    if exif.get("ISOSpeedRatings") or exif.get("PhotographicSensitivity"):
                        real_fields.append("ISO")
                    if exif.get("LensModel") or exif.get("LensMake"):
                        real_fields.append("LensInfo")
                    if has_gps:   real_fields.append("GPS")
                    if has_focal: real_fields.append("FocalLength")
                    if exif.get("DateTimeOriginal"): real_fields.append("DateTime")
                    result["real_camera_field_count"] = len(real_fields)
                    result["signals"].append(
                        f"EXIF de cámara real: {cam_str} [{', '.join(real_fields)}]"
                    )

                # Software IA embebido
                for ai_sw in self._AI_EDITING_SOFTWARE:
                    if ai_sw in software:
                        result["confirmed_ai"] = True
                        result["generator"] = result["generator"] or f"Edición IA ({ai_sw})"
                        result["signals"].append(f"Software IA en EXIF: '{ai_sw}'")
                        break

                # Detección de upscale IA por resolución exacta
                try:
                    w_img, h_img = img_pil.size
                    for bw, bh in [(512, 512), (512, 768), (768, 512), (1024, 1024)]:
                        for scale in [2, 4, 8]:
                            if w_img == bw * scale and h_img == bh * scale:
                                result["signals"].append(
                                    f"Resolución exacta de upscale IA {scale}x "
                                    f"({bw}×{bh} → {w_img}×{h_img})"
                                )
                except Exception:
                    pass

        except Exception as e:
            logger.debug(f"EXIF no disponible: {e}")

        # ── PNG / WebP chunks ────────────────────────────────────
        try:
            info = img_pil.info or {}
            result["raw_fields"]["img_info"] = {k: str(v)[:300] for k, v in info.items()}
            for key in self._PNG_AI_CHUNK_KEYS:
                val = info.get(key)
                if not val:
                    continue
                val_str = str(val)
                text_pool.append(val_str)

                if key == "parameters" and len(val_str) > 10:
                    result["confirmed_ai"] = True
                    result["generator"]    = "Stable Diffusion (A1111/WebUI)"
                    result["signals"].append("PNG chunk 'parameters' (firma A1111)")
                elif key in ("prompt", "workflow") and val_str.strip().startswith("{"):
                    result["confirmed_ai"] = True
                    result["generator"]    = result["generator"] or "Stable Diffusion (ComfyUI)"
                    result["signals"].append(f"PNG chunk '{key}' con JSON workflow")
                elif key in ("invokeai_metadata", "invokeai_graph"):
                    result["confirmed_ai"] = True
                    result["generator"]    = "InvokeAI"
                    result["signals"].append(f"Chunk InvokeAI: '{key}'")
                elif key in ("fooocus_scheme", "fooocus_task"):
                    result["confirmed_ai"] = True
                    result["generator"]    = "Fooocus"
                    result["signals"].append(f"Chunk Fooocus: '{key}'")
                elif key in ("model", "model_hash", "steps", "cfg_scale", "sampler"):
                    val_lower = val_str.lower()
                    for model_name in self._KNOWN_SD_MODELS:
                        if model_name in val_lower:
                            result["confirmed_ai"] = True
                            result["generator"]    = result["generator"] or f"SD ({model_name})"
                            result["signals"].append(f"Modelo SD en '{key}': {model_name}")
                            break
                elif key == "source" and "novelai" in val_str.lower():
                    result["confirmed_ai"] = True
                    result["generator"]    = "NovelAI"
                    result["signals"].append("Firma NovelAI en chunk 'source'")
        except Exception as e:
            logger.debug(f"PNG/WebP info: {e}")

        # ── XMP ──────────────────────────────────────────────────
        try:
            xmp_data: Optional[str] = None
            for xk in ["XML:com.adobe.xmp", "xmp", "XMP"]:
                if xk in (img_pil.info or {}):
                    xmp_data = str(img_pil.info[xk])
                    break
            if not xmp_data and self._WEBP_XMP_SIGNATURE in data[:self._JUMBF_SCAN_LIMIT]:
                start = data.find(self._WEBP_XMP_SIGNATURE)
                if start != -1:
                    xmp_data = data[start:start + self._XMP_EXTRACT_LIMIT].decode("ascii", errors="ignore")
            if xmp_data:
                result["raw_fields"]["xmp_snippet"] = xmp_data[:500]
                text_pool.append(xmp_data)
                if self._C2PA_XMP_MARKER in xmp_data.lower():
                    result["c2pa_present"] = True
                    result["signals"].append("Manifesto C2PA en XMP")
        except Exception as e:
            logger.debug(f"XMP: {e}")

        # ── C2PA binario ──────────────────────────────────────────
        try:
            scan = data[:self._JUMBF_SCAN_LIMIT]
            if self._C2PA_JUMBF_SIGNATURE in scan:
                result["c2pa_present"] = True
                result["signals"].append("Manifesto C2PA/JUMBF en binario")
                scan_lower = scan.lower()
                if b"google" in scan_lower or b"deepmind" in scan_lower:
                    text_pool.append("google deepmind c2pa")
        except Exception as e:
            logger.debug(f"C2PA: {e}")

        # ── Escaneo binario Grok/Aurora ───────────────────────────
        try:
            scan_lower = data[:self._JUMBF_SCAN_LIMIT].lower()
            for sig in self._GROK_BINARY_SIGNATURES:
                if sig in scan_lower:
                    result["confirmed_ai"] = True
                    result["generator"]    = "Grok/Aurora"
                    result["signals"].append(
                        f"Firma binaria Grok/Aurora: '{sig.decode('utf-8', errors='replace')}'"
                    )
                    break
        except Exception as e:
            logger.debug(f"Binario Grok: {e}")

        # ── Búsqueda de firmas IA en texto acumulado ─────────────
        if not result["confirmed_ai"]:
            combined = " ".join(text_pool).lower()
            for generator, tokens in self._AI_SIGNATURES.items():
                matched = [t for t in tokens if t in combined]
                if matched:
                    result["confirmed_ai"] = True
                    result["generator"]    = generator
                    result["signals"].append(
                        f"Firma IA en metadata: '{matched[0]}' → {generator}"
                    )
                    if result["c2pa_present"]:
                        result["c2pa_ai_signed"] = True
                    break

        # ── Detección de resolución nativa IA (sin metadata) ─────
        if not result["confirmed_ai"]:
            try:
                w_img, h_img = img_pil.size
                if (w_img, h_img) in _AI_NATIVE_RESOLUTIONS_SET and not result["real_camera"]:
                    result["signals"].append(
                        f"Resolución nativa de generador IA: {w_img}×{h_img}"
                    )
                    # No marcar confirmed_ai — solo señal de apoyo
            except Exception:
                pass

        return result


# ═══════════════════════════════════════════════════════════════
# MÓDULO 4: ELA — ERROR LEVEL ANALYSIS
# ═══════════════════════════════════════════════════════════════
class ELAAnalyzer:
    _FALLBACK: Dict[str, Any] = {
        "ela_mean": None, "ela_std": None, "ela_max": None,
        "ela_region_std": None, "ela_uniformity": None,
        "tampered_hotspot": None, "partial_ai_suspected": False,
        "heatmap_b64": None,
    }

    def __init__(self, quality: int = 95):
        self.quality = quality

    def analyze(self, data: bytes, img_pil: Image.Image) -> Dict[str, Any]:
        try:
            buf = io.BytesIO()
            img_rgb = img_pil.convert("RGB")
            img_rgb.save(buf, format="JPEG", quality=self.quality, subsampling=0)
            buf.seek(0)
            img_recomp = Image.open(buf).convert("RGB")

            orig_arr   = np.array(img_rgb, dtype=np.float32)
            recomp_arr = np.array(img_recomp, dtype=np.float32)
            ela_diff   = np.abs(orig_arr - recomp_arr)
            ela_gray   = ela_diff.mean(axis=2)

            ela_mean = float(ela_gray.mean())
            ela_std  = float(ela_gray.std())
            ela_max  = float(ela_gray.max())

            h, w   = ela_gray.shape
            rh, rw = h // 4, w // 4
            region_means = [
                float(ela_gray[i*rh:(i+1)*rh, j*rw:(j+1)*rw].mean())
                for i in range(4) for j in range(4)
                if ela_gray[i*rh:(i+1)*rh, j*rw:(j+1)*rw].size > 0
            ]
            ela_region_std = float(np.std(region_means)) if region_means else 0.0
            ela_uniformity = float(
                1.0 - np.clip(ela_region_std / (ela_mean + 1e-5), 0.0, 1.0)
            )

            grid_size = 16
            rh16, rw16 = max(1, h // grid_size), max(1, w // grid_size)
            max_error = 0.0
            tampered: Optional[Dict[str, Any]] = None
            for i in range(grid_size):
                for j in range(grid_size):
                    patch = ela_gray[i*rh16:(i+1)*rh16, j*rw16:(j+1)*rw16]
                    if patch.size > 0:
                        pm = float(patch.mean())
                        if pm > max_error:
                            max_error = pm
                            tampered = {
                                "x": round(j / grid_size, 2),
                                "y": round(i / grid_size, 2),
                                "error_score": round(pm, 2),
                            }

            partial_ai = bool(max_error > (ela_mean * 2.5) and ela_region_std > 15.0)

            visual   = np.clip(ela_gray * 15.0, 0, 255).astype(np.uint8)
            heatmap  = cv2.applyColorMap(visual, cv2.COLORMAP_INFERNO)
            orig_bgr = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
            overlay  = cv2.addWeighted(orig_bgr, 0.4, heatmap, 0.6, 0)
            _, enc   = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
            heatmap_b64 = base64.b64encode(enc).decode("utf-8")

            return {
                "ela_mean":             round(ela_mean, 3),
                "ela_std":              round(ela_std, 3),
                "ela_max":              round(ela_max, 3),
                "ela_region_std":       round(ela_region_std, 3),
                "ela_uniformity":       round(ela_uniformity, 4),
                "tampered_hotspot":     tampered,
                "partial_ai_suspected": partial_ai,
                "heatmap_b64":          heatmap_b64,
            }
        except Exception as e:
            logger.warning(f"ELA falló: {e}")
            return dict(self._FALLBACK)


# ═══════════════════════════════════════════════════════════════
# MÓDULO 5: FINGERPRINTING ESPECTRAL
# ═══════════════════════════════════════════════════════════════
class AIGeneratorFingerprinter:
    """Detecta patrones visuales específicos de cada familia de generador."""

    @staticmethod
    def analyze(
        gray: np.ndarray,
        img_cv: np.ndarray,
        features: Dict[str, float],
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "family_scores": {},
            "top_family": None,
            "top_score": 0.0,
            "signals": [],
            "fft_grid_detected": False,
            "edge_halo_detected": False,
            "noise_injection_detected": False,
        }
        h, w = gray.shape

        # A. Grid FFT — SD / Flux / PixArt (latent space 8×8 / 16×16)
        try:
            mag = np.abs(np.fft.fftshift(np.fft.fft2(gray.astype(float))))
            mag_log = np.log1p(mag)
            cy, cx = h // 2, w // 2
            grid_score = 0.0
            for divisor in [8, 16, 32]:
                fy, fx = h // divisor, w // divisor
                if fy > 0 and fx > 0:
                    rs = max(2, min(fy, fx) // 4)
                    for dy, dx in [(fy, 0), (0, fx), (fy, fx), (-fy, fx)]:
                        py, px = cy + dy, cx + dx
                        if 0 < py < h and 0 < px < w:
                            peak = mag_log[py, px]
                            nb = mag_log[
                                max(0, py-rs):min(h, py+rs),
                                max(0, px-rs):min(w, px+rs)
                            ]
                            if nb.size > 0:
                                nm = float(np.mean(nb))
                                if nm > 0:
                                    grid_score = max(grid_score, peak / nm)
            if grid_score > 1.8:
                result["fft_grid_detected"] = True
                result["signals"].append(
                    f"Patrón de rejilla VAE/DCT en FFT (ratio={grid_score:.2f}) → SD/Flux/PixArt"
                )
                result["family_scores"]["stablediffusion"] = (
                    result["family_scores"].get("stablediffusion", 0) +
                    min(35.0, grid_score * 12.0)
                )
        except Exception:
            pass

        # B. Halos en bordes — Midjourney V5/V6
        try:
            edges = cv2.Canny(gray, 50, 150)
            if float(np.mean(edges > 0)) > 0.02:
                kernel = np.ones((5, 5), np.float32) / 25
                blurred = cv2.filter2D(gray.astype(np.float32), -1, kernel)
                halo_diff = np.abs(gray.astype(np.float32) - blurred)
                ez = halo_diff[edges > 0]
                sz = halo_diff[edges == 0]
                if ez.size > 100 and sz.size > 100:
                    halo_ratio = float(np.mean(ez)) / (float(np.mean(sz)) + 1e-5)
                    if 2.0 < halo_ratio < 5.5:
                        result["edge_halo_detected"] = True
                        result["signals"].append(
                            f"Halo en bordes (ratio={halo_ratio:.2f}) → Midjourney/SDXL"
                        )
                        result["family_scores"]["midjourney"] = (
                            result["family_scores"].get("midjourney", 0) +
                            min(25.0, (5.5 - halo_ratio) * 8.0)
                        )
        except Exception:
            pass

        # C. Ruido inyectado artificialmente — DALL-E 3
        try:
            nl  = features.get("noise_level")
            bcs = features.get("block_consistency_std")
            if nl is not None and bcs is not None:
                if 0.8 < nl < 4.0 and bcs < 500.0:
                    su = 1.0 - min(1.0, bcs / 1000.0)
                    if su > 0.6:
                        result["noise_injection_detected"] = True
                        result["signals"].append(
                            f"Ruido espacialmente uniforme (n={nl:.2f}, uniformity={su:.2f}) → DALL-E 3/MJ"
                        )
                        result["family_scores"]["dalle"] = (
                            result["family_scores"].get("dalle", 0) + min(20.0, su * 22.0)
                        )
        except Exception:
            pass

        # C-bis. Superficie pulida — Grok/Flux (ruido muy bajo)
        try:
            nl = features.get("noise_level")
            if nl is not None and nl < 1.5:
                boost = min(35.0, (1.5 - nl) * 80.0)
                result["signals"].append(
                    f"Superficie sintéticamente pulida (n={nl:.2f}) → Grok/Flux"
                )
                result["family_scores"]["grok"] = (
                    result["family_scores"].get("grok", 0) + boost
                )
        except Exception:
            pass

        # D. Coeficiente de variación de saturación — Grok/Aurora
        try:
            hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
            sat = hsv[:, :, 1].astype(float)
            sat_cv = float(np.std(sat) / (np.mean(sat) + 1e-5))
            if sat_cv < 0.40:
                result["signals"].append(
                    f"Croma ultra-estrecha (CV={sat_cv:.3f}) → Grok/Flux"
                )
                result["family_scores"]["grok"] = (
                    result["family_scores"].get("grok", 0) +
                    min(30.0, (0.40 - sat_cv) * 80.0)
                )
            elif sat_cv < 0.50:
                result["family_scores"]["dalle"] = (
                    result["family_scores"].get("dalle", 0) + 10.0
                )
        except Exception:
            pass

        # E. Suavizado central selectivo — SD/MJ (piel sintética)
        try:
            ch, cw = h // 4, w // 4
            center = gray[ch:3*ch, cw:3*cw]
            border = np.concatenate([
                gray[:ch, :].ravel(), gray[3*ch:, :].ravel(),
                gray[:, :cw].ravel(), gray[:, 3*cw:].ravel(),
            ])
            if center.size > 1000 and border.size > 1000:
                ratio = float(np.var(border)) / (float(np.var(center)) + 1e-5)
                if ratio > 4.0:
                    result["signals"].append(
                        f"Suavizado central selectivo (ratio={ratio:.1f}) → SD/MJ"
                    )
                    result["family_scores"]["midjourney"] = (
                        result["family_scores"].get("midjourney", 0) +
                        min(20.0, (ratio - 4.0) * 5.0)
                    )
                    result["family_scores"]["stablediffusion"] = (
                        result["family_scores"].get("stablediffusion", 0) +
                        min(15.0, (ratio - 4.0) * 3.5)
                    )
        except Exception:
            pass

        # F. Sobre-nitidez — Flux / SDXL Lightning
        try:
            lv = features.get("laplacian_var")
            fr = features.get("fft_ratio")
            if lv is not None and fr is not None:
                if lv > 3000 and fr > 0.45:
                    result["signals"].append(
                        f"Sobre-nitidez artificial (LapVar={lv:.0f}, FFT={fr:.2f}) → Flux/SDXL Lightning"
                    )
                    result["family_scores"]["flux"] = (
                        result["family_scores"].get("flux", 0) +
                        min(25.0, (fr - 0.45) * 80.0)
                    )
        except Exception:
            pass

        # G. NUEVO V9: Entropía local baja — detecta texturas IA uniformes
        try:
            lte = features.get("local_texture_entropy")
            if lte is not None and lte < 4.2:
                ai_entropy_score = min(25.0, (4.2 - lte) * 20.0)
                result["signals"].append(
                    f"Entropía local de textura baja (lte={lte:.2f}) → textura IA"
                )
                result["family_scores"]["grok"] = (
                    result["family_scores"].get("grok", 0) + ai_entropy_score * 0.6
                )
                result["family_scores"]["stablediffusion"] = (
                    result["family_scores"].get("stablediffusion", 0) + ai_entropy_score * 0.4
                )
        except Exception:
            pass

        # H. NUEVO V9: Ausencia de dominancia verde
        try:
            gd = features.get("green_channel_dominance")
            if gd is not None and gd < 1.05:
                result["signals"].append(
                    f"Sin dominancia de canal verde (gd={gd:.2f}) → sin sensor Bayer → IA"
                )
                for fam in ["grok", "dalle", "midjourney"]:
                    result["family_scores"][fam] = (
                        result["family_scores"].get(fam, 0) + 10.0
                    )
        except Exception:
            pass

        # I. NUEVO V10: Paleta de color estrecha → Grok/Flux
        try:
            cc = features.get("color_cluster_count")
            hs = features.get("color_hue_spread")
            if cc is not None and hs is not None:
                if cc <= 5 and hs < 60.0:
                    palette_score = min(30.0, (6 - cc) * 8.0 + (60.0 - hs) * 0.3)
                    result["signals"].append(
                        f"Paleta de color estrecha ({cc} clusters, hue_spread={hs:.0f}°) → Grok/Flux"
                    )
                    result["family_scores"]["grok"] = (
                        result["family_scores"].get("grok", 0) + palette_score * 0.7
                    )
                    result["family_scores"]["flux"] = (
                        result["family_scores"].get("flux", 0) + palette_score * 0.3
                    )
                elif cc <= 4:
                    result["family_scores"]["grok"] = (
                        result["family_scores"].get("grok", 0) + 12.0
                    )
        except Exception:
            pass

        # J. NUEVO V10: FFT mid-band ringing → Grok/Aurora upsampler
        try:
            mbr = features.get("fft_midband_ring")
            if mbr is not None and mbr > 1.5:
                ring_score = min(30.0, (mbr - 1.5) * 25.0)
                result["signals"].append(
                    f"Ringing espectral mid-band (ratio={mbr:.2f}) → artefacto upsampler Grok/Aurora"
                )
                result["family_scores"]["grok"] = (
                    result["family_scores"].get("grok", 0) + ring_score
                )
        except Exception:
            pass

        # K. NUEVO V10: Simetría bilateral anormal → IA genérica
        try:
            bsym = features.get("bilateral_symmetry")
            if bsym is not None and bsym > 0.85:
                sym_score = min(25.0, (bsym - 0.85) * 180.0)
                result["signals"].append(
                    f"Simetría bilateral anormal (corr={bsym:.3f}) → generación IA"
                )
                for fam in ["grok", "dalle", "midjourney"]:
                    result["family_scores"][fam] = (
                        result["family_scores"].get(fam, 0) + sym_score * 0.33
                    )
        except Exception:
            pass

        # L. NUEVO V10: Distribución uniforme de bordes → IA
        try:
            eqr = features.get("edge_quad_ratio")
            if eqr is not None and eqr < 0.15:
                edge_score = min(20.0, (0.15 - eqr) * 150.0)
                result["signals"].append(
                    f"Bordes distribuidos uniformemente (eqr={eqr:.3f}) → sin foco óptico → IA"
                )
                result["family_scores"]["grok"] = (
                    result["family_scores"].get("grok", 0) + edge_score * 0.5
                )
                result["family_scores"]["dalle"] = (
                    result["family_scores"].get("dalle", 0) + edge_score * 0.3
                )
                result["family_scores"]["flux"] = (
                    result["family_scores"].get("flux", 0) + edge_score * 0.2
                )
        except Exception:
            pass

        # Consolidar familia dominante
        if result["family_scores"]:
            lap_v = features.get("laplacian_var")
            gd_v  = features.get("green_channel_dominance")
            bcs_v = features.get("block_consistency_std")
            ca_v  = features.get("chromatic_aberration")
            lte_v = features.get("local_texture_entropy")

            is_pro_sharp = False
            if lap_v is not None and lap_v > 800.0:
                if (gd_v is not None and gd_v >= 1.15) or \
                   (bcs_v is not None and bcs_v > 800.0) or \
                   (ca_v is not None and ca_v >= 0.35):
                    is_pro_sharp = True
            
            # V10.3: No restringir si la convergencia de familia es muy fuerte o IA confirmada por textura
            if is_pro_sharp:
                top_temp = max(result["family_scores"].values()) if result["family_scores"] else 0
                if top_temp > 70.0 or (lte_v is not None and lte_v < 1.3):
                    is_pro_sharp = False
                    result["signals"].append("⚠️ Firma espectral liberada: Evidencia micro-estructural IA detectada.")

            if is_pro_sharp:
                for fam in result["family_scores"]:
                    result["family_scores"][fam] *= 0.15
                result["signals"].append("Firma espectral restringida al 15% (nitidez de sensor óptico confirmada)")

            top = max(result["family_scores"].items(), key=lambda x: x[1])
            result["top_family"] = top[0]
            result["top_score"]  = round(top[1], 1)

        return result



# ═══════════════════════════════════════════════════════════════
# MÓDULO 6: CLASIFICADOR GROK INDEPENDIENTE (V10.2)
# ═══════════════════════════════════════════════════════════════
class GrokClassifier:
    """
    Clasificador independiente para Grok/Aurora/Flux/Gemini.
    Opera en un pipeline SEPARADO del MetaClassifier para evitar
    que la detección de Grok cause falsos positivos en fotos reales.

    Principio: si la imagen tiene EXIF de cámara → 0% inmediato.
    Si no tiene EXIF, evalúa 10 señales forenses específicas de IA.
    """

    @staticmethod
    def classify(
        features: Dict[str, float],
        metadata: Optional[Dict[str, Any]],
        img_size: Tuple[int, int],
    ) -> Tuple[float, List[str]]:
        """Retorna (probabilidad_grok, señales)."""

        # ── GATE 1: EXIF de cámara real → imposible que sea Grok ────
        if metadata and metadata.get("real_camera"):
            return 0.0, ["Clasificador Grok: EXIF de cámara real → descartado"]

        # ── GATE 2: IA confirmada por metadata → ya resuelta ────────
        if metadata and metadata.get("confirmed_ai"):
            return 95.0, [f"Clasificador Grok: IA confirmada ({metadata.get('generator', '?')})"]

        score  = 0.0
        max_sc = 0.0
        signals: List[str] = []

        has_exif = bool(metadata and metadata.get("has_any_exif", False))

        # ── 1. Sin EXIF (señal débil pero necesaria) ────────────────
        if not has_exif:
            score += 15.0
            signals.append("Sin EXIF fotográfico")
        max_sc += 15.0

        # ── 1b. Detección de Gráficos Digitales / UI (Screenshot) ────
        # Si tiene muy pocos colores y sombras perfectas, es probable que
        # sea una captura de pantalla, no una foto ni IA.
        cc_val = features.get("color_cluster_count")
        se_val = features.get("shadow_entropy")
        is_digital_ui = False
        if cc_val is not None and se_val is not None:
            is_digital_ui = bool(cc_val <= 3 and se_val > 4.8)
        
        if is_digital_ui:
            signals.append("🔍 Detectado Gráfico Digital/UI (no foto)")

        # ── 2. Resolución nativa de generador IA ────────────────────
        w, h = img_size
        if (w, h) in _AI_NATIVE_RESOLUTIONS_SET:
            score += 15.0
            max_sc += 15.0
            signals.append(f"Resolución nativa IA: {w}×{h}")

        # ── 3-12. Señales forenses (cada una con su peso) ───────────
        # V10.3: Evaluación secuencial para evitar duplicidad de pesos
        checks_nested = [
            ("local_texture_entropy", [
                (lambda v: v < 1.0, 30.0, "Textura microscópica NULA (Confirmación IA)"),
                (lambda v: v < 1.8, 20.0, "Textura microescala artificial (Grok/Flux)"),
                (lambda v: v < 3.8, 12.0, "Textura microescala suave")
            ]),
            ("fft_midband_ring", [
                (lambda v: v > 3.2, 28.0, "Ringing mid-band CRÍTICO (Confirmación IA)"),
                (lambda v: v > 2.2, 18.0, "Ringing mid-band del upsampler"),
                (lambda v: v > 1.5, 10.0, "Ringing espectral leve")
            ]),
            ("noise_level", [
                (lambda v: v < 1.0, 15.0, "Ruido sintéticamente bajo (Grok/Flux)"),
                (lambda v: v < 1.8, 8.0,  "Luminancia suave")
            ]),
        ]

        # Señales binarias simples
        checks_flat = [
            ("shadow_entropy",        lambda v: v < 2.5,   12.0, "Sombras sintéticas"),
            ("gradient_smoothness",   lambda v: v < 8.0,   10.0, "Gradientes artificialmente suaves"),
            ("dct_block_uniformity",  lambda v: v < 25.0,  10.0, "DCT uniforme"),
            ("green_channel_dominance", lambda v: v < 1.10, 10.0, "Sin dominancia verde (no Bayer)"),
            ("color_cluster_count",   lambda v: v <= 5,     8.0, "Paleta de color estrecha"),
            ("bilateral_symmetry",    lambda v: v > 0.85,   8.0, "Simetría bilateral anormal"),
        ]

        conditions_met = 0
        if not has_exif:
             conditions_met += 1 # Sin EXIF es media señal

        for feat_name, thresholds in checks_nested:
            val = features.get(feat_name)
            if val is not None:
                max_sc += thresholds[0][1] # El peso máximo posible para esta feature
                for check_fn, weight, label in thresholds:
                    if check_fn(val):
                        score += weight
                        conditions_met += 1
                        signals.append(f"{label} ({feat_name}={val:.2f})")
                        break

        for feat_name, check_fn, weight, label in checks_flat:
            val = features.get(feat_name)
            if val is not None:
                max_sc += weight
                if check_fn(val):
                    score += weight
                    conditions_met += 1
                    signals.append(f"{label} ({feat_name}={val:.2f})")

        ca_val = features.get("chromatic_aberration")
        lte_val = features.get("local_texture_entropy")
        fft_ring = features.get("fft_midband_ring")

        # Condición de "Convergencia IA Indiscutible"
        is_extreme_ai = False
        if lte_val is not None:
            is_extreme_ai = bool(
                (conditions_met >= 7) or
                (lte_val < 0.3) or
                (fft_ring is not None and lte_val < 1.8 and fft_ring > 2.8 and conditions_met >= 4)
            )

        lap   = features.get("laplacian_var")
        gd    = features.get("green_channel_dominance")
        b_std = features.get("block_consistency_std")

        # Escudo pro-óptico: requiere que las biometrías existan y den positivo
        is_pro_sharp = False
        if lap is not None and lap > 800.0:
            if (gd is not None and gd >= 1.15) or \
               (b_std is not None and b_std > 800.0) or \
               (ca_val is not None and ca_val >= 0.35):
                is_pro_sharp = True

        # V10.3: El escudo se debilita si hay señales micro-estructurales IA extremas
        # o si hay una convergencia masiva de señales.
        if is_pro_sharp:
            if is_extreme_ai or conditions_met >= 9:
                is_pro_sharp = False
                signals.append("⚠️ Escudo Pro-óptico desactivado por convergencia IA indiscutible.")
            else:
                score = score * 0.35
                signals.append("🛡 Escudo Pro-óptico: Señales suavizadas/Ringing asumen origen en post-proceso de cámara real.")

        # Aplicar penalizaciones (solo si NO es IA extrema confirmada micro-estructuralmente)
        if not is_extreme_ai:
            penalties = [
                ("chromatic_aberration",   lambda v: v > 1.0,   25.0),
                ("chromatic_aberration",   lambda v: v > 0.5,   15.0),
                ("noise_level",            lambda v: v > 2.5,   25.0),
                ("laplacian_var",          lambda v: v > 4000,  20.0),
                ("local_texture_entropy",  lambda v: v > 5.5,   30.0),
            ]
            for feat_name, check_fn, weight in penalties:
                val = features.get(feat_name)
                if val is not None and check_fn(val):
                    score = max(0.0, score - weight)
        else:
            signals.append("🔍 Penalizaciones orgánicas anuladas por confirmación micro-estructural IA")

        # ── Calcular probabilidad ───────────────────────────────────
        prob = (score / max_sc) * 100.0 if max_sc > 0 else 0.0

        # ── Bonus por convergencia
        if conditions_met >= 9 and not has_exif:
            prob = min(100.0, prob * 1.25)
        elif conditions_met >= 6 and not has_exif:
            prob = min(100.0, prob * 1.10)

        if ca_val is not None and ca_val > 0.6 and not is_extreme_ai and conditions_met < 6:
            prob = min(prob, 35.0)
            if not any("Aberración cromática" in s for s in signals):
                signals.append(f"🔍 Freno forense: Aberración cromática detectada ({ca_val:.2f})")


        # Digital UI (Screenshot) freno adicional
        if is_digital_ui:
            prob = min(prob, 25.0)

        if has_exif:
            prob = min(prob, 30.0)

        prob, signals = round(prob, 1), signals
        
        # Escudo Social Media: Si no hay EXIF y es sospechosa de ser redes sociales, 
        # bajamos la confianza de señales que la compresión imita.
        if metadata and metadata.get("is_social_media") and not metadata.get("confirmed_ai"):
            # Si el veredicto es IA pero no es "Extrema", bajamos la prob
            if prob > 50 and not is_extreme_ai:
                prob = prob * 0.7  # Reducción del 30%
                signals.append("🛡 Escudo Social Media: Probabilidad reducida por posible compresión de red social")

        return prob, signals


# ═══════════════════════════════════════════════════════════════
# MÓDULO 2: META-CLASIFICADOR
# ═══════════════════════════════════════════════════════════════
class MetaClassifier:
    """Sistema probabilístico unificado por agregación ponderada de señales."""

    def __init__(self, config: Optional[ThresholdConfig] = None):
        self.cfg = config or ThresholdConfig()

    def evaluate(
        self,
        features:    Dict[str, Any],
        nn_prob:     float,
        metadata:    Optional[Dict[str, Any]] = None,
        ela:         Optional[Dict[str, Any]] = None,
        fingerprint: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, str, str, List[str]]:
        cfg      = self.cfg
        score    = 0.0
        max_sc   = 0.0
        reasons: List[str] = []
        is_social = bool(metadata and metadata.get("is_social_media"))
        has_real_cam = bool(metadata and metadata.get("real_camera"))

        # V10.3: Pre-conteo de señales IA para bypass de escudos
        # V10.3: Pre-conteo de señales IA para bypass de escudos (con pesos dinámicos)
        def count_ai_sigs(rs):
            c = 0
            for r in rs:
                rl = r.lower()
                if "nativa ia" in rl or "native ia" in rl: c += 3
                elif "firma ia" in rl: c += 5
                elif any(x in rl for x in ["artificial", "ia", "ringing", "espectral", "sintética"]):
                    c += 1
            return c

        # ── A: SHADOW ENTROPY ──────────────────────────────────────
        s_ent = features.get("shadow_entropy")
        if s_ent is not None:
            max_sc += cfg.shadow_entropy_weight
            if s_ent < cfg.shadow_entropy_ai_max:
                p = min(cfg.shadow_entropy_weight, (cfg.shadow_entropy_ai_max - s_ent) * 45.0)
                score += p
                reasons.append(f"Sombras sintéticamente puras (E={s_ent:.2f})")
            elif s_ent > cfg.shadow_entropy_real_min:
                score -= 15.0

        # ── C: RUIDO RESIDUAL — condicional a EXIF/Social Media ─────
        noise = features.get("noise_level")
        if noise is not None:
            max_sc += cfg.noise_weight
            noise_thr = 1.2 if has_real_cam else (cfg.noise_ai_max * 0.7 if is_social else cfg.noise_ai_max)
            if noise < noise_thr:
                p = min(cfg.noise_weight, (noise_thr - noise) * 35.0)
                score += p
                reasons.append(f"Carencia anómala de luminancia (n={noise:.2f})")

        # ── D: FFT RATIO ───────────────────────────────────────────

        # ── D: FFT RATIO ───────────────────────────────────────────
        fft_ratio = features.get("fft_ratio")
        if fft_ratio is not None:
            max_sc += cfg.fft_weight
            if fft_ratio > cfg.fft_ai_min:
                p = min(cfg.fft_weight, (fft_ratio - cfg.fft_ai_min) * 100.0)
                score += p
                reasons.append(f"Patrón de reconstrucción espectral (FFT={fft_ratio:.2f})")

        # ── G: HOMOGENEIDAD DE COLOR — condicional a EXIF ──────────
        sat_var  = features.get("saturation_var")
        val_var  = features.get("value_var")
        sat_mean = features.get("saturation_mean")
        
        if sat_var is not None and val_var is not None and sat_mean is not None:
            image_has_color = sat_mean > 25.0
            grok_w_eff = cfg.grok_weight if not has_real_cam else cfg.grok_weight * 0.20
            max_sc += grok_w_eff
            if image_has_color and sat_var < cfg.grok_sat_max and val_var < cfg.grok_val_max:
                p = min(grok_w_eff, (cfg.grok_val_max - val_var) * 0.12 + (cfg.grok_sat_max - sat_var) * 0.20)
                score += p
                reasons.append(
                    f"Uniformidad de color {'(sin EXIF) ' if not has_real_cam else '(EXIF, reducido) '}"
                    f"(sat_var={sat_var:.0f}, val_var={val_var:.0f})"
                )

        # ── E: ELA — reducido si hay EXIF de cámara ─────────────────
        if ela:
            ela_region_std = ela.get("ela_region_std")
            ela_mean_v     = ela.get("ela_mean")
            ela_uniformity = ela.get("ela_uniformity")
            
            if ela_region_std is not None and ela_mean_v is not None:
                ela_w_eff = cfg.ela_weight * (0.30 if has_real_cam else 1.0)
                max_sc += ela_w_eff

                if ela_region_std < cfg.ela_region_std_ai_max and ela_mean_v > cfg.ela_mean_ai_min:
                    ela_sc = min(
                        ela_w_eff,
                        (cfg.ela_region_std_ai_max - ela_region_std)
                        / cfg.ela_region_std_ai_max
                        * ela_w_eff
                        * (ela_uniformity ** 1.5 if ela_uniformity is not None else 1.0),
                    )
                    score += ela_sc
                    reasons.append(
                        f"ELA uniforme (region_std={ela_region_std:.2f}, mean={ela_mean_v:.2f})"
                    )
                elif ela.get("partial_ai_suspected"):
                    score += ela_w_eff * 0.85
                    hspot = ela.get("tampered_hotspot") or {}
                    reasons.append(
                        f"Inpainting/Manipulación Regional "
                        f"[X:{hspot.get('x', 0):.2f}, Y:{hspot.get('y', 0):.2f}] "
                        f"(Error:{hspot.get('error_score', 0)})"
                    )
                elif ela_region_std > cfg.ela_region_std_ai_max * 2.5:
                    score -= 20.0
                    reasons.append(f"ELA heterogéneo (region_std={ela_region_std:.2f}) → compresión previa")

        # ── F: METADATA ────────────────────────────────────────────
        if metadata:
            if metadata.get("confirmed_ai"):
                score  += cfg.metadata_confirmed_ai_weight
                max_sc += cfg.metadata_confirmed_ai_weight
                reasons.extend(metadata.get("signals", []))
                reasons.append(f"Generador: {metadata.get('generator', 'IA desconocida')}")

            elif metadata.get("c2pa_present") and not metadata.get("c2pa_ai_signed"):
                score  += cfg.metadata_c2pa_unsigned_weight
                max_sc += cfg.metadata_c2pa_unsigned_weight
                reasons.append("C2PA sin firma IA conocida (origen ambiguo)")

            elif metadata.get("real_camera"):
                extra_fields = metadata.get("real_camera_field_count", 0)
                bonus  = min(extra_fields * 5.0, 25.0)
                total  = cfg.metadata_real_camera_weight - bonus
                score  += total
                max_sc += abs(total)
                reasons.extend(metadata.get("signals", []))
            else:
                max_sc += 10.0

        # ── H: AUSENCIA DE EXIF (nueva V9) ─────────────────────────
        if metadata:
            has_any_exif = metadata.get("has_any_exif", False)
            has_cam_make = bool(
                metadata.get("real_camera") or
                (metadata.get("raw_fields", {}).get("exif", {}).get("Make"))
            )
            if not has_any_exif and not metadata.get("confirmed_ai"):
                score  += 4.0
                max_sc += 15.0
                reasons.append("Sin EXIF fotográfico (imagen descargada/huérfana)")
            elif has_any_exif and not has_cam_make:
                score  += 6.0
                max_sc += 10.0

        # ── K: ABERRACIÓN CROMÁTICA (nueva V9) ─────────────────────
        ca_score = features.get("chromatic_aberration")
        if ca_score is not None:
            max_sc += cfg.ca_weight
            if ca_score < cfg.ca_ai_max and not has_real_cam:
                if block_std is not None and block_std < 800.0:
                    p = min(cfg.ca_weight, (cfg.ca_ai_max - ca_score) * 60.0)
                    score += p
                    reasons.append(f"Sin aberración cromática (CA={ca_score:.3f}) → alineación pixel perfecta (IA)")
                else:
                    reasons.append(f"Óptica limpia sin aberración (lente alta resolución deportiva)")
            elif ca_score >= cfg.ca_ai_max:
                score -= 10.0

        # ── L: ENTROPÍA LOCAL DE TEXTURA (nueva V9) ────────────────
        lte = features.get("local_texture_entropy")
        if lte is not None:
            max_sc += cfg.local_entropy_weight
            if lte < cfg.local_entropy_ai_max:
                p = min(cfg.local_entropy_weight, (cfg.local_entropy_ai_max - lte) * 15.0)
                score += p
                reasons.append(f"Textura microscópica artificial (lte={lte:.2f})")
            elif lte > 5.5:
                score -= 12.0

        # ── M: DOMINANCIA DE CANAL VERDE (nueva V9) ────────────────
        gd = features.get("green_channel_dominance")
        if gd is not None:
            if gd > 1.3:
                score -= 15.0
                reasons.append(f"Dominancia canal verde (gd={gd:.2f}) → sensor Bayer real")
            elif gd < 1.05 and not has_real_cam and (sat_mean is not None and sat_mean < 180):
                score += 10.0
                max_sc += 10.0
                reasons.append(f"Sin dominancia canal verde (gd={gd:.2f}) → sin sensor Bayer")

        # ── N: SIMETRÍA BILATERAL (nueva V10) ──────────────────────
        bsym = features.get("bilateral_symmetry")
        if bsym is not None:
            max_sc += cfg.bilateral_symmetry_weight
            if bsym > cfg.bilateral_symmetry_ai_min and not has_real_cam:
                p = min(cfg.bilateral_symmetry_weight, (bsym - cfg.bilateral_symmetry_ai_min) * 130.0)
                score += p
                reasons.append(f"Simetría bilateral anormal (corr={bsym:.3f}) → IA")
            elif bsym < 0.4:
                score -= 5.0

        # ── O: CLUSTERING DE PALETA (nueva V10) ────────────────────
        cc = features.get("color_cluster_count")
        hs_feat = features.get("color_hue_spread")
        if cc is not None:
            max_sc += cfg.color_cluster_weight
            if cc <= cfg.color_cluster_ai_max and (hs_feat is not None and hs_feat < 50.0) and not has_real_cam:
                p = min(cfg.color_cluster_weight, (cfg.color_cluster_ai_max + 1 - cc) * 4.0)
                score += p
                reasons.append(f"Paleta de color estrecha ({cc} clusters, hue={hs_feat:.0f}°) → generación IA")

        # ── P: FFT MID-BAND RINGING (nueva V10) ────────────────────
        mbr = features.get("fft_midband_ring")
        if mbr is not None:
            max_sc += cfg.fft_midband_ring_weight
            if mbr > cfg.fft_midband_ring_min and not has_real_cam:
                p = min(cfg.fft_midband_ring_weight, (mbr - cfg.fft_midband_ring_min) * 20.0)
                score += p
                reasons.append(f"Ringing espectral mid-band (ratio={mbr:.2f}) → artefacto upsampler")

        # ── Q: VARIANZA DE BORDES POR CUADRANTE (nueva V10) ────────
        eqr = features.get("edge_quad_ratio")
        if eqr is not None:
            max_sc += cfg.edge_variance_weight
            if eqr < cfg.edge_variance_ratio_ai_max * 0.15 and not has_real_cam:
                p = min(cfg.edge_variance_weight, (0.20 - eqr) * 80.0)
                score += p
                reasons.append(f"Distribución uniforme de bordes (eqr={eqr:.3f}) → sin foco óptico")
            elif eqr > 0.5:
                score -= 8.0


        # ── I: FINGERPRINT ESPECTRAL ────────────────────────────────
        if fingerprint:
            top_sc  = fingerprint.get("top_score", 0.0)
            top_fam = fingerprint.get("top_family", "")
            if top_sc > 15.0:
                fp_contrib = min(60.0, top_sc * 1.5)
                score  += fp_contrib
                max_sc += 45.0
                reasons.extend(fingerprint.get("signals", []))
                if top_fam:
                    reasons.append(f"Fingerprint espectral: '{top_fam}' (score={top_sc:.1f})")
            elif top_sc > 5.0:
                score  += top_sc * 1.0
                max_sc += 20.0
                reasons.extend(fingerprint.get("signals", []))

        # ── J: GROK/AURORA — DETECTOR POR INTERSECCIÓN V10 ─────────
        grok_n = 0
        grok_ev: List[str] = []

        noise2     = features.get("noise_level")
        shadow_e   = features.get("shadow_entropy")
        grad_sm    = features.get("gradient_smoothness")
        dct_uni    = features.get("dct_block_uniformity")
        sat_mean2  = features.get("saturation_mean")
        sat_var2   = features.get("saturation_var")

        if noise2 is not None and noise2 < cfg.grok_noise_max:
            grok_n += 1
            grok_ev.append(f"ruido casi cero (n={noise2:.2f})")
        if shadow_e is not None and shadow_e < cfg.grok_shadow_entropy_max:
            grok_n += 1
            grok_ev.append(f"sombras sintéticas (E={shadow_e:.2f})")
        if grad_sm is not None and grad_sm < cfg.grok_gradient_smoothness_max:
            grok_n += 1
            grok_ev.append(f"gradiente sintético (var={grad_sm:.2f})")
        if dct_uni is not None and dct_uni < cfg.grok_dct_uniformity_max:
            grok_n += 1
            grok_ev.append(f"DCT uniforme (std={dct_uni:.2f})")
        if metadata:
            if not has_real_cam and not metadata.get("has_any_exif", False):
                grok_n += 1
                grok_ev.append("sin EXIF fotográfico")
        if sat_mean2 is not None and sat_var2 is not None:
            if sat_mean2 > 20.0 and sat_var2 < cfg.grok_sat_max:
                grok_n += 1
                grok_ev.append(f"color uniforme (sat_mean={sat_mean2:.0f}, sat_var={sat_var2:.0f})")
        if lte is not None and lte < 3.8:
            grok_n += 1
            grok_ev.append(f"entropía microescala extrema (lte={lte:.2f})")
        if cc is not None and cc <= 5:
            grok_n += 1
            grok_ev.append(f"paleta estrecha ({cc} clusters)")
        if mbr is not None and mbr > 1.4:
            grok_n += 1
            grok_ev.append(f"mid-band ringing (ratio={mbr:.2f})")
        if bsym is not None and bsym > 0.82:
            grok_n += 1
            grok_ev.append(f"simetría bilateral alta (corr={bsym:.3f})")
        if eqr is not None and eqr < 0.12:
            grok_n += 1
            grok_ev.append(f"bordes uniformes (eqr={eqr:.3f})")

        if grok_n >= cfg.grok_min_conditions:
            is_pro_sharp = (
                features.get("laplacian_var", 0.0) > 800.0 and
                (features.get("green_channel_dominance", 1.0) >= 1.15 or
                 features.get("block_consistency_std", 0.0) > 800.0 or
                 features.get("chromatic_aberration", 0.0) >= 0.35)
            )
            # V10.3: Evitar atenuación si hay convergencia extrema
            if is_pro_sharp and (lte is not None and lte < 1.5):
                is_pro_sharp = False

            if is_pro_sharp:
                grok_score_val = min(15.0, grok_n * 2.0)
                score  += grok_score_val
                max_sc += 15.0
                reasons.append(
                    f"Atenuación Grok — {grok_n}/11 señales derivadas de edición natural/estudio fotográfico: "
                    f"{', '.join(grok_ev[:3])}"
                )
            else:
                grok_score_val = min(40.0, grok_n * 6.0)
                score  += grok_score_val
                max_sc += 40.0
                reasons.append(
                    f"Señales tipo Grok — {grok_n}/11: "
                    f"{', '.join(grok_ev[:3])}"
                )
        else:
            max_sc += 10.0

        # ── IMPACTO NEURONAL ────────────────────────────────────────
        # V10.2: Peso normal para TODOS los casos. La detección Grok
        # la hace un clasificador independiente (GrokClassifier).
        nn_weight = 1.2 if (nn_prob > 90 or nn_prob < 10) else 0.8
        score  += (nn_prob - 50.0) * nn_weight
        max_sc += 50.0 * nn_weight

        # ── Z: ESCUDOS ÓPTICOS (POST-PROCESO V10.3) ────────────────
        ai_sig_count_total = count_ai_sigs(reasons)
        is_mass_convergence = ai_sig_count_total >= 5
        
        # B: LAPLACIANO (Protección Bokeh)
        lap_var = features.get("laplacian_var")
        block_std = features.get("block_consistency_std")
        if lap_var is not None:
            max_sc += cfg.laplacian_weight
            if lap_var > cfg.laplacian_professional_min and not is_mass_convergence:
                gd = features.get("green_channel_dominance", 1.0)
                if gd >= 1.05:
                    score -= 15.0 # Escudo moderado para fotos reales nítidas
                    reasons.append("Textura compatible con sensor fotográfico (Protección V10.3)")
                elif gd < 1.04:
                    score += 10.0 # Sospecha de nitidez artificial (sin Bayer)
                    reasons.append("Nitidez artificialmente perfecta (sin sensor Bayer)")
            elif lap_var < cfg.laplacian_ai_max:
                if block_std is not None and block_std > 500.0 and not is_mass_convergence:
                    score -= 25.0
                    reasons.append("Dinámica focal fotográfica: profundidad de campo óptica (Bokeh)")
                else:
                    p = min(cfg.laplacian_weight, (cfg.laplacian_ai_max - lap_var) * 0.5)
                    score += p
                    reasons.append(f"Suavizado de bordes artificial (LapVar={lap_var:.0f})")

        # K: ABERRACIÓN CROMÁTICA (Freno forense)
        if ca_score is not None:
            if ca_score > 0.35 and not is_mass_convergence:
                score -= 35.0
                reasons.append(f"Freno forense: Aberración cromática detectada ({ca_score:.2f})")

        # Bonus final por convergencia masiva
        if ai_sig_count_total >= 5:
            score  += 15.0

        # ── CÁLCULO FINAL ───────────────────────────────────────────
        if max_sc <= 0:
            max_sc = 1.0

        final_prob = float(np.clip(
            (score / max_sc) * 100.0 + cfg.prior_bias, 0.0, 100.0
        ))

        # Cortocircuitos deterministas por metadata
        if metadata and metadata.get("confirmed_ai") and final_prob < 70.0:
            final_prob = max(final_prob, 88.0)
        
        # V10.3: Los escudos se anulan si la convergencia de señales IA es indiscutible
        is_mass_convergence = count_ai_sigs(reasons) >= 5 or (fingerprint and fingerprint.get("top_score", 0) > 70.0)
        
        if metadata and metadata.get("real_camera") and not is_mass_convergence:
            if final_prob > 30.0:
                final_prob = min(final_prob, 20.0)

            # V10.1: ESCUDO EXIF REFORZADO
            exif_fields = metadata.get("real_camera_field_count", 0)
            if exif_fields >= 4 and final_prob > 15.0:
                reasons.append(
                    f"🛡 Escudo EXIF: {exif_fields} campos de cámara real → techo 15%"
                )
                final_prob = min(final_prob, 15.0)
            elif exif_fields >= 2 and final_prob > 22.0:
                reasons.append(
                    f"🛡 Escudo EXIF: {exif_fields} campos de cámara real → techo 22%"
                )
                final_prob = min(final_prob, 22.0)
            elif final_prob > 28.0:
                reasons.append(
                    f"🛡 Escudo EXIF: cámara real detectada → techo 28%"
                )
                final_prob = min(final_prob, 28.0)

        # V10.1: Floor Grok ELIMINADO — era demasiado burdo y causaba
        # falsos positivos. La detección Grok ahora depende del scoring
        # acumulado normal, sin pisos artificiales.

        # Vveredict logic V10.3
        # Mejorar proactividad en convergencias masivas (proporcional, no binario)
        ai_sig_count_eval = count_ai_sigs(reasons)
        if ai_sig_count_eval >= 7 and final_prob < 60.0:
            final_prob = min(59.9, final_prob * 1.1) # Solo subimos hasta el borde si no es masiva
        elif ai_sig_count_eval >= 9 and final_prob < 75.0:
            final_prob = max(final_prob, 70.0) # Boost para evidencias masivas indiscutibles

        if final_prob >= 60.0:
            verdict = "IA"
            conf    = min(99.0, final_prob + 10.0 if ai_sig_count_eval >= 3 else final_prob)
        else:
            verdict = "REAL"
            conf    = min(99.0, 100.0 - final_prob)

        if not reasons:
            reasons = ["Análisis fotográfico estándar y consistente."]

        return final_prob, verdict, f"{conf:.1f}%", reasons


# ═══════════════════════════════════════════════════════════════
# MEJORA 5: REGISTRO DE MODELOS CON FALLBACK Y VERSIONING
# ═══════════════════════════════════════════════════════════════
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "sdxl_detector": {
        "hf_path":       "Organika/sdxl-detector",
        "version":       "main",
        "fallback_local": None,   # ruta local si existe caché corrupto
        "required":      True,    # si falla y required=True → log crítico
        "weight":        1.0,     # peso relativo en el ensamble neuronal
    },
    "ai_human": {
        "hf_path":       "umm-maybe/AI-image-detector",
        "version":       "main",
        "fallback_local": None,
        "required":      False,
        "weight":        0.8,
    },
}


# ═══════════════════════════════════════════════════════════════
# MEJORA 3: MAPEO DE RAZONES TÉCNICAS → LENGUAJE HUMANO
# ═══════════════════════════════════════════════════════════════
_REASON_TRANSLATIONS: List[Tuple[str, str]] = [
    # (fragmento técnico a buscar,  texto en lenguaje humano)
    ("sombras sintéticamente",       "Las sombras son demasiado perfectas para una cámara real"),
    ("sin aberración cromática",     "No hay distorsión óptica natural de lentes (aberración cromática)"),
    ("textura microscópica",         "La textura a nivel de píxel es artificialmente uniforme"),
    ("sin dominancia canal verde",   "Ausencia de la huella del sensor Bayer de cámaras reales"),
    ("carencia anómala",             "Nivel de ruido anormalmente bajo para una foto real"),
    ("sin exif fotográfico",         "No contiene datos de cámara fotográfica (EXIF)"),
    ("sensor bayer real",            "Huella de sensor físico de cámara real confirmada"),
    ("escudo exif",                  "Datos EXIF de cámara real limitan la probabilidad de IA"),
    ("paleta de color",              "Paleta de color estrecha, típica de generadores IA"),
    ("simetría bilateral anormal",   "Simetría izquierda-derecha antinatural, característica de IA"),
    ("ringing espectral",            "Artefactos espectrales del upsampler de IA detectados"),
    ("patrón de rejilla",            "Patrón de rejilla VAE/DCT propio de Stable Diffusion / Flux"),
    ("halo en bordes",               "Halos artificiales en bordes, característicos de Midjourney"),
    ("ruido espacialmente uniforme", "Ruido añadido artificialmente, patrón de DALL-E"),
    ("superficie sintéticamente",    "Superficie demasiado pulida para ser fotografía real"),
    ("croma ultra-estrecha",         "Rango de color inusualmente limitado"),
    ("suavizado de bordes",          "Bordes artificialmente suavizados"),
    ("reconstrucción espectral",     "Patrón espectral de reconstrucción sintética"),
    ("inpainting",                   "Posible manipulación o relleno regional con IA (inpainting)"),
    ("generador:",                   "Generador identificado"),
    ("firma binaria",                "Firma binaria del generador IA encontrada en el archivo"),
    ("firma ia",                     "Firma del generador IA encontrada en los metadatos"),
    ("resolución nativa",            "Resolución exacta de generador IA conocido"),
    ("dinámica focal",               "Profundidad de campo óptica real (bokeh) detectada"),
    ("compatible con sensor",        "Textura compatible con sensor fotográfico profesional"),
    ("dominancia canal verde",       "Dominancia del canal verde confirma sensor Bayer real"),
    ("señales tipo grok",            "Múltiples señales convergentes de Grok/Aurora"),
]


def _build_evidence_block(
    prob: float,
    verdict: str,
    reasons: List[str],
    metadata: Dict[str, Any],
    grok_signals: List[str],
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    MEJORA 3: Construye el bloque 'evidence' orientado al usuario final.
    Traduce señales técnicas a lenguaje humano y asigna nivel semántico.
    """
    # Nivel de evidencia basado en probabilidad
    if prob >= 75:
        level = "high"
    elif prob >= 40:
        level = "medium"
    elif prob >= 20:
        level = "low"
    else:
        level = "minimal"

    # Traducir razones técnicas a lenguaje humano
    human_reasons: List[str] = []
    for raw in reasons + grok_signals:
        raw_lower = raw.lower()
        translated = None
        for fragment, human_text in _REASON_TRANSLATIONS:
            if fragment in raw_lower:
                translated = human_text
                break
        if translated and translated not in human_reasons:
            human_reasons.append(translated)

    # Si no se tradujeron razones, usar mensaje genérico
    if not human_reasons:
        if verdict == "REAL":
            human_reasons = ["La imagen presenta características consistentes con fotografía real"]
        elif verdict == "IA":
            human_reasons = ["La imagen presenta múltiples características de imagen generada por IA"]
        else:
            human_reasons = ["Las señales forenses son ambiguas o insuficientes para un veredicto definitivo"]

    # Razón principal: la más relevante
    main_reason = human_reasons[0] if human_reasons else ""

    # Sugerencia de generador
    generator_hint = ""
    gen = metadata.get("generator", "")
    if gen:
        generator_hint = f"Generador identificado: {gen}"
    elif prob >= 70:
        # Inferir desde fingerprint signals
        for sig in reasons:
            if "grok" in sig.lower() or "aurora" in sig.lower():
                generator_hint = "Posiblemente Grok/Aurora o Flux"
                break
            elif "midjourney" in sig.lower() or "mj" in sig.lower():
                generator_hint = "Posiblemente Midjourney"
                break
            elif "stable diffusion" in sig.lower() or "sdxl" in sig.lower():
                generator_hint = "Posiblemente Stable Diffusion / SDXL"
                break
            elif "dall-e" in sig.lower() or "dalle" in sig.lower():
                generator_hint = "Posiblemente DALL-E"
                break

    evidence: Dict[str, Any] = {
        "level":              level,
        "main_reason":        main_reason,
        "generator_hint":     generator_hint,
        "confidence_factors": human_reasons[:5],  # máx 5 para la UI
        "verdict_label":      {
            "IA":       "Imagen Generada por IA",
            "REAL":     "Fotografía Real",
            "REAL": "Resultado Indeterminado",
            "ERROR":    "Error de Análisis",
        }.get(verdict, verdict),
    }

    if verbose:
        evidence["technical_details"] = {
            "raw_reasons": reasons,
            "grok_signals": grok_signals,
        }

    return evidence


# ═══════════════════════════════════════════════════════════════
# MEJORA 2: ENSAMBLE PROBABILÍSTICO PONDERADO
# ═══════════════════════════════════════════════════════════════
def _weighted_ensemble(
    meta_prob: float,
    grok_prob: float,
    grok_signals: List[str],
    metadata: Dict[str, Any],
    features: Dict[str, float],
) -> Tuple[float, str]:
    """
    Combina MetaClassifier y GrokClassifier mediante promedio ponderado
    con pesos dinámicos basados en contexto.

    Regla clave: cuando Grok tiene alta confianza SIN EXIF, no se le
    promedia a la baja — se garantiza un PISO mínimo proporcional.
    """
    has_real_cam   = bool(metadata.get("real_camera"))
    has_confirm_ai = bool(metadata.get("confirmed_ai"))
    has_any_exif   = bool(metadata.get("has_any_exif", False))
    is_social      = bool(metadata.get("is_social_media"))

    # Contar señales forenses reales (excluir mensajes de gate)
    forensic_signals = [
        s for s in grok_signals
        if not s.startswith(("Clasificador Grok", "🔍", "🔀"))
    ]
    n_forensic = len(forensic_signals)

    # ── Caso 1: Cámara real en EXIF → MetaClassifier domina totalmente ──
    if has_real_cam:
        exif_fields = metadata.get("real_camera_field_count", 0)
        meta_w = min(0.97, 0.85 + exif_fields * 0.03)
        grok_w = 1.0 - meta_w
        combined = round(meta_prob * meta_w + grok_prob * grok_w, 1)
        return combined, f"EXIF real ({exif_fields} campos) — Meta domina"

    # ── Caso 2: IA confirmada por metadata → Meta domina ────────────
    if has_confirm_ai:
        combined = round(meta_prob * 0.95 + grok_prob * 0.05, 1)
        return combined, "metadata IA confirmada — Meta domina"

    # ── Caso 3: Sin EXIF — Grok es el detector primario ────────────────
    # Con muchas señales forenses sin EXIF, Grok es más fiable que Meta.
    # Garantizamos un PISO: el resultado no puede caer por debajo de
    # grok_prob * factor_piso, evitando que Meta lo “promedy” a la baja.
    if not has_any_exif:
        if n_forensic >= 10:
            # Alta confianza: Grok decide con Meta
            meta_w, grok_w = 0.25, 0.75
            blend_note = f"sin EXIF — convergencia IA masiva ({n_forensic} señales)"
        elif n_forensic >= 8:
            meta_w, grok_w = 0.40, 0.60
            blend_note = f"sin EXIF — alta convergencia Grok ({n_forensic} señales)"
        elif n_forensic >= 6:
            meta_w, grok_w = 0.50, 0.50
            blend_note = f"sin EXIF — Grok moderado ({n_forensic} señales)"
        else:
            meta_w, grok_w = 0.70, 0.30
            blend_note = "sin EXIF — ensamble estándar"

        # ── MEJORA V10.3: Protección específica para sensores reales (Bayer) ──
        gd = features.get("green_channel_dominance", 1.0)
        if gd >= 1.05 and n_forensic < 10:
             meta_w, grok_w = 0.65, 0.35
             blend_note += " + protección Bayer"

        combined = round(meta_prob * meta_w + grok_prob * grok_w, 1)

        # PISO GARANTIZADO: Grok sin EXIF con alta confianza no puede ser
        # arrastrado por debajo del umbral de detección por un Meta bajo,
        # A MENOS que Meta tenga pruebas físicas orgánicas concluyentes (< 40%).
        if meta_prob >= 40.0:
            if grok_prob >= 60.0 and n_forensic >= 9:
                # Alta confianza Grok con múltiples señales (mínimo 9) → prevalece
                combined = max(combined, grok_prob)
                blend_note += " [Grok prevalece]"
            elif grok_prob >= 50.0 and n_forensic >= 7:
                # Confianza moderada con alta convergencia → piso del 85%
                floor = round(grok_prob * 0.85, 1)
                if combined < floor:
                    combined = floor
                    blend_note += " [piso Grok aplicado]"
            elif grok_prob >= 50.0:
                # Confianza moderada → piso del 90% del score Grok
                floor = round(grok_prob * 0.90, 1)
                if combined < floor:
                    combined = floor
                    blend_note += " [piso Grok aplicado]"

        # ── MEJORA V10.3: Protección final contra falsos positivos de redes sociales ──
        if is_social and not has_confirm_ai and combined > 55:
            # Si no hay pruebas deterministas (metadata o fingerprint extremo),
            # capamos la probabilidad para evitar el falso veredicto IA.
            limit = 40.0 # Justo por debajo del umbral INCIERTO de routes_analyze (41%)
            if combined > limit:
                # Solo bajamos si MetaProb no es extremadamente alta (>80)
                if meta_prob < 80.0:
                    combined = limit
                    blend_note += " [escudo social media reforzado]"

        return combined, blend_note

    # ── Caso 4: Tiene EXIF pero no es cámara conocida ───────────────
    # EXIF parcial/desconocido — pesos estándar, ligera ventaja a Meta
    combined = round(meta_prob * 0.65 + grok_prob * 0.35, 1)
    return combined, "EXIF parcial — ensamble estándar"


# ═══════════════════════════════════════════════════════════════
# MOTOR TALOS V10.2
# ═══════════════════════════════════════════════════════════════
class TalosEngineV9:

    def __init__(self, config: Optional[ThresholdConfig] = None):
        self.device      = 0 if torch.cuda.is_available() else -1
        self.torch_dtype = torch.float16 if self.device == 0 else torch.float32
        self._model_caches: Dict[str, _LRUCache] = {}

        cfg = config or ThresholdConfig()
        self._threshold_cfg = cfg
        self.extractor         = UniversalFeatureExtractor()
        self.meta_classifier   = MetaClassifier(cfg)
        self.metadata_analyzer = MetadataAnalyzer()
        self.ela_analyzer      = ELAAnalyzer(quality=cfg.ela_quality)
        self.fingerprinter     = AIGeneratorFingerprinter()

        self._hybrid_model: Optional[Any] = None
        self._hybrid_processor: Optional[Any] = None
        self._hybrid_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._hybrid_lock = Lock()
        self._hybrid_load_error: Optional[str] = None
        self._hybrid_calibration: Dict[str, Any] = {}

        self._load_models()
        self._load_hybrid_calibration()
        self._config_hash = f"V10.2-{hashlib.md5(str(cfg).encode()).hexdigest()[:8]}"

    def _load_hybrid_calibration(self):
        """Carga el umbral optimizado desde el archivo de calibración."""
        calib_path = _HYBRID_CKPT.with_name("ai_image_detector_quick_calibration.json")
        if calib_path.exists():
            try:
                self._hybrid_calibration = json.loads(calib_path.read_text(encoding="utf-8"))
                thr = self._hybrid_calibration.get("threshold", 0.5)
                logger.info("Calibración híbrida cargada: umbral=%s", thr)
            except Exception as e:
                logger.warning("Fallo al cargar calibración: %s", e)

    def _load_models(self):
        """MEJORA 5: Carga desde MODEL_REGISTRY con versioning y fallback."""
        self._active_detectors: Dict[str, Any] = {}
        for name, spec in MODEL_REGISTRY.items():
            loaded = False
            # Intento 1: cargar desde HuggingFace (usa caché local automáticamente)
            try:
                mod = pipeline(
                    "image-classification",
                    model=spec["hf_path"],
                    revision=spec.get("version", "main"),
                    device=self.device,
                    torch_dtype=self.torch_dtype,
                )
                self._active_detectors[name] = mod
                self._model_caches[name] = _LRUCache(200)
                logger.info(f"Modelo '{name}' cargado desde '{spec['hf_path']}' (v={spec.get('version','main')})")
                loaded = True
            except Exception as e:
                logger.warning(f"Modelo '{name}' no cargó desde HF (Red): {e}. Intentando carga local...")
                # Intento 1b: Forzar modo offline si la red falló (usar lo que haya en cache)
                try:
                    mod = pipeline(
                        "image-classification",
                        model=spec["hf_path"],
                        revision=spec.get("version", "main"),
                        device=self.device,
                        torch_dtype=self.torch_dtype,
                        local_files_only=True
                    )
                    self._active_detectors[name] = mod
                    self._model_caches[name] = _LRUCache(200)
                    logger.info(f"Modelo '{name}' recuperado exitosamente desde caché local.")
                    loaded = True
                except Exception as e_off:
                    logger.warning(f"Carga local de '{name}' también falló: {e_off}")

            # Intento 2: fallback local (si está configurado)
            if not loaded and spec.get("fallback_local"):
                try:
                    mod = pipeline(
                        "image-classification",
                        model=spec["fallback_local"],
                        device=self.device,
                        torch_dtype=self.torch_dtype,
                    )
                    self._active_detectors[name] = mod
                    self._model_caches[name] = _LRUCache(200)
                    logger.info(f"Modelo '{name}' cargado desde fallback local: {spec['fallback_local']}")
                    loaded = True
                except Exception as e2:
                    logger.warning(f"Fallback local para '{name}' también falló: {e2}")

            if not loaded:
                level = "critical" if spec.get("required") else "warning"
                getattr(logger, level)(f"Modelo '{name}' no disponible (required={spec.get('required', False)})")

    def _score_neural(self, img_hash: str, img_pil: Image.Image) -> float:
        """MEJORA 5: Pondera scores neurales según el weight del MODEL_REGISTRY."""
        weighted_scores: List[float] = []
        weights: List[float] = []
        for name, model in self._active_detectors.items():
            cache = self._model_caches[name]
            score = cache.get(img_hash)
            if score is None:
                try:
                    res    = model(img_pil)
                    res    = res if isinstance(res, list) else [res]
                    parsed = {str(r["label"]).lower(): float(r["score"]) for r in res}
                    score  = 50.0
                    for k, sc in parsed.items():
                        if any(x in k for x in ["fake", "ai", "synthetic"]):
                            score = sc * 100.0; break
                        if any(x in k for x in ["real", "human", "photo"]):
                            score = (1.0 - sc) * 100.0; break
                    cache.set(img_hash, score)
                except Exception as e:
                    logger.warning(f"[{name}] Inferencia fallida: {e}")
                    continue
            w = MODEL_REGISTRY.get(name, {}).get("weight", 1.0)
            weighted_scores.append(score * w)
            weights.append(w)
        if not weighted_scores:
            return 50.0
        return float(sum(weighted_scores) / sum(weights))

    def _hybrid_env_enabled(self) -> bool:
        return os.getenv("TALOS_HYBRID_IMAGE", "1").strip().lower() not in ("0", "false", "no")

    def _ensure_hybrid(self) -> None:
        if not self._threshold_cfg.enable_hybrid_image_branch or not self._hybrid_env_enabled():
            return
        if self._hybrid_model is not None:
            return
        if self._hybrid_load_error is not None:
            return
        with self._hybrid_lock:
            if self._hybrid_model is not None or self._hybrid_load_error is not None:
                return
            if not _HYBRID_CKPT.exists():
                self._hybrid_load_error = "checkpoint_missing"
                logger.warning("Rama híbrida desactivada: no existe %s", _HYBRID_CKPT)
                return
            try:
                from models.modules.ai_image_detector import AIImageDetector, get_processor

                model = AIImageDetector().to(self._hybrid_device)
                model.eval()
                with torch.no_grad():
                    _ = model(torch.rand(2, 3, 224, 224, device=self._hybrid_device))
                try:
                    state = torch.load(_HYBRID_CKPT, map_location=self._hybrid_device, weights_only=False)
                except TypeError:
                    state = torch.load(_HYBRID_CKPT, map_location=self._hybrid_device)
                model.load_state_dict(state["model_state_dict"], strict=False)
                
                # Aplicar umbral calibrado si existe
                calib_thr = self._hybrid_calibration.get("threshold")
                if calib_thr is not None:
                    model.set_threshold(float(calib_thr))
                
                self._hybrid_model = model
                self._hybrid_processor = get_processor()
                logger.info("Rama híbrida imagen activa: %s", _HYBRID_CKPT)
            except Exception as exc:
                self._hybrid_load_error = str(exc)
                logger.warning("Rama híbrida no pudo cargarse: %s", exc)

    def _score_hybrid_branch(self, img_pil: Image.Image) -> Optional[Dict[str, Any]]:
        """Devuelve diccionario con prob_ai e is_ai, o None."""
        if not self._threshold_cfg.enable_hybrid_image_branch or not self._hybrid_env_enabled():
            return None
        self._ensure_hybrid()
        if self._hybrid_model is None or self._hybrid_processor is None:
            return None
        try:
            with torch.no_grad():
                pv = self._hybrid_processor(images=img_pil, return_tensors="pt")["pixel_values"].to(
                    self._hybrid_device
                )
                # predict_proba ahora devuelve Dict[str, any]
                out = self._hybrid_model.predict_proba(pv)
                return {
                    "prob_ai": float(out["prob_ai"].squeeze().item()),
                    "is_ai": bool(out["is_ai"].squeeze().item() > 0.5)
                }
        except Exception as exc:
            logger.warning("_score_hybrid_branch falló: %s", exc)
            return None

    @staticmethod
    def _blend_neural_with_hybrid(nn_legacy: float, hybrid_res: Optional[Dict[str, Any]], weight: float) -> Tuple[float, Optional[float]]:
        if hybrid_res is None:
            return nn_legacy, None
        
        hybrid_prob = hybrid_res["prob_ai"]
        w = max(0.0, min(1.0, float(weight)))
        hybrid_pct = hybrid_prob * 100.0
        
        # Si la neurona híbrida está 100% segura (calibrada), aumentamos su peso en el blend
        if hybrid_res.get("is_ai"):
            w = min(0.90, w * 1.5)
            
        blended = (1.0 - w) * nn_legacy + w * hybrid_pct
        return float(blended), float(hybrid_pct)

    @staticmethod
    def _validate_input(data: bytes) -> None:
        if len(data) > _MAX_IMAGE_BYTES:
            raise ValueError(
                f"Imagen demasiado grande: {len(data) / 1_000_000:.1f} MB "
                f"(máximo {_MAX_IMAGE_BYTES // 1_000_000} MB)"
            )
        try:
            with Image.open(io.BytesIO(data)) as probe:
                pw, ph = probe.size
                if pw * ph > _MAX_IMAGE_PIXELS:
                    raise ValueError(
                        f"Imagen demasiado grande: {pw}×{ph} px "
                        f"(máximo {_MAX_IMAGE_PIXELS // 1_000_000} Mpx)"
                    )
        except Exception as e:
            raise ValueError(f"No se pudo leer la imagen: {e}") from e

    def process_image(self, data: bytes, verbose: bool = False, is_social_media: bool = False) -> Dict[str, Any]:
        try:
            self._validate_input(data)

            # RAW load for metadata preservation
            img_pil_raw = Image.open(io.BytesIO(data))
            
            # Convert to RGB and arrays for analysis
            img_pil  = img_pil_raw.convert("RGB")
            img_cv   = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            gray     = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            img_hash = hashlib.md5(data).hexdigest()

            # Early exit: imagen sin textura evaluable
            if np.std(gray) < 2.0:
                res = {
                    "status": "success", "tipo": "imagen",
                    "probabilidad": 50, "probability": 50, "percentage": 50,
                    "prob": 50, "score": 50,
                    "verdict": "REAL", "confidence": "100.0%",
                    "nota": "Imagen descartada: lienzo sólido o sin textura.",
                    "reasons": ["La imagen carece de textura evaluable."],
                    "evidence": {
                        "level": "minimal",
                        "main_reason": "La imagen no contiene textura suficiente para analizar",
                        "generator_hint": "",
                        "confidence_factors": ["Sin textura evaluable"],
                        "verdict_label": "Resultado Indeterminado",
                    },
                    "semantico": {
                        "veredicto": "REAL", "confianza": "100.0%",
                        "explicacion": "Análisis abortado por falta de entropía visual.",
                        "score_ajuste": 0.0,
                    },
                    "detalles": {},
                }
                return {**res, "data": res, "result": res, "response": res}

            # Análisis paralelo (neural legacy + rama híbrida opcional + metadata + ELA + HIVE)
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
                f_neural = pool.submit(self._score_neural, img_hash, img_pil)
                f_hybrid = pool.submit(self._score_hybrid_branch, img_pil)
                f_metadata = pool.submit(self.metadata_analyzer.analyze, data, img_pil_raw)
                f_ela = pool.submit(self.ela_analyzer.analyze, data, img_pil)
                
                # [V10.3 Upgrade] Integración Hive para Imágenes
                def run_hive():
                    if not self._threshold_cfg.enable_hive:
                        return {"available": False}
                    try:
                        hive = HiveAnalyzer()
                        return hive.analyze_image_bytes(data)
                    except Exception as e:
                        logger.warning(f"Hive Image Falló: {e}")
                        return {"available": False}
                
                f_hive = pool.submit(run_hive)

                nn_prob_legacy = f_neural.result()
                hybrid_res = f_hybrid.result() # Ahora es un dict
                metadata = f_metadata.result()
                if is_social_media:
                    metadata["is_social_media"] = True
                    if "Origen detectado: URL de red social" not in metadata["signals"]:
                        metadata["signals"].append("Origen detectado: URL de red social")
                ela = f_ela.result()
                hive_res = f_hive.result()

            nn_prob, hybrid_pct = self._blend_neural_with_hybrid(
                nn_prob_legacy,
                hybrid_res,
                self._threshold_cfg.hybrid_blend_weight,
            )

            features    = self.extractor.extract(gray, img_cv)
            fingerprint = self.fingerprinter.analyze(gray, img_cv, features)

            prob, verdict, conf_str, reasons = self.meta_classifier.evaluate(
                features, nn_prob, metadata, ela, fingerprint
            )
            meta_prob_raw = prob   # guardar antes de que el ensamble lo sobreescriba

            # ── [V10.3 Upgrade] Refuerzo Hive para Imágenes ──────────
            hive_res = f_hive.result()
            hive_prob = 0.0
            if hive_res.get("available"):
                hive_prob = hive_res.get("suspicion", 0.0) * 100
                hive_suspect = hive_res.get("top_suspect", "unknown")
                
                # REPORTE EN CONSOLA (DEBUG)
                print(f"   >>> [HIVE SCAN] Sospecha: {hive_prob:.1f}% | Generador: {hive_suspect}")
                
                if hive_prob > 60:
                    # Nudge ultra-agresivo: Hive domina casi totalmente si detecta IA clara
                    prob = max(prob, hive_prob * 0.95 + prob * 0.05)
                    reasons.append(f"Validado por The Hive (SOTA): Alta sospecha de {hive_suspect.upper()} ({hive_prob:.1f}%)")
                    metadata["generator"] = metadata.get("generator") or f"Detectado por Hive ({hive_suspect})"
                elif hive_prob > 35: 
                    prob = max(prob, (prob * 0.4 + hive_prob * 0.6))
                    reasons.append(f"Señal prioritaria Hive: Patrón sospechoso de {hive_suspect.upper()} ({hive_prob:.1f}%)")
                elif hive_prob < 15 and hive_res.get("available"):
                    # ── HIVE VETO [V10.4] ────────────────────────────────────
                    # Si Hive dice REAL con alta confianza, desconfiamos de los falsos
                    # positivos de nuestros motores internos (especialmente Grok).
                    if prob > 40:
                        # Si es social media o no hay EXIF, el veto es casi total
                        if metadata.get("is_social_media") or not metadata.get("has_exif"):
                            prob = min(prob, 40.0) # Forzamos a zona REAL
                            reasons.append(f"Veto Orgánico Hive: Confirmación SOTA de autenticidad ({hive_prob:.1f}%)")
                        else:
                            # En otros casos, solo promediamos a la baja
                            prob = (prob + hive_prob) / 2
                            reasons.append(f"Atenuación Hive: Baja sospecha en motor externo ({hive_prob:.1f}%)")

            # ── MEJORA 2: Clasificador Grok + Ensamble Ponderado ──────────
            grok_prob, grok_signals = GrokClassifier.classify(
                features, metadata, img_pil.size
            )

            # Combinar Meta y Grok con promedio ponderado dinámico
            prob, blend_note = _weighted_ensemble(
                meta_prob=prob,
                grok_prob=grok_prob,
                grok_signals=grok_signals,
                metadata=metadata,
                features=features,
            )
            reasons.append(
                f"🔀 Ensamble [{blend_note}]: Meta={int(meta_prob_raw)}% · Grok={int(grok_prob)}% → {int(prob)}%"
            )
            reasons.extend(grok_signals)

            if hybrid_res and self._threshold_cfg.hybrid_final_nudge_weight > 0:
                h_w = max(0.0, min(1.0, float(self._threshold_cfg.hybrid_final_nudge_weight)))
                h_pct = hybrid_res["prob_ai"] * 100.0
                prob_before_hybrid_nudge = prob
                
                # Si la neurona confirma IA usando el umbral de calibración forense
                if hybrid_res.get("is_ai"):
                    # Nudge agresivo
                    prob = (1.0 - h_w) * prob + h_w * h_pct
                    reasons.append(
                        f"Confirmado por firma neural forense (Hybrid-Core): "
                        f"{int(prob_before_hybrid_nudge)}% → {int(prob)}%"
                    )
                elif hybrid_res["prob_ai"] <= float(self._threshold_cfg.hybrid_nudge_low_threshold):
                    h_w_soft = h_w * 0.55
                    prob = (1.0 - h_w_soft) * prob + h_w_soft * h_pct
                    reasons.append(
                        f"Híbrido tardío REAL (w={h_w_soft:.2f}, p={hybrid_res['prob_ai']:.2f}): "
                        f"{int(prob_before_hybrid_nudge)}% → {int(prob)}%"
                    )

            # ── HIVE MASTER VETO [V10.4] ──────────────────────────────────────
            # Si The Hive (SOTA) dice REAL con < 5% de sospecha, forzamos REAL 
            # a menos que haya una firma de metadatos o generador confirmada.
            if hive_res.get("available") and hive_prob < 5.0 and not metadata.get("confirmed_ai"):
                if prob > 40:
                    old_p = prob
                    prob = min(prob, 35.0)
                    reasons.append(f"🏆 Veredicto Maestro Hive SOTA: Falso positivo interno corregido ({int(old_p)}% -> {int(prob)}%)")

            # Recalcular veredicto con probabilidad del ensamble (Estándar Pulzo)
            cfg = self.meta_classifier.cfg
            if prob >= cfg.verdict_ai_threshold:
                verdict  = "IA"
                conf_str = f"{min(99.0, prob):.1f}%"
            else:
                verdict  = "REAL"
                conf_str = f"{min(99.0, 100.0 - prob):.1f}%"

            # ── MEJORA 3: Bloque de evidencia orientado al usuario ─────────
            evidence = _build_evidence_block(
                prob=prob,
                verdict=verdict,
                reasons=reasons,
                metadata=metadata,
                grok_signals=grok_signals,
                verbose=verbose,
            )

            ela_public = {k: v for k, v in ela.items() if k != "heatmap_b64"}
            if verbose:
                ela_public["heatmap_b64"] = ela.get("heatmap_b64")

            # ── MEJORA 4: Log de predicción para calibración futura ────────
            logger.info(
                f"PRED|hash={img_hash[:8]}|prob={prob:.1f}|verdict={verdict}|"
                f"neural={nn_prob:.1f}|neural_leg={nn_prob_legacy:.1f}|"
                f"hybrid={hybrid_res['prob_ai'] if hybrid_res else 'na'}|"
                f"meta={meta_prob_raw:.1f}|grok={grok_prob:.1f}|"
                f"real_cam={metadata.get('confirmed_ai', False)}|"
                f"meta_confirmed={metadata.get('real_camera', False)}"
            )

            res: Dict[str, Any] = {
                "status":       "success",
                "tipo":         "imagen",
                "probabilidad": int(prob),
                "probability":  int(prob),
                "percentage":   int(prob),
                "prob":         int(prob),
                "score":        int(prob),
                "verdict":      verdict,
                "confidence":   conf_str,
                "nota":         " | ".join(reasons),
                "reasons":      reasons,
                "module_scores": {
                    "Neural CNN": f"{int(nn_prob)}%",
                    "Neural ViT (Hybrid)": f"{int(hybrid_pct)}%" if hybrid_pct is not None else "N/A",
                    "Grok Heuristics": f"{int(grok_prob)}%",
                    "Meta Ensemble": f"{int(meta_prob_raw)}%",
                    "The Hive (SOTA)": f"{int(hive_prob)}%" if 'hive_prob' in locals() and hive_res.get('available') else "N/A",
                    "Fingerprint Spectral": f"{int(fingerprint.get('top_score', 0.0))}%" if fingerprint else "N/A",
                    "ELA Anomaly": f"{int(ela_public.get('score_ia', 0.0))}%" if ela_public else "N/A"
                },
                # ── MEJORA 3: Bloque para el usuario final ────────────────
                "evidence":     evidence,
                "semantico": {
                    "veredicto":    verdict,
                    "confianza":    conf_str,
                    "explicacion":  "Análisis Biométrico + ELA + Metadata + CA + Entropy + Symmetry + Palette V10.2",
                    "score_ajuste": round(prob - meta_prob_raw, 1),
                },
                "detalles": {
                    "neural":       round(nn_prob, 1),
                    "neural_legacy": round(nn_prob_legacy, 1),
                    "neural_hybrid_prob": round(hybrid_res["prob_ai"], 4) if hybrid_res else None,
                    "neural_hybrid_pct": round(hybrid_pct, 1) if hybrid_pct is not None else None,
                    "hybrid_blend_w": self._threshold_cfg.hybrid_blend_weight,
                    "hybrid_final_nudge_w": self._threshold_cfg.hybrid_final_nudge_weight,
                    "hybrid_nudge_high_thr": self._threshold_cfg.hybrid_nudge_high_threshold,
                    "hybrid_nudge_low_thr": self._threshold_cfg.hybrid_nudge_low_threshold,
                    "meta_prob":    round(meta_prob_raw, 1),
                    "grok_prob":    round(grok_prob, 1),
                    "blend_note":   blend_note,
                    "features":     {k: (round(v, 2) if v is not None else None) for k, v in features.items()},
                    "ela":          ela_public,
                    "metadata": {
                        "confirmed_ai":   metadata["confirmed_ai"],
                        "generator":      metadata["generator"],
                        "real_camera":    metadata["real_camera"],
                        "c2pa_present":   metadata["c2pa_present"],
                        "c2pa_ai_signed": metadata["c2pa_ai_signed"],
                        "signals":        metadata["signals"],
                        **({"raw_fields": metadata["raw_fields"]} if verbose else {}),
                    },
                    "fingerprint": {
                        "family_scores": fingerprint["family_scores"],
                        "top_family":    fingerprint["top_family"],
                        "signals":       fingerprint["signals"],
                        "fft_grid":      fingerprint["fft_grid_detected"],
                        "edge_halo":     fingerprint["edge_halo_detected"],
                        "noise_inject":  fingerprint["noise_injection_detected"],
                    },
                    "predicciones": {
                        "IA":     f"{int(prob)}%",
                        "Humano": f"{100 - int(prob)}%",
                    },
                },
            }
            return {**res, "data": res, "result": res, "response": res}

        except Exception as e:
            logger.error(f"process_image falló: {e}", exc_info=True)
            return self._error_result(str(e))

    def process_batch(
        self,
        data_list: List[bytes],
        max_workers: int = 4,
        progress_callback=None,
    ) -> List[Dict[str, Any]]:
        results: List[Optional[Dict[str, Any]]] = [None] * len(data_list)
        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(self.process_image, d): i for i, d in enumerate(data_list)}
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Batch item {idx} falló: {e}", exc_info=True)
                    results[idx] = self._error_result(str(e))
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(data_list))
        return results  # type: ignore[return-value]

    @staticmethod
    def _error_result(msg: str) -> Dict[str, Any]:
        err: Dict[str, Any] = {
            "status": "error", "tipo": "imagen",
            "probabilidad": 0, "probability": 0, "percentage": 0,
            "prob": 0, "score": 0,
            "verdict": "ERROR", "error": msg, "nota": msg,
            "reasons": [msg], "confidence": "0%",
            "semantico": {"veredicto": "ERROR", "confianza": "0%", "explicacion": msg, "score_ajuste": 0},
            "detalles": {},
        }
        return {**err, "data": err, "result": err, "response": err}


# ═══════════════════════════════════════════════════════════════
# SINGLETON CON DOUBLE-CHECKED LOCKING
# ═══════════════════════════════════════════════════════════════
_engine: Optional[TalosEngineV9] = None
_engine_lock = Lock()


def reset_image_engine() -> None:
    """Libera el singleton del motor de imagen (p. ej. para pruebas A/B con distinta config/env)."""
    global _engine
    with _engine_lock:
        _engine = None


def get_engine(config: Optional[ThresholdConfig] = None) -> TalosEngineV9:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = TalosEngineV9(config)
    return _engine


def analyze_image(image_bytes: bytes, verbose: bool = False, is_social_media: bool = False) -> Dict[str, Any]:
    """Helper wrapper compatible con el router V4.1."""
    return get_engine().process_image(image_bytes, verbose=verbose, is_social_media=is_social_media)
                                                   