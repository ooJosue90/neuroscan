"""
MÓDULO: Forensic Analyzer V4
Análisis forense de señal de imagen:
  - PRNU (Photo-Response Non-Uniformity) — firma de sensor
  - ELA (Error Level Analysis) — detección de edición/splice
  - Detección de upscaling IA (ESRGAN/GFPGAN artifacts)
  - Análisis de ruido de sensor vs ruido sintético
  - Artefactos de compresión JPEG/H.264
  - SRM (Steganalysis Rich Model) simplificado
  - Iluminación inconsistente entre frames
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import tempfile
import os
import concurrent.futures

class ForensicAnalyzer:
    """
    Análisis forense de bajo nivel de señal de imagen.
    Detecta manipulación sintética que es invisible visualmente.
    """

    def __init__(self):
        print(">>> [ForensicAnalyzer] Inicializado")

    # ------------------------------------------------------------------
    # PRNU — Photo-Response Non-Uniformity
    # ------------------------------------------------------------------
    def _extract_noise_residual(self, frame_gray: np.ndarray) -> np.ndarray:
        """
        Extrae el residual de ruido usando filtro de Wiener.
        El residual contiene la firma del sensor (PRNU).
        """
        try:
            from scipy.signal import wiener
            img_float = frame_gray.astype(np.float64) / 255.0
            # Al procesar videos 100% vectoriales/IA, pueden haber áreas de 0 varianza.
            # SciPy's Wiener arroja divide-by-zero. Añadimos epsilon de ruido térmico.
            img_float_safe = img_float + np.random.randn(*img_float.shape) * 1e-6
            denoised  = wiener(img_float_safe, mysize=3)
            residual  = img_float - denoised
            return residual
        except ImportError:
            # Fallback: Gaussian blur como denoiser
            img_float = frame_gray.astype(np.float64) / 255.0
            denoised  = cv2.GaussianBlur(img_float, (3, 3), 0)
            return img_float - denoised

    def _analyze_prnu(self, frames_gray: List[np.ndarray]) -> Dict:
        """
        Analiza correlación de residuales PRNU entre frames.
        
        Cámara real: residuales altamente correlacionados (misma firma de sensor).
        IA generativa: residuales sin correlación (ruido aleatorio sin firma).
        Face-swap: correlación inconsistente (dos fuentes de ruido).
        """
        if len(frames_gray) < 4:
            return {"prnu_correlation": 0.0, "suspicion": 0.4, "available": False}

        # Submuestrear para eficiencia
        sample_indices = np.linspace(0, len(frames_gray)-1, min(15, len(frames_gray)), dtype=int)
        residuals = [self._extract_noise_residual(frames_gray[i]) for i in sample_indices]

        # Correlación entre residuales consecutivos
        correlations = []
        for i in range(len(residuals) - 1):
            r1 = residuals[i].flatten()
            r2 = residuals[i+1].flatten()
            # Pearson correlation
            corr = np.corrcoef(r1, r2)[0, 1]
            if not np.isnan(corr):
                correlations.append(float(corr))

        # Correlación con primer frame (PRNU reference)
        ref_residual = residuals[0].flatten()
        ref_correlations = []
        for res in residuals[1:]:
            corr = np.corrcoef(ref_residual, res.flatten())[0, 1]
            if not np.isnan(corr):
                ref_correlations.append(float(corr))

        avg_consecutive = float(np.mean(correlations)) if correlations else 0.0
        avg_ref         = float(np.mean(ref_correlations)) if ref_correlations else 0.0

        # Varianza de correlaciones: face-swap tiene varianza alta (dos sensores)
        corr_std = float(np.std(correlations)) if len(correlations) > 2 else 0.0

        # Real camera: avg_consecutive ≈ 0.05-0.30 (sensor noise pattern)
        # AI generated: avg_consecutive ≈ 0.00-0.02 (random noise)
        # Face-swap: avg_consecutive ≈ 0.00-0.05 + alta varianza
        suspicion = 0.0
        if avg_consecutive < 0.015:
            suspicion += 0.40  # Sin firma de sensor (reducido por compresión, pero capaz de ver IA plana)
        elif avg_consecutive < 0.035:
            suspicion += 0.20
        if corr_std > 0.10:
            suspicion += 0.15  # Varianza alta = múltiples fuentes

        return {
            "prnu_consecutive_corr": round(avg_consecutive, 4),
            "prnu_reference_corr":   round(avg_ref, 4),
            "prnu_corr_std":         round(corr_std, 4),
            "frames_analyzed":       len(residuals),
            "suspicion":             round(min(1.0, suspicion), 3),
            "available":             True
        }

    # ------------------------------------------------------------------
    # ELA — Error Level Analysis
    # ------------------------------------------------------------------
    def _compute_ela(self, frame_bgr: np.ndarray, quality: int = 90) -> np.ndarray:
        """
        Recomprime el frame en memoria y calcula el mapa de error.
        Zonas editadas/pegadas tienen error diferente al resto.
        """
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, encoded_img = cv2.imencode('.jpg', frame_bgr, encode_param)

        if not success:
            return np.zeros_like(frame_bgr[:,:,0], dtype=float)

        recompressed = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)

        if recompressed is None:
            return np.zeros_like(frame_bgr[:,:,0], dtype=float)

        # Amplificar error para análisis
        ela = np.abs(frame_bgr.astype(float) - recompressed.astype(float))
        ela_amplified = np.clip(ela * 10, 0, 255).astype(np.uint8)
        return ela_amplified

    def _analyze_ela_splice_detection(self, frames_bgr: List[np.ndarray]) -> Dict:
        """
        Detecta regiones de face-swap usando ELA.
        Face-swap: región facial tiene error de recompresión diferente al fondo.
        Busca contornos ovales/circulares de alta discrepancia ELA.
        """
        splice_scores = []

        # Analizar subset de frames
        step = max(1, len(frames_bgr) // 8)
        for frame in frames_bgr[::step]:
            ela = self._compute_ela(frame)
            ela_gray = ela.mean(axis=2) if ela.ndim == 3 else ela

            # Percentil alto → zonas de mayor error
            high_ela = (ela_gray > np.percentile(ela_gray, 93)).astype(np.uint8)

            # Buscar contornos con forma plausible de región facial
            contours, _ = cv2.findContours(high_ela, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            frame_splice_score = 0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 300 or area > 0.25 * frame.shape[0] * frame.shape[1]:
                    continue
                hull      = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                if hull_area == 0:
                    continue
                solidity  = area / hull_area
                # Forma "de cara" → solidity entre 0.4 y 0.85
                if 0.4 < solidity < 0.85:
                    # Puntaje proporcional al área relativa
                    relative_area = area / (frame.shape[0] * frame.shape[1])
                    frame_splice_score += relative_area * 10
            splice_scores.append(min(1.0, frame_splice_score))

        if not splice_scores:
            return {"ela_splice_score": 0.0, "suspicion": 0.1}

        avg_splice = float(np.mean(splice_scores))
        suspicion  = min(1.0, avg_splice * 2.0)

        return {
            "ela_splice_score": round(avg_splice, 4),
            "ela_max_score":    round(float(np.max(splice_scores)), 4),
            "suspicion":        round(suspicion, 3)
        }

    # ------------------------------------------------------------------
    # Detección de Upscaling IA
    # ------------------------------------------------------------------
    def _detect_ai_upscaling(self, frames_gray: List[np.ndarray]) -> Dict:
        """
        ESRGAN/Real-ESRGAN/GFPGAN producen periodicidad en el espectro FFT.
        Detecta picos anómalos a frecuencias características de upscaling.
        """
        upscale_scores = []

        step = max(1, len(frames_gray) // 6)
        for gray in frames_gray[::step]:
            fft     = np.fft.fft2(gray.astype(float))
            shifted = np.fft.fftshift(fft)
            mag     = np.log1p(np.abs(shifted))

            h, w = mag.shape
            cy, cx = h // 2, w // 2

            # Energía en anillos de frecuencia específicos
            # ESRGAN 4x: picos en r ≈ H/4 (aliasing del downsampling)
            def ring_energy(r, thickness=3):
                mask = np.zeros_like(mag)
                cv2.circle(mask, (cx, cy), r, 1, thickness)
                return float(np.sum(mag * mask))

            def disk_energy(r):
                mask = np.zeros_like(mag)
                cv2.circle(mask, (cx, cy), r, 1, -1)
                return float(np.sum(mag * mask)) + 1e-8

            radii = [h//8, h//6, h//5, h//4, h//3]
            ratios = []
            for r in radii:
                re = ring_energy(r)
                de = disk_energy(r // 2)
                ratios.append(re / de)

            # Upscaling IA: ratio elevado en frecuencias medias-altas
            max_ratio = float(max(ratios))
            upscale_scores.append(max_ratio)

        if not upscale_scores:
            return {"upscale_fft_ratio": 0.0, "suspicion": 0.1}

        avg_ratio = float(np.mean(upscale_scores))
        # Umbral empírico: > 0.12 es sospechoso
        suspicion = min(1.0, max(0.0, (avg_ratio - 0.08) / 0.10))

        return {
            "upscale_fft_ratio": round(avg_ratio, 4),
            "upscale_max_ratio": round(float(np.max(upscale_scores)), 4),
            "suspicion":         round(suspicion, 3)
        }

    # ------------------------------------------------------------------
    # Detección de patrones de cuadrícula (GAN/Diffusion signatures)
    # ------------------------------------------------------------------
    def _analyze_frequency_grid(self, frames_gray: List[np.ndarray]) -> Dict:
        """
        Detecta artefactos de rejilla (checkerboard) en el espectro de frecuencia.
        GANs y modelos de Difusión dejan picos periódicos en el dominio FFT
        debido a las capas de upsampling (ConvTranspose2d o Nearest+Conv).
        """
        grid_scores = []
        
        step = max(1, len(frames_gray) // 6)
        for gray in frames_gray[::step]:
            h, w = gray.shape
            # Tomar un parche central cuadrado para FFT limpia
            s = min(h, w, 512)
            y, x = (h - s) // 2, (w - s) // 2
            roi = gray[y:y+s, x:x+s]
            
            f = np.fft.fft2(roi.astype(float))
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = np.log(np.abs(fshift) + 1e-8)
            
            # Normalizar espectro
            mag_min, mag_max = magnitude_spectrum.min(), magnitude_spectrum.max()
            mag_norm = (magnitude_spectrum - mag_min) / (mag_max - mag_min + 1e-8)
            
            # Buscar picos en las esquinas del espectro de alta frecuencia
            # (típicamente donde se manifiestan los patrones de de-convolución)
            cy, cx = s // 2, s // 2
            # Zonas de interés: picos simétricos lejos del centro
            peak_zones = [
                mag_norm[cy-100:cy-40, cx-100:cx-40],
                mag_norm[cy-100:cy-40, cx+40:cx+100],
                mag_norm[cy+40:cy+100, cx-100:cx-40],
                mag_norm[cy+40:cy+100, cx+40:cx+100]
            ]
            
            max_peaks = [float(np.max(z)) if z.size > 0 else 0.0 for z in peak_zones]
            grid_scores.append(float(np.mean(max_peaks)))

        if not grid_scores:
            return {"grid_peak_score": 0.0, "suspicion": 0.1}
            
        avg_grid = float(np.mean(grid_scores))
        # Los generadores modernos son más limpios, pero aún dejan rastro en > 0.45
        suspicion = min(1.0, max(0.0, (avg_grid - 0.40) / 0.20))
        
        return {
            "grid_peak_score": round(avg_grid, 4),
            "grid_max_score":  round(float(np.max(grid_scores)), 4),
            "suspicion":       round(suspicion, 3)
        }

    # ------------------------------------------------------------------
    # Análisis de ruido de sensor vs ruido sintético
    # ------------------------------------------------------------------
    def _analyze_noise_signature(self, frames_gray: List[np.ndarray]) -> Dict:
        """
        Caracteriza el ruido de alta frecuencia.
        
        Ruido de sensor real:
          - Distribución Gaussiana (Poisson a bajas frecuencias)
          - Varía entre canales RGB de forma correlacionada
          - Aumenta en zonas oscuras (shot noise)
        
        Ruido sintético IA:
          - Demasiado uniforme (std muy baja)
          - O ausente (std < 1.5 → super-smooth)
          - No correlaciona con luminancia local
        """
        noise_stds    = []
        noise_kurtosis = []
        noise_skews   = []
        dark_noise    = []
        bright_noise  = []

        step = max(1, len(frames_gray) // 12)
        for gray in frames_gray[::step]:
            img_f = gray.astype(np.float64)

            # Extraer ruido con HPF
            blurred = cv2.GaussianBlur(img_f, (3, 3), 0)
            noise   = img_f - blurred

            noise_std  = float(np.std(noise))
            noise_flat = noise.flatten()

            if noise_std < 0.01:
                noise_stds.append(noise_std)
                noise_kurtosis.append(0.0)
                noise_skews.append(0.0)
                continue

            # Estadísticas de la distribución de ruido
            mean_n = float(np.mean(noise_flat))
            std_n  = noise_std
            kurt   = float(np.mean(((noise_flat - mean_n) / (std_n + 1e-10))**4)) - 3.0
            skew   = float(np.mean(((noise_flat - mean_n) / (std_n + 1e-10))**3))

            noise_stds.append(noise_std)
            noise_kurtosis.append(kurt)
            noise_skews.append(skew)

            # Ruido en zonas oscuras vs brillantes
            dark_mask   = (gray < 64)
            bright_mask = (gray > 192)
            if np.sum(dark_mask) > 100:
                dark_noise.append(float(np.std(noise[dark_mask])))
            if np.sum(bright_mask) > 100:
                bright_noise.append(float(np.std(noise[bright_mask])))

        if not noise_stds:
            return {"noise_std": 0.0, "suspicion": 0.5}

        avg_std  = float(np.mean(noise_stds))
        avg_kurt = float(np.mean(noise_kurtosis))
        avg_skew = float(np.mean(noise_skews))
        dark_avg = float(np.mean(dark_noise)) if dark_noise else avg_std
        bright_avg = float(np.mean(bright_noise)) if bright_noise else avg_std

        # Shot noise real: dark_noise > bright_noise (inverso en IA)
        shot_noise_ratio = dark_avg / (bright_avg + 1e-8)

        suspicion = 0.0
        # Sin ruido (IA súper suave) — más relajado para evitar castigar compresión
        if avg_std < 0.5:
            suspicion += 0.25
        elif avg_std < 1.0:
            suspicion += 0.10
        # Kurtosis exactamente 0: ruido perfectamente gaussiano = artificial
        if abs(avg_kurt) < 0.1:
            suspicion += 0.20
        # Shot noise al revés: IA no modela correlación ruido-luminancia
        if shot_noise_ratio < 0.8:  # Debería ser > 1.0 en sensor real
            suspicion += 0.25

        return {
            "noise_std":         round(avg_std, 3),
            "noise_kurtosis":    round(avg_kurt, 3),
            "noise_skew":        round(avg_skew, 3),
            "shot_noise_ratio":  round(shot_noise_ratio, 3),
            "dark_noise":        round(dark_avg, 3),
            "bright_noise":      round(bright_avg, 3),
            "suspicion":         round(min(1.0, suspicion), 3)
        }

    # ------------------------------------------------------------------
    # Consistencia de iluminación entre frames
    # ------------------------------------------------------------------
    def _analyze_lighting_consistency(self, frames_bgr: List[np.ndarray]) -> Dict:
        """
        Detecta cambios bruscos de iluminación típicos de Pika/Runway:
        la iluminación del sujeto cambia sin razón física.
        También detecta inconsistencia de sombras (deepfake de foto).
        """
        if len(frames_bgr) < 4:
            return {"lighting_consistency": 1.0, "suspicion": 0.1}

        # Histograma de luminancia por frame (canal V de HSV)
        luminance_means = []
        luminance_stds  = []
        highlight_ratios = []

        for frame in frames_bgr:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            v_channel = hsv[:, :, 2].astype(float)
            luminance_means.append(float(np.mean(v_channel)))
            luminance_stds.append(float(np.std(v_channel)))
            # Ratio de píxeles muy brillantes (especular highlights)
            highlight_ratios.append(float(np.mean(v_channel > 230)))

        lm_array = np.array(luminance_means)

        # Cambios bruscos frame a frame
        lm_deltas    = np.abs(np.diff(lm_array))
        sudden_jumps = float(np.mean(lm_deltas > 15))  # >15 lum units = brusco
        max_jump     = float(np.max(lm_deltas)) if len(lm_deltas) > 0 else 0.0

        # Varianza de iluminación a largo plazo (drift gradual = natural, saltos = IA)
        lm_var = float(np.var(lm_array))

        # Inconsistencia de highlights (especulares que aparecen/desaparecen)
        hl_array     = np.array(highlight_ratios)
        hl_variance  = float(np.var(hl_array))

        suspicion = 0.0
        if sudden_jumps > 0.25:
            suspicion += 0.20  # >25% frames con saltos (más tolerante a cortes reales)
        if max_jump > 60:
            suspicion += 0.15
        if hl_variance > 0.002:
            suspicion += 0.10  # Highlights muy variables

        return {
            "luminance_mean":     round(float(np.mean(lm_array)), 2),
            "sudden_jump_ratio":  round(sudden_jumps, 4),
            "max_lum_jump":       round(max_jump, 2),
            "lm_variance":        round(lm_var, 2),
            "highlight_variance": round(hl_variance, 6),
            "suspicion":          round(min(1.0, suspicion), 3)
        }

    # ------------------------------------------------------------------
    # SRM simplificado (Steganalysis Rich Model)
    # ------------------------------------------------------------------
    def _analyze_srm_features(self, frames_gray: List[np.ndarray]) -> Dict:
        """
        Versión simplificada de SRM: analiza residuales de múltiples filtros de predicción.
        Los generadores IA tienen patrones estadísticos diferentes en residuales de predicción.
        """
        # Filtros de predicción SRM (subconjunto)
        kernels = [
            # Predicción horizontal
            np.array([[0, 0, 0], [1/2, 0, -1/2], [0, 0, 0]], dtype=np.float32),
            # Predicción vertical
            np.array([[0, 1/2, 0], [0, 0, 0], [0, -1/2, 0]], dtype=np.float32),
            # Predicción diagonal
            np.array([[1/4, 0, 0], [0, 0, 0], [0, 0, -1/4]], dtype=np.float32),
            # Laplaciano
            np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32) / 4,
        ]

        all_features = []
        step = max(1, len(frames_gray) // 6)

        for gray in frames_gray[::step]:
            img_f = gray.astype(np.float32) / 255.0
            frame_feats = []
            for k in kernels:
                res = cv2.filter2D(img_f, -1, k)
                # Truncar a [-T, T] (clipping típico de SRM)
                T = 3.0 / 255.0
                res_clipped = np.clip(res, -T, T)
                # Co-occurrence features simplificadas
                frame_feats.extend([
                    float(np.mean(np.abs(res_clipped))),
                    float(np.std(res_clipped)),
                    float(np.mean(res_clipped**2))
                ])
            all_features.append(frame_feats)

        if not all_features:
            return {"srm_anomaly": 0.0, "suspicion": 0.2}

        feat_array = np.array(all_features)
        feat_means = feat_array.mean(axis=0)
        feat_vars  = feat_array.var(axis=0)

        # Anomalía: alta varianza temporal en features SRM → cambios estadísticos anómalos
        srm_anomaly = float(np.mean(feat_vars))

        # Comparar con valores baseline típicos
        # IA generativa: anomalía baja (producción uniforme)
        # Videos reales con compresión: anomalía moderada (variación natural de escenas)
        suspicion = min(1.0, max(0.0, srm_anomaly * 1000 - 0.5))  # Umbral empírico

        return {
            "srm_anomaly":  round(srm_anomaly, 6),
            "srm_feat_std": round(float(np.mean(np.sqrt(feat_vars))), 6),
            "suspicion":    round(suspicion, 3)
        }

    # ------------------------------------------------------------------
    # Método principal
    # ------------------------------------------------------------------
    def analyze(self, frames_bgr: List[np.ndarray]) -> Dict:
        """
        Análisis forense completo.
        
        Args:
            frames_bgr: Lista de frames BGR
        
        Returns:
            Dict con todos los análisis forenses y suspicion global [0-1]
        """
        if not frames_bgr:
            return {"suspicion": 0.3, "available": False}

        frames_gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames_bgr]

        # Micro-paralelismo: Ejecutar los 6 análisis intensivos a la vez con hilos,
        # aprovechando que OpenCV y las rutinas C internas sueltan el GIL
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            fut_prnu    = executor.submit(self._analyze_prnu, frames_gray)
            fut_ela     = executor.submit(self._analyze_ela_splice_detection, frames_bgr)
            fut_upscale = executor.submit(self._detect_ai_upscaling, frames_gray)
            fut_noise   = executor.submit(self._analyze_noise_signature, frames_gray)
            fut_light   = executor.submit(self._analyze_lighting_consistency, frames_bgr)
            fut_srm     = executor.submit(self._analyze_srm_features, frames_gray)
            fut_grid    = executor.submit(self._analyze_frequency_grid, frames_gray)

            prnu_result      = fut_prnu.result()
            ela_result       = fut_ela.result()
            upscale_result   = fut_upscale.result()
            noise_result     = fut_noise.result()
            lighting_result  = fut_light.result()
            srm_result       = fut_srm.result()
            grid_result      = fut_grid.result()

        # Pesos de cada sub-análisis
        sub_suspicions = {
            "prnu":     prnu_result["suspicion"],
            "ela":      ela_result["suspicion"],
            "upscale":  upscale_result["suspicion"],
            "noise":    noise_result["suspicion"],
            "lighting": lighting_result["suspicion"],
            "srm":      srm_result["suspicion"],
            "grid":     grid_result["suspicion"]
        }

        weights = {
            "prnu":     0.25,
            "ela":      0.15,
            "upscale":  0.10,
            "noise":    0.20,
            "lighting": 0.05,
            "srm":      0.05,
            "grid":     0.20
        }

        total = sum(sub_suspicions[k] * weights[k] for k in sub_suspicions)

        # OVERRIDE: Si no hay huella de sensor absoluta (PRNU correlation < 0.015)
        # Es imposible que esto venga de una cámara real física, es CGI o IA.
        # No dejamos que el promedio lo diluya asumiendo que "por tener buena resolución es real".
        if prnu_result.get("prnu_consecutive_corr", 1.0) < 0.015:
            total = max(total, 0.95)

        return {
            "suspicion":            round(min(1.0, total), 3),
            "suspicion_components": sub_suspicions,
            "prnu":                 prnu_result,
            "ela_splice":           ela_result,
            "upscaling":            upscale_result,
            "noise_signature":      noise_result,
            "lighting_consistency": lighting_result,
            "srm":                  srm_result,
            "frequency_grid":       grid_result,
            "available":            True
        }
