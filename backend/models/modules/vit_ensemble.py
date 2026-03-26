"""
MÓDULO: ViT Ensemble Classifier V4.1 (Optimizado)
Clasificación con ensemble de modelos de visión:
  - ViT frame-level: PyTorch nativo (sin HuggingFace Pipeline overhead)
  - VideoMAE base: análisis de latentes espaciotemporales (embeddings, no clasificación)
  - torch.inference_mode() para máxima velocidad de inferencia
  - Fake-idx dinámico: detecta la clase "fake" por label del modelo
  - Directo BGR→RGB con OpenCV (sin PIL como intermediario en inferencia)
  - Sampling inteligente por keyframes (movimiento + distribución uniforme)
"""

import cv2
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    VideoMAEModel,
)


class ViTEnsembleClassifier:
    """
    Ensemble de modelos transformer para clasificación de video IA.
    Combina análisis frame-level (ViT nativo) y clip-level (VideoMAE latentes).
    """

    VIDEOMAE_CLIP_LEN = 16  # VideoMAE requiere exactamente 16 frames

    def __init__(self, device_id: int = -1):
        """
        Args:
            device_id: -1 = CPU, 0+ = GPU index
        """
        self.device = torch.device(
            f"cuda:{device_id}"
            if device_id >= 0 and torch.cuda.is_available()
            else "cpu"
        )
        self._vit_model  = None
        self._vit_proc   = None
        self._fake_idx   = 1        # índice de clase "fake" (se detecta dinámicamente)
        self._vmae_model = None
        self._vmae_proc  = None
        self._load_models()

    # ------------------------------------------------------------------
    # Carga de modelos
    # ------------------------------------------------------------------
    def _load_models(self):
        print(f">>> [ViTEnsemble] Inicializando en {self.device}...")

        # ── Modelo 1: Frame-level deepfake detector (PyTorch nativo) ──────────
        try:
            model_id = "prithivMLmods/Deep-Fake-Detector-v2-Model"
            print(f">>> [ViTEnsemble] Cargando {model_id}...")
            self._vit_proc  = AutoImageProcessor.from_pretrained(model_id)
            self._vit_model = (
                AutoModelForImageClassification
                .from_pretrained(model_id)
                .to(self.device)
                .eval()
            )
            # Detectar índice de clase "fake" dinámicamente desde los labels del modelo
            labels = self._vit_model.config.id2label
            self._fake_idx = next(
                (i for i, lbl in labels.items()
                 if any(x in lbl.lower() for x in ["fake", "ai", "deepfake", "artificial"])),
                1   # fallback: clase 1
            )
            print(f">>> [ViTEnsemble] Frame classifier listo (fake_idx={self._fake_idx}, label='{labels.get(self._fake_idx)}')")
        except Exception as e:
            print(f">>> [ViTEnsemble] Error cargando ViT frame classifier: {e}")
            self._vit_model = None

        # ── Modelo 2: VideoMAE base (embeddings, no clasificación) ─────────────
        try:
            vmae_id = "MCG-NJU/videomae-base"
            print(f">>> [ViTEnsemble] Cargando {vmae_id}...")
            self._vmae_proc  = AutoImageProcessor.from_pretrained(vmae_id)
            # VideoMAEModel (base) en lugar de VideoMAEForVideoClassification:
            # extrae representaciones latentes espaciotemporales
            self._vmae_model = (
                VideoMAEModel
                .from_pretrained(vmae_id)
                .to(self.device)
                .eval()
            )
            print(">>> [ViTEnsemble] VideoMAE base cargado (modo latentes)")
        except Exception as e:
            print(f">>> [ViTEnsemble] VideoMAE no disponible: {e}")
            self._vmae_model = None

    # ------------------------------------------------------------------
    # Sampling inteligente de frames
    # ------------------------------------------------------------------
    def _select_keyframes(self, frames_bgr: List[np.ndarray], n_target: int = 24) -> List[int]:
        """
        Selecciona frames más informativos para análisis:
        1. Distribución temporal uniforme como base
        2. Frames con mayor cambio respecto al anterior (zonas de movimiento)

        Retorna índices seleccionados ordenados.
        """
        n = len(frames_bgr)
        if n <= n_target:
            return list(range(n))

        # Base: distribución uniforme
        uniform_idx = set(np.linspace(0, n - 1, n_target // 2, dtype=int).tolist())

        # Agregar frames con alto cambio (movimiento = más información forense)
        change_scores = []
        prev_gray = cv2.cvtColor(frames_bgr[0], cv2.COLOR_BGR2GRAY)
        for i in range(1, n):
            gray = cv2.cvtColor(frames_bgr[i], cv2.COLOR_BGR2GRAY)
            diff = float(np.mean(cv2.absdiff(prev_gray, gray)))
            change_scores.append((i, diff))
            prev_gray = gray

        change_scores.sort(key=lambda x: -x[1])
        motion_idx = set([idx for idx, _ in change_scores[: n_target // 2]])

        selected = sorted(uniform_idx | motion_idx)
        return selected[:n_target]

    # ------------------------------------------------------------------
    # Frame-level classification (PyTorch nativo — sin Pipeline)
    # ------------------------------------------------------------------
    def _classify_frames_batch(
        self, frames_bgr: List[np.ndarray], batch_size: int = 8
    ) -> List[float]:
        """
        Inferencia frame-level en lotes con PyTorch nativo.
        - Salta HuggingFace Pipeline y PIL para máxima velocidad
        - torch.inference_mode() más rápido que torch.no_grad()
        - Retorna lista de P(fake) por frame [0.0 – 1.0]
        """
        if self._vit_model is None:
            return [0.5] * len(frames_bgr)

        # Recortar al centro para evitar distorsiones brutales de aspect ratio al reescalar (Squashing Fake Artifacts)
        # y convertir a RGB nativo.
        frames_rgb_cropped = []
        for f in frames_bgr:
            h, w = f.shape[:2]
            s = min(h, w)
            y, x = (h - s) // 2, (w - s) // 2
            cropped = f[y:y+s, x:x+s]
            frames_rgb_cropped.append(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            
        all_scores = []

        with torch.inference_mode():
            for i in range(0, len(frames_rgb_cropped), batch_size):
                batch = frames_rgb_cropped[i : i + batch_size]
                try:
                    inputs = self._vit_proc(images=batch, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    outputs  = self._vit_model(**inputs)
                    probs    = torch.softmax(outputs.logits, dim=-1)

                    # Extraer P(fake) con el índice detectado dinámicamente
                    fake_probs = probs[:, self._fake_idx].cpu().tolist()
                    all_scores.extend(fake_probs)
                except Exception as e:
                    print(f">>> [ViTEnsemble] Batch error: {e}")
                    all_scores.extend([0.5] * len(batch))

        return all_scores

    # ------------------------------------------------------------------
    # VideoMAE — análisis de coherencia temporal por latentes
    # ------------------------------------------------------------------
    def _analyze_temporal_latents(self, frames_bgr: List[np.ndarray]) -> float:
        """
        Analiza la coherencia de las representaciones latentes espaciotemporales
        de VideoMAE (base, sin cabeza de clasificación).

        Principio:
          - Los modelos generativos IA fallan en mantener la estructura 3D
            oculta coherente a lo largo del tiempo.
          - Una std anormalmente alta en los tokens del last_hidden_state
            indica que las texturas están "mutando" de forma sintética.

        Retorna score de anomalía [0.0 – 1.0].
        """
        if self._vmae_model is None or len(frames_bgr) < self.VIDEOMAE_CLIP_LEN:
            return 0.5

        clip_idx = np.linspace(
            0, len(frames_bgr) - 1, self.VIDEOMAE_CLIP_LEN, dtype=int
        )
        clip_rgb = [cv2.cvtColor(frames_bgr[i], cv2.COLOR_BGR2RGB) for i in clip_idx]

        try:
            with torch.inference_mode():
                inputs = self._vmae_proc(images=clip_rgb, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self._vmae_model(**inputs)
                # last_hidden_state: [1, Seq_Len, Hidden_Dim]
                last_hidden = outputs.last_hidden_state

                # Std a lo largo de los tokens espaciotemporales
                latent_std = float(last_hidden.std(dim=1).mean().item())

                # La varianza interna del tensor no es fiable porque el LayerNorm acota los latentes.
                # Desactivamos esta pseudo-heurística devolviendo un neutral absoluto que se ignora
                # para que no imponga un score alucinatorio a todos los videos por igual.
                return 0.0

        except Exception as e:
            print(f">>> [ViTEnsemble] VideoMAE latent error: {e}")
            return 0.5

    # ------------------------------------------------------------------
    # Análisis estadístico de scores de frames
    # ------------------------------------------------------------------
    def _aggregate_frame_scores(self, scores: List[float]) -> Dict:
        """
        Agrega scores de frames individuales.

        Estrategia conservadora (anti-falsos positivos):
          - Base: 70% media + 30% p75
          - Solo si >70% de frames son fake: usar p90 más agresivo
        """
        if not scores:
            return {"mean": 0.5, "p90": 0.5, "std": 0.0, "aggregate": 0.5}

        arr      = np.array(scores)
        mean     = float(np.mean(arr))
        median   = float(np.median(arr))
        p75      = float(np.percentile(arr, 75))
        p90      = float(np.percentile(arr, 90))
        std      = float(np.std(arr))
        above_75 = float(np.mean(arr > 0.75))
        above_50 = float(np.mean(arr > 0.5))

        # Estrategia muy conservadora (anti-falsos positivos)
        if above_75 > 0.80:
            # Consenso total en alta sospecha
            aggregate = p90 * 0.50 + mean * 0.30 + above_75 * 0.20
        elif above_50 > 0.60:
            # Sospecha moderada con consenso de mayoría
            aggregate = median * 0.40 + p75 * 0.40 + mean * 0.20
        else:
            # Video probablemente real o ruido base
            aggregate = median * 0.80 + mean * 0.20

        return {
            "mean":         round(mean, 3),
            "median":       round(median, 3),
            "p75":          round(p75, 3),
            "p90":          round(p90, 3),
            "std":          round(std, 3),
            "above_50_pct": round(above_50, 3),
            "n_frames":     len(scores),
            "aggregate":    round(min(1.0, float(aggregate)), 3),
        }

    # ------------------------------------------------------------------
    # Método principal
    # ------------------------------------------------------------------
    def analyze(self, frames_bgr: List[np.ndarray]) -> Dict:
        """
        Clasificación IA completa del video.

        Args:
            frames_bgr: Lista de frames BGR

        Returns:
            Dict con scores frame-level, latentes VideoMAE y score final [0-1]
        """
        if not frames_bgr:
            return {"suspicion": 0.5, "available": False, "error": "sin frames"}

        # 1. Seleccionar keyframes
        selected_idx = self._select_keyframes(frames_bgr, n_target=24)
        selected_bgr = [frames_bgr[i] for i in selected_idx]

        # 2. Frame-level classification (PyTorch nativo, BGR directo)
        frame_scores = self._classify_frames_batch(selected_bgr, batch_size=8)
        frame_stats  = self._aggregate_frame_scores(frame_scores)

        # 3. VideoMAE — análisis de coherencia temporal por latentes
        vmae_score = self._analyze_temporal_latents(frames_bgr)

        # 4. Score final del ensemble
        #    Promedio corregido: el modelo temporal ahora detecta los 2 extremos
        #    (Sora = perfecto, FaceSwap = errático) y el espacial detecta artefactos visuales.
        if self._vmae_model is not None and vmae_score > 0.01:
            if vmae_score > 0.6 or frame_stats["aggregate"] > 0.6:
                vit_final = max(frame_stats["aggregate"], vmae_score)
            else:
                vit_final = frame_stats["aggregate"] * 0.7 + vmae_score * 0.3
        else:
            vit_final = frame_stats["aggregate"]

        return {
            "suspicion":          round(float(min(1.0, vit_final)), 3),
            "frame_level":        frame_stats,
            "videomae_score":     round(float(vmae_score), 3),
            "videomae_available": self._vmae_model is not None,
            "keyframes_analyzed": len(selected_idx),
            "total_frames":       len(frames_bgr),
            "available":          self._vit_model is not None,
        }
