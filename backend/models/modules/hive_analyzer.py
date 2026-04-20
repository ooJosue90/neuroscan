import requests
import json
import base64
import logging
import time
import os
import cv2
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# ── Singleton de estado de rate-limit ────────────────────────────────────────
# Si Hive nos da 429, esperamos antes de volver a llamar.
_last_429_time: float = 0.0
_RATE_LIMIT_COOLDOWN_S = 60.0  # esperar 60s tras un 429 antes de reintentar


class HiveAnalyzer:
    """
    Analizador multimodal optimizado para ahorro de créditos.
    Extrae frames clave localmente para evitar el sobrecoste de video en la API.

    MEJORAS V10.3:
    - Reducido max_frames de 10 → 3 (reduce llamadas API un 70%).
    - Parada inmediata ante código 429 (Rate Limit) + cooldown global.
    - Agregado modo 'representativo': 1 frame inicio, 1 medio, 1 final.
    """

    API_URL = "https://api.thehive.ai/api/v3/hive/ai-generated-and-deepfake-content-detection"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = (
            api_key
            or os.getenv("HIVE_API_KEY")
            or "D9inOUrXtmBwOpWiMh4nnA=="
        ).strip()

        self.clases_excluidas = [
            'ai_generated', 'not_ai_generated', 'ai_generated_audio',
            'not_ai_generated_audio', 'none', 'inconclusive',
            'inconclusive_video', 'deepfake'
        ]

    # ── API call único ────────────────────────────────────────────────────────
    def analyze_image_bytes(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> Dict[str, Any]:
        """Analiza una sola imagen usando la arquitectura modular oficial de Hive."""
        return self._send_request(image_bytes, mime_type)

    def analyze_audio_bytes(self, audio_bytes: bytes, mime_type: str = "audio/mpeg") -> Dict[str, Any]:
        """Analiza un clip de audio usando la arquitectura modular oficial de Hive."""
        return self._send_request(audio_bytes, mime_type)

    def _is_rate_limited(self) -> bool:
        """Comprueba si estamos en cooldown tras un 429."""
        global _last_429_time
        return (time.monotonic() - _last_429_time) < _RATE_LIMIT_COOLDOWN_S

    def _send_request(self, media_bytes: bytes, mime_type: str) -> Dict[str, Any]:
        """Envía una solicitud genérica a la API de Hive con gestión de rate-limit."""
        global _last_429_time

        # Guard: Si estamos en cooldown tras un 429, no llamar
        if self._is_rate_limited():
            remaining = _RATE_LIMIT_COOLDOWN_S - (time.monotonic() - _last_429_time)
            logger.warning("Hive en cooldown por rate-limit. %.0fs restantes.", remaining)
            return {"suspicion": 0.5, "available": False, "error": "rate_limit_cooldown"}

        t0 = time.monotonic()
        try:
            base64_encoded = base64.b64encode(media_bytes).decode('utf-8')
            base64_data_uri = f"data:{mime_type};base64,{base64_encoded}"

            payload = {
                "media_metadata": True,
                "input": [{"media_base64": base64_data_uri}]
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            response = requests.post(
                self.API_URL, headers=headers,
                data=json.dumps(payload), timeout=30
            )

            if response.status_code == 429:
                # Rate limit — registrar tiempo y devolver sin explotar
                _last_429_time = time.monotonic()
                msg = f"Hive Error 429: {response.text}"
                print(f"!!! {msg} — Cooldown activado por {_RATE_LIMIT_COOLDOWN_S:.0f}s")
                return {"suspicion": 0.5, "available": False, "error": "rate_limit"}

            if response.status_code != 200:
                msg = f"Hive Error {response.status_code}: {response.text}"
                print(f"!!! {msg}")
                return {"suspicion": 0.5, "available": False, "error": msg}

            datos = response.json()
            return self._process_response(datos, t0)

        except Exception as e:
            logger.error("Fallo en Hive API: %s", e)
            return {"suspicion": 0.5, "available": False, "error": str(e)}

    # ── Análisis de video ─────────────────────────────────────────────────────
    def analyze(self, video_path: str, max_frames: int = 3) -> Dict[str, Any]:
        """
        Analiza un video extrayendo frames clave localmente.

        [OPT V10.3] max_frames reducido de 10 → 3 (inicio, centro, fin).
        Esto reduce las llamadas API un 70% y evita el 429 por exceso de peticiones.
        3 frames representativos son suficientes: los artefactos de IA son
        consistentes en todo el video — no necesitamos 10 muestras.
        """
        global _last_429_time

        # Guard de rate limit ANTES de abrir el video
        if self._is_rate_limited():
            remaining = _RATE_LIMIT_COOLDOWN_S - (time.monotonic() - _last_429_time)
            logger.warning("Hive saltado: en cooldown %.0fs.", remaining)
            return {"suspicion": 0.5, "available": False, "error": "rate_limit_cooldown"}

        t0 = time.monotonic()
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            return {"suspicion": 0.5, "available": False, "error": "Video inválido"}

        # Posiciones representativas: inicio (10%), centro (50%), final (90%)
        positions = [
            int(total_frames * 0.10),
            int(total_frames * 0.50),
            int(total_frames * 0.90),
        ]
        # Limitar a max_frames
        positions = positions[:max_frames]

        results = []

        for pos in positions:
            # Parar si entramos en rate-limit durante el análisis
            if self._is_rate_limited():
                logger.warning("Hive: rate-limit durante análisis — deteniendo.")
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if not ret:
                continue

            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            res = self.analyze_image_bytes(buffer.tobytes())

            error = str(res.get("error", ""))
            # Parar en cualquier error de cuota/rate-limit
            if "rate_limit" in error or "429" in error or "405" in error:
                logger.warning("Hive: parando análisis por cuota.")
                break

            if res.get("available"):
                results.append(res)

        cap.release()

        if not results:
            return {
                "suspicion": 0.5,
                "available": False,
                "error": "No se pudo analizar ningún frame (rate-limit o error)"
            }

        # Consolidar: máxima sospecha encontrada (conservador)
        max_suspicion = max(r["suspicion"] for r in results)
        suspects = [r.get("top_suspect", "unknown") for r in results if r.get("top_suspect") != "unknown"]
        top_suspect = max(set(suspects), key=suspects.count) if suspects else "unknown"

        return {
            "suspicion": max_suspicion,
            "available": True,
            "top_suspect": top_suspect,
            "frames_analyzed": len(results),
            "_latency_s": round(time.monotonic() - t0, 3)
        }

    # ── Procesado de respuesta ────────────────────────────────────────────────
    def _process_response(self, datos: Dict, t0: float) -> Dict[str, Any]:
        try:
            outputs = datos.get('output', [])
            if not outputs:
                return {"suspicion": 0.5, "available": False}

            resp = outputs[0].get('response', outputs[0])
            frame_data = resp.get('output', [resp])[0]

            clases = {c['class']: c['value'] for c in frame_data.get('classes', [])}

            es_ia_image = clases.get('ai_generated', 0.0)
            es_ia_audio = clases.get('ai_generated_audio', 0.0)
            es_ia = max(es_ia_image, es_ia_audio)

            specifics = [c for c in frame_data.get('classes', []) if c['class'] not in self.clases_excluidas]
            specifics.sort(key=lambda x: x['value'], reverse=True)

            top_suspect = "unknown"
            top_conf = 0.0
            if specifics and specifics[0]['value'] > 0.1:
                top_suspect = specifics[0]['class']
                top_conf = specifics[0]['value']
            elif es_ia_audio > 0.5:
                top_suspect = "synthetic_audio"
                top_conf = es_ia_audio

            return {
                "suspicion": round(es_ia, 4),
                "available": True,
                "top_suspect": top_suspect,
                "top_suspect_confidence": round(top_conf * 100, 2),
                "_latency_s": round(time.monotonic() - t0, 3)
            }
        except Exception as e:
            return {"suspicion": 0.5, "available": False, "error": str(e)}
