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

# Configurar logger
logger = logging.getLogger(__name__)

class HiveAnalyzer:
    """
    Analizador multimodal optimizado para ahorro de créditos.
    Extrae frames clave localmente para evitar el sobrecoste de video en la API.
    """
    
    API_URL = "https://api.thehive.ai/api/v3/hive/ai-generated-and-deepfake-content-detection"
    
    def __init__(self, api_key: Optional[str] = None):
        # API Key de Paul Farias (V3 Modular)
        self.api_key = (api_key or 
                        os.getenv("HIVE_API_KEY") or 
                        "LkWTl7EPcM12chORdyaj5g==").strip()
        
        self.clases_excluidas = [
            'ai_generated', 'not_ai_generated', 'ai_generated_audio', 
            'not_ai_generated_audio', 'none', 'inconclusive', 
            'inconclusive_video', 'deepfake'
        ]

    def analyze_image_bytes(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> Dict[str, Any]:
        """Analiza una sola imagen usando la arquitectura modular oficial de Hive."""
        t0 = time.monotonic()
        try:
            base64_encoded = base64.b64encode(image_bytes).decode('utf-8')
            base64_data_uri = f"data:{mime_type};base64,{base64_encoded}"

            # Estructura exacta según cURL oficial
            payload = {
                "media_metadata": True,
                "input": [{ 
                    "media_base64": base64_data_uri 
                }]
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(self.API_URL, headers=headers, data=json.dumps(payload), timeout=30)
            
            if response.status_code != 200:
                # Log detallado para el dashboard de Nueroscan
                msg = f"Hive Error {response.status_code}: {response.text}"
                print(f"!!! {msg}") 
                return {"suspicion": 0.5, "available": False, "error": msg}
                
            datos = response.json()
            return self._process_response(datos, t0)
            
        except Exception as e:
            logger.error(f"Fallo en Hive API: {e}")
            return {"suspicion": 0.5, "available": False, "error": str(e)}

    def analyze(self, video_path: str, max_frames: int = 10) -> Dict[str, Any]:
        """
        Analiza un video extrayendo frames localmente (Ahorro de créditos).
        En lugar de enviar el video, envía N imágenes representativas.
        """
        t0 = time.monotonic()
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            return {"suspicion": 0.5, "available": False, "error": "Video inválido"}

        # Calcular intervalos para obtener max_frames distribuidos
        interval = max(1, total_frames // max_frames)
        
        results = []
        frames_processed = 0
        
        for i in range(0, total_frames, interval):
            if frames_processed >= max_frames: break
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret: break
            
            # Codificar frame a JPEG en memoria
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            res = self.analyze_image_bytes(buffer.tobytes())
            
            if res.get("available"):
                results.append(res)
                frames_processed += 1
            
            # Si el primero falla por cuota (405), no seguimos intentando
            if "HTTP 405" in str(res.get("error", "")):
                break

        cap.release()

        if not results:
            return {"suspicion": 0.5, "available": False, "error": "No se pudo analizar ningún frame"}

        # Consolidar resultados (Veredicto conservador: máximo sospecha encontrada)
        max_suspicion = max(r["suspicion"] for r in results)
        
        # Encontrar el sospechoso más frecuente
        suspects = [r.get("top_suspect", "unknown") for r in results if r.get("top_suspect") != "unknown"]
        top_suspect = max(set(suspects), key=suspects.count) if suspects else "unknown"
        
        return {
            "suspicion": max_suspicion,
            "available": True,
            "top_suspect": top_suspect,
            "frames_analyzed": frames_processed,
            "_latency_s": round(time.monotonic() - t0, 3)
        }

    def _process_response(self, datos: Dict, t0: float) -> Dict[str, Any]:
        try:
            outputs = datos.get('output', [])
            if not outputs: return {"suspicion": 0.5, "available": False}
            
            resp = outputs[0].get('response', outputs[0])
            frame_data = resp.get('output', [resp])[0]
            
            clases = {c['class']: c['value'] for c in frame_data.get('classes', [])}
            es_ia = clases.get('ai_generated', 0.0)
            
            specifics = [c for c in frame_data.get('classes', []) if c['class'] not in self.clases_excluidas]
            specifics.sort(key=lambda x: x['value'], reverse=True)
            
            return {
                "suspicion": round(es_ia, 4),
                "available": True,
                "top_suspect": specifics[0]['class'] if specifics else "unknown",
                "top_suspect_confidence": round(specifics[0]['value'] * 100, 2) if specifics else 0.0
            }
        except Exception as e:
            return {"suspicion": 0.5, "available": False, "error": str(e)}
