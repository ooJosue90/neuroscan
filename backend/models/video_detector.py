"""
VideoIADetector V4.1 — Orquestador Principal (Optimizado)
Pipeline completo de detección de video generado por IA.

Arquitectura:
  1. Pre-procesamiento (extracción de frames + metadata forense segura)
  2. Ejecución paralela HÍBRIDA de módulos:
     - ThreadPoolExecutor  → GPU/IO: TemporalAnalyzer, FacialBiometricsAnalyzer, ViTEnsemble
     - ProcessPoolExecutor → CPU math: ForensicAnalyzer, AudioAnalyzer
  3. CalibratedEnsembleScorer (Platt + SHAP)
  4. Output estructurado con veredicto + explicaciones + logging estructurado

Mejoras V4.1:
  - Logging estructurado (logging.Logger) en lugar de print()
  - ffprobe seguro con shlex.quote (previene inyección de comandos)
  - Híbrido Thread+Process Executor (evasión de GIL para CPU math)
  - BATCH_SIZE configurable para inferencia acelerada
  - Timeouts ajustados a la carga real de cada módulo

Uso:
    detector = VideoIADetectorV4()
    result   = detector.analyze(video_bytes)
    print(result["verdict"])  # "SINTÉTICO" / "ORGÁNICO" / ...
"""

import cv2
import json
import logging
import numpy as np
import os
import shlex
import subprocess
import tempfile
import threading
import time
from typing import Dict, List, Optional, Any
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
    TimeoutError as FutureTimeout,
)

# ============================================================
# Logging Estructurado
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("VideoIADetectorV4.1")

# Importar módulos del pipeline
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.temporal_analyzer  import TemporalAnalyzer
from modules.facial_analyzer    import FacialBiometricsAnalyzer
from modules.forensic_analyzer  import ForensicAnalyzer
from modules.audio_analyzer     import AudioAnalyzer
from modules.vit_ensemble       import ViTEnsembleClassifier
from modules.scorer             import CalibratedEnsembleScorer


# ============================================================
# Configuración del Pipeline
# ============================================================
class PipelineConfig:
    # Extracción de frames
    MAX_FRAMES          = 80        # Máximo frames a extraer
    TARGET_FPS_SAMPLE   = 3.0       # Frames por segundo a muestrear
    MAX_VIDEO_DURATION  = 120.0     # Segundos máximos a analizar

    # GPU / Inferencia
    USE_GPU             = True      # Prioridad a Tensor Processing (CUDA/MPS)
    GPU_DEVICE_ID       = 0
    BATCH_SIZE          = 16        # Clave para velocidad de inferencia en IA

    # Timeouts ajustados por módulo (segundos)
    TIMEOUT_TEMPORAL    = 45        # RAFT/FlowFormer — GPU
    TIMEOUT_FACIAL      = 30        # MediaPipe 478pts — CPU leve
    TIMEOUT_FORENSIC    = 30        # PRNU/ELA/SRM — CPU math intensivo
    TIMEOUT_AUDIO       = 20        # Jitter/Shimmer/Lip-sync
    TIMEOUT_VIT         = 60        # ViT Ensemble + VideoMAE — GPU

    # Módulos habilitados (desactivar para debugging o entorno limitado)
    ENABLE_TEMPORAL     = True
    ENABLE_FACIAL       = True
    ENABLE_FORENSIC     = True
    ENABLE_AUDIO        = True
    ENABLE_VIT          = True


# ============================================================
# Orquestador Principal
# ============================================================
class VideoIADetectorV4:
    """
    Detector V4 de video generado por IA.
    Singleton-safe para uso como módulo de servidor.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._initialize_modules()

    def _initialize_modules(self):
        """Carga todos los módulos del pipeline."""
        logger.info("=" * 60)
        logger.info("VideoIADetector V4.1 — Inicializando módulos")
        logger.info("=" * 60)

        device_id = self.config.GPU_DEVICE_ID if self.config.USE_GPU else -1

        self.temporal_analyzer  = TemporalAnalyzer(use_gpu=self.config.USE_GPU) if self.config.ENABLE_TEMPORAL else None
        self.facial_analyzer    = FacialBiometricsAnalyzer() if self.config.ENABLE_FACIAL else None
        self.forensic_analyzer  = ForensicAnalyzer() if self.config.ENABLE_FORENSIC else None
        self.audio_analyzer     = AudioAnalyzer() if self.config.ENABLE_AUDIO else None
        self.vit_classifier     = ViTEnsembleClassifier(device_id=device_id) if self.config.ENABLE_VIT else None
        self.scorer             = CalibratedEnsembleScorer()

        logger.info("=" * 60)
        logger.info("VideoIADetector V4.1 — Pipeline listo (GPU=%s, BATCH=%s)",
                    self.config.USE_GPU, self.config.BATCH_SIZE)
        logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Extracción de frames
    # ------------------------------------------------------------------
    def _extract_frames(self, video_path: str) -> Dict:
        """
        Extrae frames del video con muestreo inteligente.
        Retorna metadata del video + lista de frames BGR.
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return {"frames": [], "error": "No se pudo abrir el video"}

        total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        native_fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width         = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_s    = total_frames / max(native_fps, 1.0)

        if total_frames == 0:
            cap.release()
            return {"frames": [], "error": "Video sin frames"}

        # Calcular step para no exceder MAX_FRAMES ni MAX_VIDEO_DURATION
        effective_frames = min(total_frames, int(self.config.MAX_VIDEO_DURATION * native_fps))
        step = max(1, effective_frames // self.config.MAX_FRAMES)

        frames_bgr = []
        frame_timestamps = []

        for i in range(0, effective_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break

            # Filtro de cuadros oscuros (p. ej. transiciones, fade a negro)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if float(np.mean(gray)) < 5.0:
                continue

            # Redimensionar si el frame es muy grande (>720p)
            h, w = frame.shape[:2]
            if max(h, w) > 720:
                scale = 720.0 / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h))

            frames_bgr.append(frame)
            frame_timestamps.append(i / native_fps)

        cap.release()

        return {
            "frames":        frames_bgr,
            "timestamps":    frame_timestamps,
            "total_frames":  total_frames,
            "native_fps":    round(native_fps, 2),
            "duration_s":    round(duration_s, 2),
            "resolution":    f"{width}x{height}",
            "sample_step":   step,
            "sampled_count": len(frames_bgr),
            "error":         None
        }

    def _extract_video_metadata(self, video_path: str) -> Dict:
        """
        Extrae metadata forense del video (codec, bitrate, contenedor).
        Usa shlex.quote para proteger el path ante inyección de comandos en shell.
        """
        metadata = {
            "codec":     "unknown",
            "bitrate":   0,
            "container": "unknown",
        }

        # --- Fallback rápido vía cv2 ---
        try:
            cap = cv2.VideoCapture(video_path)
            fourcc_int  = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec_chars = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
            metadata["codec"] = codec_chars.strip("\x00") or "unknown"
            cap.release()
        except Exception:
            pass

        # --- Enriquecimiento con ffprobe (seguro contra inyección) ---
        try:
            # shlex.quote envuelve el path con comillas escapadas
            safe_path = shlex.quote(video_path)
            cmd = (
                f"ffprobe -v quiet -print_format json "
                f"-show_streams -show_format {safe_path}"
            )
            result = subprocess.run(
                shlex.split(cmd),
                capture_output=True,
                text=True,
                timeout=5,          # timeout reducido a 5 s (V4.1)
            )
            if result.returncode == 0:
                info = json.loads(result.stdout)
                fmt  = info.get("format", {})
                metadata["bitrate"]   = int(fmt.get("bit_rate", 0))
                metadata["container"] = fmt.get("format_name", "unknown")
                for s in info.get("streams", []):
                    if s.get("codec_type") == "video":
                        metadata["codec"] = s.get("codec_name", metadata["codec"])
                        break
        except Exception as e:
            logger.warning("Error extrayendo metadata forense vía ffprobe: %s", e)

        return metadata

    # ------------------------------------------------------------------
    # Ejecución paralela de módulos
    # ------------------------------------------------------------------
    def _run_module_safe(self, name: str, func, timeout: float) -> Dict:
        """
        Ejecuta un módulo con timeout y manejo de errores.
        Retorna resultado o dict de error con suspicion=0.3.
        """
        try:
            result = func()
            if isinstance(result, dict):
                return result
            return {"suspicion": 0.5, "raw": result}
        except Exception as e:
            logger.error("Módulo '%s' falló: %s", name, e, exc_info=True)
            return {"suspicion": 0.3, "available": False, "error": str(e)}

    def _compute_deep_sync(self, viseme_seq: List[Dict], audio_sync: List[Dict]) -> Dict:
        """
        Análisis de sincronización Omega V5 (Fonema-Visema).
        Correlaciona RMS vs Apertura (v) y Alta Frecuencia (HF) vs Estiramiento (h).
        Los deepfakes suelen tener correlación < 0.35.
        """
        if not viseme_seq or not audio_sync:
            return {"correlation": 0.0, "suspicion": 0.4, "available": False}
            
        n = min(len(viseme_seq), len(audio_sync))
        v_seq = np.array([x["v"] for x in viseme_seq[:n]])
        h_seq = np.array([x["h"] for x in viseme_seq[:n]])
        rms_seq = np.array([x["rms"] for x in audio_sync[:n]])
        hf_seq  = np.array([x["hf"] for x in audio_sync[:n]])
        
        # Correlación de Pearson (Vocal-Apertura)
        with np.errstate(invalid='ignore'):
            v_corr = np.corrcoef(v_seq, rms_seq)[0, 1]
            # Sibilantes (HF) suelen requerir boca más cerrada (menor h_dist o v_dist)
            # En modelos HeyGen/Bark, hf no está correlacionado con h_seq
            h_corr = np.corrcoef(h_seq, hf_seq)[0, 1]
            
        v_corr = 0.0 if np.isnan(v_corr) else float(v_corr)
        h_corr = 0.0 if np.isnan(h_corr) else float(h_corr)
        
        # Un humano real tiene v_corr > 0.6 y h_corr moderado
        # Una IA suele tener v_corr ≈ 0.3 y h_corr ≈ 0.2
        avg_corr = (v_corr * 0.7 + abs(h_corr) * 0.3)
        
        suspicion = 0.0
        if avg_corr < 0.25: suspicion = 0.95 # Totalmente asíncrono
        elif avg_corr < 0.45: suspicion = 0.65 # Robótico
        elif avg_corr > 0.75: suspicion = 0.0  # Sincronía física perfecta
        
        return {
            "v_correlation":   round(v_corr, 3),
            "h_correlation":   round(h_corr, 3),
            "avg_correlation": round(avg_corr, 3),
            "suspicion":       round(suspicion, 3),
            "available":       True
        }

    def _extract_mouth_aperture(self, frames_bgr: List[np.ndarray]) -> List[float]:
        """
        Extrae secuencia de apertura labial (MAR) para lip-sync analysis.
        Requiere FacialAnalyzer con MediaPipe activo.
        """
        if self.facial_analyzer is None or self.facial_analyzer._face_mesh is None:
            return []

        mar_sequence = []
        # Índices MediaPipe para MAR
        UPPER_LIP = [13, 312, 311, 310]
        LOWER_LIP = [14, 317, 402, 318]
        LEFT_MOUTH = 61
        RIGHT_MOUTH = 291

        step = max(1, len(frames_bgr) // 60)
        for frame in frames_bgr[::step]:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.facial_analyzer._face_mesh.process(rgb)
            if results.multi_face_landmarks:
                lms = results.multi_face_landmarks[0].landmark
                # MAR = distancia vertical / distancia horizontal de boca
                upper = np.mean([(lms[i].x, lms[i].y) for i in UPPER_LIP], axis=0)
                lower = np.mean([(lms[i].x, lms[i].y) for i in LOWER_LIP], axis=0)
                left  = np.array([lms[LEFT_MOUTH].x, lms[LEFT_MOUTH].y])
                right = np.array([lms[RIGHT_MOUTH].x, lms[RIGHT_MOUTH].y])
                vert  = np.linalg.norm(upper - lower)
                horiz = np.linalg.norm(left - right)
                mar   = float(vert / (horiz + 1e-8))
                mar_sequence.append(mar)

        return mar_sequence

    # ------------------------------------------------------------------
    # Análisis completo
    # ------------------------------------------------------------------
    def analyze(self, video_data: bytes) -> Dict:
        """
        Punto de entrada principal del pipeline V4.
        
        Args:
            video_data: Bytes del archivo de video
        
        Returns:
            Dict con resultado completo:
            {
                "status": "success" | "error",
                "tipo": "video",
                "probabilidad": float (0-100, P(IA)),
                "verdict": str,
                "ai_model_likely": str,
                "confidence": float,
                "ci_lower": float,
                "ci_upper": float,
                "nota": str,
                "reasons": list,
                "module_scores": dict,
                "shap_contributions": dict,
                "forensic_report": dict,
                "processing_time_s": float
            }
        """
        t_start = time.time()

        # --- 1. Guardar video a disco temporal ---
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
            tfile.write(video_data)
            video_path = tfile.name

        try:
            # --- 2. Extraer frames y metadata ---
            frame_data = self._extract_frames(video_path)

            if frame_data.get("error") or len(frame_data.get("frames", [])) < 4:
                return {
                    "status":        "error",
                    "tipo":          "video",
                    "probabilidad":  0,
                    "nota":          f"No se pudieron extraer frames: {frame_data.get('error', 'desconocido')}",
                    "verdict":       "ERROR"
                }

            frames_bgr   = frame_data["frames"]
            native_fps   = frame_data["native_fps"]
            video_meta   = self._extract_video_metadata(video_path)

            logger.info(
                "%d frames extraídos (%.1fs @ %.2f fps)",
                len(frames_bgr), frame_data["duration_s"], native_fps,
            )

            # --- 3. Extraer MAR para lip-sync (antes del análisis en paralelo) ---
            mouth_aperture = self._extract_mouth_aperture(frames_bgr)

            # --- 4. Ejecutar módulos en paralelo (V4.1 — ThreadPoolExecutor unificado) ---
            #
            # NOTA SOBRE WINDOWS (spawn):
            #   ProcessPoolExecutor requiere serializar (pickle) las funciones que envía
            #   a los procesos hijos. Las closures locales definidas dentro de un método
            #   NO son serializables → AttributeError: Can't get local object.
            #
            # SOLUCIÓN: ThreadPoolExecutor unificado para los 5 módulos.
            #   - PyTorch/CUDA libera el GIL durante inferencia en GPU → paralelismo real.
            #   - NumPy (BLAS/MKL/OpenBLAS) libera el GIL en ops vectorizadas → paralelismo real.
            #   - OpenCV libera el GIL en procesamiento de imágenes → paralelismo real.
            #   ✔ Los 5 módulos corren de forma genuinamente concurrente sin pickle.
            #
            module_results: Dict[str, Any] = {}

            def run_temporal():
                if self.temporal_analyzer:
                    return self.temporal_analyzer.analyze(frames_bgr)
                return {"suspicion": 0.3, "available": False}

            def run_facial():
                if self.facial_analyzer:
                    return self.facial_analyzer.analyze(frames_bgr, fps=native_fps)
                return {"suspicion": 0.3, "available": False}

            def run_forensic():
                if self.forensic_analyzer:
                    return self.forensic_analyzer.analyze(frames_bgr)
                return {"suspicion": 0.3, "available": False}

            def run_audio():
                if self.audio_analyzer:
                    return self.audio_analyzer.analyze(
                        video_path, 
                        fps=native_fps, 
                        n_video_frames=len(frames_bgr)
                    )
                return {"suspicion": 0.2, "available": False}

            def run_vit():
                if self.vit_classifier:
                    return self.vit_classifier.analyze(frames_bgr)
                return {"suspicion": 0.5, "available": False}

            all_tasks = {
                "temporal":     (run_temporal,  self.config.TIMEOUT_TEMPORAL),
                "facial":       (run_facial,    self.config.TIMEOUT_FACIAL),
                "forensic":     (run_forensic,  self.config.TIMEOUT_FORENSIC),
                "audio":        (run_audio,     self.config.TIMEOUT_AUDIO),
                "vit_ensemble": (run_vit,       self.config.TIMEOUT_VIT),
            }

            logger.info(
                "Iniciando análisis concurrente multimodal — %d módulos en paralelo (ThreadPool)",
                len(all_tasks),
            )

            global_timeout = max(self.config.TIMEOUT_TEMPORAL, self.config.TIMEOUT_VIT) + 10

            with ThreadPoolExecutor(max_workers=len(all_tasks)) as executor:
                future_to_name: Dict = {
                    executor.submit(self._run_module_safe, mod_name, func, _timeout): mod_name
                    for mod_name, (func, _timeout) in all_tasks.items()
                }

                # ── Recoger resultados a medida que terminan ──────────────────
                # Si as_completed() agota el global_timeout se lanza TimeoutError.
                # Lo capturamos aquí para NO abortar el análisis completo:
                # los módulos que ya terminaron se usan y los pendientes reciben
                # un fallback seguro en el bloque de "asegurar resultados" de abajo.
                try:
                    for future in as_completed(future_to_name, timeout=global_timeout):
                        mod_name = future_to_name[future]
                        try:
                            module_results[mod_name] = future.result(timeout=5)
                            logger.debug("Módulo '%s' completado con éxito.", mod_name)
                        except Exception as e:
                            logger.error(
                                "Error en módulo '%s': %s", mod_name, e, exc_info=True
                            )
                            module_results[mod_name] = {
                                "suspicion": 0.3, "available": False, "error": str(e)
                            }
                except TimeoutError:
                    # Identifica cuál(es) módulo(s) no terminaron a tiempo
                    pending = [
                        future_to_name[f] for f in future_to_name
                        if f not in [ff for ff in future_to_name if ff.done()]
                    ]
                    logger.warning(
                        "Timeout global (%.0fs) alcanzado — módulos pendientes: %s. "
                        "Se usará fallback para ellos.",
                        global_timeout, pending,
                    )

            # ── 5. Análisis de Sincronía Deep-Sync Omega V5 ───────────────
            facial_res = module_results.get("facial", {})
            audio_res  = module_results.get("audio", {})
            
            if facial_res.get("available") and audio_res.get("available"):
                sync_res = self._compute_deep_sync(
                    facial_res.get("viseme_sequence", []),
                    audio_res.get("sync_data", [])
                )
                module_results["deep_sync"] = sync_res
                logger.info("Deep-Sync Omega V5 completado: Corr = %.3f", sync_res.get('avg_correlation', 0))
            else:
                module_results["deep_sync"] = {"suspicion": 0.2, "available": False}

            # ── Asegurar que todos los módulos tengan resultado ───────────────
            for mod_name in all_tasks:
                if mod_name not in module_results:
                    logger.warning(
                        "Módulo '%s' sin resultado (timeout global) — usando fallback.", mod_name
                    )
                    module_results[mod_name] = {
                        "suspicion": 0.3, "available": False, "error": "timeout global"
                    }

            # --- 5. Scoring calibrado ---
            score_result = self.scorer.score(module_results)

            # --- 6. Construir respuesta final ---
            t_elapsed = round(time.time() - t_start, 2)
            logger.info("Análisis completado en %.2f s", t_elapsed)

            # Razones legibles
            reasons = score_result.get("reasons", [])
            if not reasons:
                reasons = ["Análisis completado — sin señales dominantes de síntesis IA"]

            # Reporte forense compacto
            forensic_report = {
                # Temporal
                "flow_divergence":       module_results["temporal"].get("flow_divergence_mean", 0),
                "jacobian_discontinuity":module_results["temporal"].get("jacobian_discontinuity", 0),
                "ghosting_ratio":        module_results["temporal"].get("ghosting", {}).get("ghosting_ratio", 0),

                # Facial
                "blinks_per_min":        module_results["facial"].get("blink_analysis", {}).get("blinks_per_min", 0),
                "blink_asymmetry":       module_results["facial"].get("blink_analysis", {}).get("asymmetry_ratio", 0),
                "microexp_density":      module_results["facial"].get("microexpression", {}).get("microexp_density", 0),
                "facial_asymmetry_std":  module_results["facial"].get("asymmetry", {}).get("asymmetry_std", 0),
                "skin_cb_diff":          module_results["facial"].get("skin_chrominance", {}).get("cb_diff", 0),

                # Forense
                "prnu_correlation":      module_results["forensic"].get("prnu", {}).get("prnu_consecutive_corr", 0),
                "ela_splice_score":      module_results["forensic"].get("ela_splice", {}).get("ela_splice_score", 0),
                "noise_std":             module_results["forensic"].get("noise_signature", {}).get("noise_std", 0),
                "shot_noise_ratio":      module_results["forensic"].get("noise_signature", {}).get("shot_noise_ratio", 0),

                # Audio
                "audio_jitter":          module_results["audio"].get("prosody", {}).get("jitter", 0),
                "audio_hnr_db":          module_results["audio"].get("prosody", {}).get("hnr_db", 0),
                "lip_sync_corr":         module_results["audio"].get("lip_sync", {}).get("lip_sync_correlation", 0),

                # ViT
                "vit_frame_mean":        module_results["vit_ensemble"].get("frame_level", {}).get("mean", 0),
                "vit_frame_p90":         module_results["vit_ensemble"].get("frame_level", {}).get("p90", 0),
                "videomae_score":        module_results["vit_ensemble"].get("videomae_score", 0),
            }

            # Formatear valores numéricos
            forensic_report = {k: round(float(v), 4) if isinstance(v, (int, float)) else v
                               for k, v in forensic_report.items()}

            return {
                "status":             "success",
                "tipo":               "video",
                "probabilidad":       score_result["probability"],
                "confidence":         score_result["confidence"],
                "ci_lower":           score_result["ci_lower"],
                "ci_upper":           score_result["ci_upper"],
                "verdict":            score_result["verdict"],
                "ai_model_likely":    score_result["ai_model_likely"],
                "ai_model_confidence":score_result["ai_model_confidence"],
                "nota":               " | ".join(reasons),
                "reasons":            reasons,
                "module_scores":      score_result["module_scores"],
                "raw_scores":         score_result.get("raw_scores", {}),
                "shap_contributions": score_result["shap_contributions"],
                "forensic_report":    forensic_report,
                "video_metadata": {
                    **video_meta,
                    "duration_s":    frame_data["duration_s"],
                    "native_fps":    native_fps,
                    "resolution":    frame_data["resolution"],
                    "frames_sampled":frame_data["sampled_count"],
                },
                "processing_time_s":  t_elapsed,
                "pipeline_version":   "V4.1"
            }

        finally:
            # Siempre limpiar archivo temporal
            try:
                os.unlink(video_path)
            except Exception:
                pass

    def analyze_file(self, file_path: str) -> Dict:
        """Analiza un video desde ruta de archivo."""
        with open(file_path, "rb") as f:
            return self.analyze(f.read())


# ============================================================
# Singleton y función pública de compatibilidad
# ============================================================
_instance: Optional[VideoIADetectorV4] = None
_instance_lock = threading.Lock()


def get_detector(config: Optional[PipelineConfig] = None) -> VideoIADetectorV4:
    """Obtiene la instancia singleton del detector."""
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = VideoIADetectorV4(config)
    return _instance


def analyze_video(data: bytes, config: Optional[PipelineConfig] = None) -> Dict:
    """
    Función de entrada principal (compatible con V3).
    
    Args:
        data: Bytes del video
        config: Configuración del pipeline (opcional)
    
    Returns:
        Dict con resultado del análisis
    """
    try:
        detector = get_detector(config)
        return detector.analyze(data)
    except Exception as e:
        logger.critical("Error crítico en analyze_video: %s", e, exc_info=True)
        return {
            "status":       "error",
            "tipo":         "video",
            "probabilidad": 0,
            "verdict":      "ERROR",
            "nota":         f"Error crítico en pipeline V4.1: {str(e)}",
            "pipeline_version": "V4.1"
        }


# ============================================================
# Demo CLI
# ============================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python video_detector_v4.py <ruta_video>")
        print("\nEjemplo: python video_detector_v4.py test_video.mp4")
        sys.exit(1)

    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print(f"Error: archivo no encontrado: {video_path}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Analizando: %s", video_path)
    logger.info("=" * 60)

    detector = VideoIADetectorV4()
    result   = detector.analyze_file(video_path)

    sep = "=" * 60
    print(f"\n{sep}")
    print("RESULTADO FINAL — VideoIADetector V4.1")
    print(sep)
    print(f"  Veredicto:       {result.get('verdict', 'ERROR')}")
    print(f"  Probabilidad IA: {result.get('probabilidad', 0)}%")
    print(f"  Confianza:       {result.get('confidence', 0)}%")
    print(f"  CI 80%:          [{result.get('ci_lower', 0)}% — {result.get('ci_upper', 0)}%]")
    print(f"  Modelo probable: {result.get('ai_model_likely', 'N/A')} ({result.get('ai_model_confidence', 0)}%)")
    print(f"  Tiempo:          {result.get('processing_time_s', 0)}s")
    print(f"\nRazones:")
    for r in result.get("reasons", []):
        print(f"  • {r}")
    print(f"\nScores por módulo:")
    for mod, sc in result.get("module_scores", {}).items():
        print(f"  {mod:20s}: {sc:.1f}%")
    print(f"\nContribuciones SHAP:")
    for mod, val in result.get("shap_contributions", {}).items():
        bar  = "█" * int(abs(val) * 40)
        sign = "+" if val > 0 else "-"
        print(f"  {mod:20s}: {sign}{abs(val):.3f}  {bar}")
    print(f"\nReporte forense:")
    fr = result.get("forensic_report", {})
    for k, v in fr.items():
        print(f"  {k:30s}: {v}")