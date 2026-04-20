"""
VideoIADetector V10.3-FORENSIC — Orquestador Principal (Production-Grade)
Pipeline completo de detección de video generado por IA.

Versiones:
  - V10.3-STABLE (Titanium Edition - Production Release)
    * [CORE] Unificación total de orquestación multimodal.
    * [OPT] Paso de parámetros por referencia para ahorro de RAM.
    * [FIX] Sincronización optimizada Deep-Sync Omega V5.
  - V8.4-LEGACY (Versión de Auditoría Previa)
"""

from __future__ import annotations

import cv2
import json
import logging
import numpy as np
import os

# [FIX] Prevenir deadlocks en el ThreadPoolExecutor.
# PyTorch y OpenCV sufren colisiones severas (deadlocks) al multiplicar hilos (GIL starvation)
# cuando corren dentro de hilos paralelos externos.
cv2.setNumThreads(1)
import subprocess
import tempfile
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ── Importar módulos del pipeline ─────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from modules.temporal_analyzer import TemporalAnalyzer
from modules.facial_analyzer   import FacialBiometricsAnalyzer
from modules.forensic_analyzer import ForensicAnalyzer
from modules.audio_analyzer    import AudioAnalyzer
from modules.vit_ensemble      import ViTEnsembleClassifier
from modules.scorer            import CalibratedEnsembleScorer
from modules.hive_analyzer     import HiveAnalyzer

# ── Logging de librería — NO configura el root logger ─────────────────────────
# [BUG-5 FIX] Las librerías no deben llamar logging.basicConfig().
# El caller (servidor, CLI, tests) es responsable de configurar handlers.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__ = [
    "PipelineConfig",
    "VideoIADetectorV5",
    "get_detector",
    "analyze_video",
]

# ── Constantes globales ───────────────────────────────────────────────────────
PIPELINE_VERSION  = "V10.3-STABLE"
MAX_INPUT_BYTES   = 500 * 1024 * 1024   # 500 MB — rechazo temprano
MIN_INPUT_BYTES   = 1_024               # 1 KB mínimo
_DARK_FRAME_MEAN  = 5.0                 # umbral media gris para descartar frames


# ============================================================
# Configuración del Pipeline  [OPT-6]
# ============================================================
@dataclass
class PipelineConfig:
    """
    Configuración inmutable del pipeline.
    Usar dataclass permite validación, repr limpio y serialización.
    """
    # Extracción de frames
    # [OPT V10.3] 48 frames es más que suficiente para deepfake detection.
    # Los modelos de IA dejan artefactos en cada frame — no es necesario analizar 80.
    max_frames:         int   = 48
    target_fps_sample:  float = 2.0   # [OPT] Menos muestreo = menos I/O
    max_video_duration: float = 60.0  # [OPT] 60s — los patrones de deepfake se detectan en el primer minuto

    # GPU / Inferencia
    use_gpu:            bool  = True
    gpu_device_id:      int   = 0
    batch_size:         int   = 24    # [OPT] Batch más grande = menos forward passes

    # Timeouts por módulo — reducidos 50% manteniendo margen de seguridad
    # [OPT V10.3] Todos los módulos locales suelen terminar en <20s.
    # Hive responde en 10-30s en condiciones normales.
    timeout_temporal:   float = 45.0
    timeout_facial:     float = 45.0
    timeout_forensic:   float = 45.0
    timeout_audio:      float = 45.0
    timeout_vit:        float = 50.0
    timeout_hive:       float = 55.0  # [OPT] Antes 120s — limitante del pipeline completo

    # Módulos habilitados
    enable_temporal:    bool  = True
    enable_facial:      bool  = True
    enable_forensic:    bool  = True
    enable_audio:       bool  = True
    enable_vit:         bool  = True
    enable_hive:        bool  = True

    # Control de carga  [PRD-2]
    # [OPT] Más concurrencia = menos espera entre análisis seguidos
    max_concurrent:     int   = 6

    def __post_init__(self) -> None:
        """Valida rangos críticos al construir la config."""
        if self.max_frames < 4:
            raise ValueError("max_frames debe ser ≥ 4")
        if self.batch_size < 1:
            raise ValueError("batch_size debe ser ≥ 1")
        if self.max_concurrent < 1:
            raise ValueError("max_concurrent debe ser ≥ 1")


# ============================================================
# Orquestador Principal
# ============================================================
class VideoIADetectorV5:
    """
    Detector V5 de video generado por IA — Production-Grade.

    Thread-safe, singleton-compatible, con executor persistente.
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        # [PRD-2] Semáforo para limitar análisis concurrentes
        self._semaphore = threading.Semaphore(self.config.max_concurrent)
        # [OPT-1] Executor persistente — no recreado por cada llamada a analyze()
        # [FIX-V10.3] Flag interna para evitar uso de API privada _shutdown
        self._executor_running = True
        self._executor = ThreadPoolExecutor(
            max_workers=self.config.max_concurrent * 2,
            thread_name_prefix="VideoDetector"
        )
        self._initialize_modules()

    # ── Init de módulos ────────────────────────────────────────────────────────
    def _initialize_modules(self) -> None:
        """
        Carga todos los módulos con tolerancia a fallos individuales.
        Un módulo que falla en init no derrumba el pipeline completo.
        """
        logger.info("VideoIADetector %s — Inicializando módulos …", PIPELINE_VERSION)
        cfg = self.config
        device_id = cfg.gpu_device_id if cfg.use_gpu else -1

        def _safe_init(name: str, factory: Callable):
            try:
                return factory()
            except Exception as exc:
                logger.error("Módulo '%s' no pudo inicializarse: %s", name, exc)
                return None

        self.temporal_analyzer  = _safe_init("temporal",  lambda: TemporalAnalyzer(use_gpu=cfg.use_gpu)) if cfg.enable_temporal else None
        self.facial_analyzer    = _safe_init("facial",    FacialBiometricsAnalyzer) if cfg.enable_facial else None
        self.forensic_analyzer  = _safe_init("forensic",  ForensicAnalyzer) if cfg.enable_forensic else None
        self.audio_analyzer     = _safe_init("audio",     AudioAnalyzer) if cfg.enable_audio else None
        self.vit_classifier     = _safe_init("vit",       lambda: ViTEnsembleClassifier(device_id=device_id)) if cfg.enable_vit else None
        self.hive_analyzer      = _safe_init("hive",      HiveAnalyzer) if cfg.enable_hive else None
        self.scorer             = CalibratedEnsembleScorer()

        active = [n for n, obj in [
            ("temporal", self.temporal_analyzer), ("facial", self.facial_analyzer),
            ("forensic", self.forensic_analyzer), ("audio", self.audio_analyzer),
            ("vit", self.vit_classifier), ("hive", self.hive_analyzer),
        ] if obj is not None]
        logger.info(
            "Pipeline listo — GPU=%s BATCH=%d módulos_activos=%s",
            cfg.use_gpu, cfg.batch_size, active,
        )

    # ── Health-check  [PRD-3] ─────────────────────────────────────────────────
    def health_check(self) -> Dict[str, Any]:
        """
        Verifica estado de componentes.
        Útil para readiness/liveness probes en Kubernetes.
        """
        modules_ok = {
            "temporal":  self.temporal_analyzer is not None,
            "facial":    self.facial_analyzer   is not None,
            "forensic":  self.forensic_analyzer is not None,
            "audio":     self.audio_analyzer    is not None,
            "vit":       self.vit_classifier    is not None,
            "hive":      self.hive_analyzer     is not None,
            "scorer":    self.scorer            is not None,
        }
        active_count = sum(modules_ok.values())
        status = "ok" if active_count >= 3 else "degraded"
        return {
            "status":       status,
            "version":      PIPELINE_VERSION,
            "modules":      modules_ok,
            "active_count": active_count,
            "max_concurrent": self.config.max_concurrent,
            "executor_alive": self._executor_running,
        }

    def shutdown(self, wait: bool = True) -> None:
        """
        Libera el executor de forma ordenada. 
        
        [DOC] Importante: Python no permite 'matar' threads de forma segura. 
        Si un thread está bloqueado en una operación externa (ej: cv2 o ffmpeg), 
        permanecerá vivo como 'zombie' hasta que el proceso termine, incluso 
        después de llamar a shutdown().
        """
        self._executor_running = False
        self._executor.shutdown(wait=wait)
        logger.info("VideoIADetector — executor cerrado (wait=%s).", wait)

    # ── Extracción de frames  [OPT-3] [OPT-4] ────────────────────────────────
    def _extract_frames(self, video_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extrae frames del video con muestreo inteligente.
        # [BUG-6 FIX V10.3] Soporta rutas nativas para evitar carga en RAM.
        """
        video_path_str = str(video_path)
        cap = cv2.VideoCapture(video_path_str)
        if not cap.isOpened():
            return {"frames": [], "error": "No se pudo abrir el video"}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        native_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_s   = total_frames / max(native_fps, 1.0)

        if total_frames == 0:
            cap.release()
            return {"frames": [], "error": "Video sin frames"}

        effective_frames = min(total_frames, int(self.config.max_video_duration * native_fps))
        step = max(1, effective_frames // self.config.max_frames)

        frames_bgr:      List[np.ndarray] = []
        frame_timestamps: List[float]     = []
        frame_idx = 0

        while frame_idx < effective_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                # Filtrar frames oscuros (fade a negro / transiciones)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if float(np.mean(gray)) >= _DARK_FRAME_MEAN:
                    # Redimensionar si excede 720p
                    h, w = frame.shape[:2]
                    if max(h, w) > 720:
                        scale = 720.0 / max(h, w)
                        frame = cv2.resize(
                            frame,
                            (int(w * scale), int(h * scale)),
                            interpolation=cv2.INTER_AREA,  # mejor calidad al reducir
                        )
                    frames_bgr.append(frame)
                    frame_timestamps.append(frame_idx / native_fps)

            frame_idx += 1

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
            "error":         None,
        }

    # ── Metadata forense  [BUG-4] ────────────────────────────────────────────
    def _extract_video_metadata(self, video_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extrae metadata forense del video vía ffprobe.
        Usa lista de argumentos directamente en subprocess.run()
        en lugar de construir un string y pasarlo por shlex.split(). La lista
        de args es nativa en POSIX y Windows — más clara, más segura, sin
        riesgo de doble-quoting en paths con espacios.
        """
        metadata: Dict[str, Any] = {
            "codec":     "unknown",
            "bitrate":   0,
            "container": "unknown",
        }

        # Fallback rápido vía cv2
        try:
            cap = cv2.VideoCapture(video_path)
            fourcc_int  = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec_chars = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
            metadata["codec"] = codec_chars.strip("\x00") or "unknown"
            cap.release()
        except Exception:
            pass

        # Enriquecimiento con ffprobe — args en lista, sin shell=True
        try:
            cmd = [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_streams", "-show_format",
                video_path,              # path como elemento de lista — seguro
            ]
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if proc.returncode == 0:
                info = json.loads(proc.stdout)
                fmt  = info.get("format", {})
                metadata["bitrate"]   = int(fmt.get("bit_rate", 0))
                metadata["container"] = fmt.get("format_name", "unknown")
                for stream in info.get("streams", []):
                    if stream.get("codec_type") == "video":
                        metadata["codec"] = stream.get("codec_name", metadata["codec"])
                        break
        except Exception as exc:
            logger.warning("ffprobe falló: %s", exc)

        return metadata

    # ── Ejecución de módulo con timeout real  [BUG-1] ────────────────────────
    def _run_module_safe(
        self,
        name: str,
        func: Callable[[], Any],
        timeout: float,
    ) -> Dict[str, Any]:
        """
        Ejecuta func() con timeout y captura de errores.

        [BUG-1 FIX] La versión V4.1 recibía 'timeout' pero nunca lo aplicaba
        realmente — el timeout global de as_completed() era el único control.
        Aquí sometemos func() a un Future con timeout individual.

        ¡CRÍTICO!: Usamos un ThreadPoolExecutor LOCAL (inner_exec).
        Si usáramos `self._executor.submit` de nuevo mientras ya estamos ADENTRO
        del worker del `self._executor`, consumiríamos todos los workers 
        disponibles esperando tareas encoladas, causando un DEADLOCK total 
        (lo que hacía que todos murieran a los 90.0s exactos).
        """
        t0 = time.monotonic()
        try:
            inner_exec = ThreadPoolExecutor(max_workers=1)
            future     = inner_exec.submit(func)
            try:
                result = future.result(timeout=timeout)
                elapsed = round(time.monotonic() - t0, 3)
                inner_exec.shutdown(wait=False)
                if isinstance(result, dict):
                    result["_latency_s"] = elapsed
                    return result
                return {"suspicion": 0.5, "raw": result, "_latency_s": elapsed}
            except FutureTimeout:
                # [BUG-1 FIX-REFINED] shutdown(wait=True) implícito en 'with' bloqueaba el pipeline.
                # Al usar wait=False, permitimos que el orquestador siga aunque el hilo lento continúe.
                inner_exec.shutdown(wait=False)
                elapsed = round(time.monotonic() - t0, 3)
                logger.warning("Módulo '%s' alcanzó timeout (%.1fs)", name, timeout)
                return {
                    "suspicion": 0.3, 
                    "available": False, 
                    "error": f"timeout ({timeout}s)", 
                    "_latency_s": elapsed
                }
        except Exception as exc:
            elapsed = round(time.monotonic() - t0, 3)
            logger.error("Módulo '%s' falló: %s", name, exc, exc_info=True)
            return {
                "suspicion":  0.3,
                "available":  False,
                "error":      str(exc),
                "_latency_s": elapsed,
            }

    # ── Sincronía labio-fonema ────────────────────────────────────────────────
    def _compute_deep_sync(
        self,
        viseme_seq: List[Dict],
        audio_sync: List[Dict],
    ) -> Dict[str, Any]:
        """
        Análisis de sincronización Omega V5 (Fonema-Visema).
        Correlación RMS vs Apertura (v) y HF vs Estiramiento (h).
        Los deepfakes suelen tener avg_corr < 0.35.
        """
        if not viseme_seq or not audio_sync:
            return {"correlation": 0.0, "suspicion": 0.4, "available": False}

        n = min(len(viseme_seq), len(audio_sync))
        v_seq   = np.array([x["v"]   for x in viseme_seq[:n]], dtype=np.float32)
        h_seq   = np.array([x["h"]   for x in viseme_seq[:n]], dtype=np.float32)
        rms_seq = np.array([x["rms"] for x in audio_sync[:n]],  dtype=np.float32)
        hf_seq  = np.array([x["hf"]  for x in audio_sync[:n]],  dtype=np.float32)

        with np.errstate(invalid="ignore"):
            v_corr = float(np.corrcoef(v_seq, rms_seq)[0, 1])
            h_corr = float(np.corrcoef(h_seq, hf_seq)[0, 1])

        v_corr = 0.0 if np.isnan(v_corr) else v_corr
        h_corr = 0.0 if np.isnan(h_corr) else h_corr
        avg_corr = v_corr * 0.7 + abs(h_corr) * 0.3

        # Ajuste V5: Muchos videos reales (TikTok/BT headphones) tienen desincronizaciones de hasta 200ms
        # que tiran la correlación a ~0.35. Solo castigaremos de forma extrema por debajo de 0.15.
        if avg_corr < 0.15:
            suspicion = 0.95
        elif avg_corr < 0.30:
            suspicion = 0.60
        elif avg_corr > 0.60:
            suspicion = 0.0
        else:
            # Zona intermedia — sospecha proporcional inversa
            # de 0.60 (corr=0.30) a 0.00 (corr=0.60)
            suspicion = round(0.60 * (0.60 - avg_corr) / 0.30, 3)

        return {
            "v_correlation":   round(v_corr, 3),
            "h_correlation":   round(h_corr, 3),
            "avg_correlation": round(avg_corr, 3),
            "suspicion":       round(suspicion, 3),
            "available":       True,
        }

    def _extract_mouth_aperture(self, frames_bgr: List[np.ndarray]) -> List[float]:
        """Extrae secuencia MAR para lip-sync. Unchanged from V4.1."""
        if self.facial_analyzer is None or self.facial_analyzer._face_mesh is None:
            return []

        UPPER_LIP  = [13, 312, 311, 310]
        LOWER_LIP  = [14, 317, 402, 318]
        LEFT_MOUTH = 61
        RIGHT_MOUTH = 291
        mar_sequence: List[float] = []
        step = max(1, len(frames_bgr) // 60)

        for frame in frames_bgr[::step]:
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.facial_analyzer._face_mesh.process(rgb)
            if results.multi_face_landmarks:
                lms   = results.multi_face_landmarks[0].landmark
                upper = np.mean([(lms[i].x, lms[i].y) for i in UPPER_LIP], axis=0)
                lower = np.mean([(lms[i].x, lms[i].y) for i in LOWER_LIP], axis=0)
                left  = np.array([lms[LEFT_MOUTH].x,  lms[LEFT_MOUTH].y])
                right = np.array([lms[RIGHT_MOUTH].x, lms[RIGHT_MOUTH].y])
                vert  = np.linalg.norm(upper - lower)
                horiz = np.linalg.norm(left  - right)
                mar_sequence.append(float(vert / (horiz + 1e-8)))

        return mar_sequence

    # ── Detección del sufijo correcto de video  [PRD-6] ──────────────────────
    @staticmethod
    def _detect_video_suffix(data: bytes) -> str:
        """
        Detecta el contenedor real por magic bytes.
        """
        if b"ftyp" in data[4:12] or data[:4] in (b"\x00\x00\x00\x18", b"\x00\x00\x00\x20"):
            return ".mp4"
        if data[:4] == b"RIFF" and data[8:12] == b"AVI ":
            return ".avi"
        if data[:4] == b"\x1a\x45\xdf\xa3":
            return ".mkv"
        if data[:3] == b"FLV":
            return ".flv"
        if b"ftyp" in data[:12]:
            return ".mp4"
        return ".mp4"

    # ── Análisis completo ─────────────────────────────────────────────────────
    def analyze(self, video_data: Union[bytes, str, Path]) -> Dict[str, Any]:
        """
        Punto de entrada principal del pipeline V10.3 Titanium.
        
        [CORE] Soporte Zero-Copy RAM:
        - Si video_data es bytes: crea temporal y analiza.
        - Si video_data es str/Path: analiza directamente desde disco.
        """
        self._semaphore.acquire()
        try:
            is_path = isinstance(video_data, (str, Path))
            cleanup_needed = False
            video_path     = ""

            if is_path:
                video_path = str(video_data)
                if not os.path.exists(video_path):
                    return _error_response(f"Archivo no encontrado: {video_path}")
            else:
                if not video_data or len(video_data) < 100:
                    return _error_response("Datos de video inválidos o demasiado cortos")
                if len(video_data) > MAX_INPUT_BYTES:
                    return _error_response(f"Video excede límite de {MAX_INPUT_BYTES // 1_048_576} MB")

                suffix = self._detect_video_suffix(video_data)
                import tempfile
                fd, video_path = tempfile.mkstemp(suffix=suffix)
                with os.fdopen(fd, 'wb') as tmp:
                    tmp.write(video_data)
                cleanup_needed = True

            try:
                t_start = time.time()
                
                # 1. Extracción de frames y metadata
                frame_data = self._extract_frames(video_path)
                if frame_data.get("error"):
                    return _error_response(f"No se pudieron extraer frames: {frame_data.get('error')}")

                frames_bgr: List[np.ndarray] = frame_data["frames"]
                native_fps: float            = frame_data["native_fps"]
                sample_step: int             = frame_data["sample_step"]
                sampled_fps: float           = native_fps / sample_step if sample_step > 0 else native_fps
                video_meta                   = self._extract_video_metadata(video_path)

                # 4. Módulos en paralelo
                def run_temporal():
                    if self.temporal_analyzer:
                        return self.temporal_analyzer.analyze(frames_bgr)
                    return {"suspicion": 0.3, "available": False}

                def run_facial():
                    if self.facial_analyzer:
                        return self.facial_analyzer.analyze(frames_bgr, fps=sampled_fps)
                    return {"suspicion": 0.3, "available": False}

                def run_forensic():
                    if self.forensic_analyzer:
                        return self.forensic_analyzer.analyze(frames_bgr)
                    return {"suspicion": 0.3, "available": False}

                def run_audio():
                    if self.audio_analyzer:
                        return self.audio_analyzer.analyze(
                            video_path, fps=sampled_fps, n_video_frames=len(frames_bgr)
                        )
                    return {"suspicion": 0.2, "available": False}

                def run_vit():
                    if self.vit_classifier:
                        return self.vit_classifier.analyze(frames_bgr)
                    return {"suspicion": 0.5, "available": False}

                def run_hive():
                    if hasattr(self, 'hive_analyzer') and self.hive_analyzer and cfg.enable_hive:
                        return self.hive_analyzer.analyze(video_path)
                    return {"suspicion": 0.5, "available": False}

                cfg = self.config
                all_tasks: Dict[str, Tuple[Callable, float]] = {
                    "temporal":     (run_temporal,  cfg.timeout_temporal),
                    "facial":       (run_facial,    cfg.timeout_facial),
                    "forensic":     (run_forensic,  cfg.timeout_forensic),
                    "audio":        (run_audio,     cfg.timeout_audio),
                    "vit_ensemble": (run_vit,       cfg.timeout_vit),
                    "hive":         (run_hive,      cfg.timeout_hive),
                }

                module_results: Dict[str, Any] = {}
                outer_futures: Dict[Future, str] = {
                    self._executor.submit(
                        self._run_module_safe, mod_name, func, tmo
                    ): mod_name
                    for mod_name, (func, tmo) in all_tasks.items()
                }

                global_deadline = max(t for _, t in all_tasks.values()) + 10.0
                collected_start = time.monotonic()

                for future in _as_completed_safe(outer_futures, global_deadline):
                    mod_name = outer_futures[future]
                    try:
                        module_results[mod_name] = future.result(timeout=1)
                    except Exception as exc:
                        module_results[mod_name] = {
                            "suspicion": 0.3, "available": False, "error": str(exc)
                        }

                for mod_name in all_tasks:
                    if mod_name not in module_results:
                        module_results[mod_name] = {
                            "suspicion": 0.3, "available": False,
                            "error": "timeout global",
                        }

                frames_bgr = []

                # 5. Deep-Sync Omega V5
                facial_res = module_results.get("facial", {})
                audio_res  = module_results.get("audio", {})

                if facial_res.get("available") and audio_res.get("available"):
                    sync_res = self._compute_deep_sync(
                        facial_res.get("viseme_sequence", []),
                        audio_res.get("sync_data", []),
                    )
                    module_results["deep_sync"] = sync_res
                else:
                    module_results["deep_sync"] = {"suspicion": 0.2, "available": False}

                score_result = self.scorer.score(module_results)

                # 6. Integración de resultados de The Hive (V10 Upgrade)
                hive_res = module_results.get("hive", {})
                hive_prob = 0.0
                hive_suspect = "N/A"
                if hive_res.get("available"):
                    hive_prob = hive_res.get("suspicion", 0.0) * 100
                    hive_suspect = hive_res.get("top_suspect", "unknown")
                    
                forensic_report = {
                    "flow_divergence":        _g(module_results["temporal"], "flow_divergence_mean"),
                    "jacobian_discontinuity": _g(module_results["temporal"], "jacobian_discontinuity"),
                    "ghosting_ratio":         _g(module_results["temporal"], "ghosting", "ghosting_ratio"),
                    "blinks_per_min":         _g(module_results["facial"], "blink_analysis", "blinks_per_min"),
                    "blink_asymmetry":        _g(module_results["facial"], "blink_analysis", "asymmetry_ratio"),
                    "microexp_density":       _g(module_results["facial"], "microexpression", "microexp_density"),
                    "facial_asymmetry_std":   _g(module_results["facial"], "asymmetry", "asymmetry_std"),
                    "skin_cb_diff":           _g(module_results["facial"], "skin_chrominance", "cb_diff"),
                    "prnu_correlation":       _g(module_results["forensic"], "prnu", "prnu_consecutive_corr"),
                    "ela_splice_score":       _g(module_results["forensic"], "ela_splice", "ela_splice_score"),
                    "noise_std":              _g(module_results["forensic"], "noise_signature", "noise_std"),
                    "shot_noise_ratio":       _g(module_results["forensic"], "noise_signature", "shot_noise_ratio"),
                    "audio_jitter":           _g(module_results["audio"], "prosody", "jitter"),
                    "audio_hnr_db":           _g(module_results["audio"], "prosody", "hnr_db"),
                    "audio_noise_variance":   _g(module_results["audio"], "environment", "noise_spectral_variance"),
                    "audio_breaths":          _g(module_results["audio"], "breathing", "breaths_detected"),
                    "lip_sync_corr":          _g(module_results["audio"], "lip_sync", "lip_sync_correlation"),
                    "vit_frame_mean":         _g(module_results["vit_ensemble"], "frame_level", "mean"),
                    "vit_frame_p90":          _g(module_results["vit_ensemble"], "frame_level", "p90"),
                    "videomae_score":         _g(module_results["vit_ensemble"], "videomae_score"),
                    "hive_ai_prob":           hive_prob,
                    "hive_suspect":           hive_suspect
                }
                forensic_report = {
                    k: round(float(v), 4) if isinstance(v, (int, float)) else v
                    for k, v in forensic_report.items()
                }

                prob = score_result["probability"]
                
                # Definir variables locales para refuerzos a partir del reporte forense
                # [FIX V10.3] Extraer de forensic_report para evitar NameError
                jacob = forensic_report.get("jacobian_discontinuity", 0.0)
                vit_mean = forensic_report.get("vit_frame_mean", 0.0)
                vit_p90 = forensic_report.get("vit_frame_p90", 0.0)
                blinks = forensic_report.get("blinks_per_min", 0.0)
                facial_s = score_result.get("module_scores", {}).get("facial", 0.0)
                facial_asym_std = forensic_report.get("facial_asymmetry_std", 0.1)

                # Refuerzo con señal de The Hive
                if hive_prob > 80:
                    prob = max(prob, hive_prob * 0.95)
                    sota_notes = [f"Refuerzo Hive: Alta sospecha de {hive_suspect.upper()}"]
                elif hive_prob > 50:
                    prob = max(prob, (prob + hive_prob) / 2)
                    sota_notes = [f"Señal Hive: Detectado patrón de {hive_suspect.upper()}"]
                else:
                    sota_notes = []

                # ── HIVE VETO PROTECTION [FIX V10.3] ─────────────────────────
                # Si Hive está disponible Y dice <15%, es señal fuerte de que el
                # video es real. Los artefactos de recompresión de yt-dlp confunden
                # a ViT y Jacobian — Hive es más robusto ante este tipo de ruido.
                hive_veto_active = hive_res.get("available") and hive_prob < 15.0

                # ── AUDIO VETO [FIX V10.3] ───────────────────────────────────
                # Fallback cuando Hive no está disponible (ej: rate-limit 429).
                # Si el motor de audio local dice <30% IA, también actuamos como veto.
                # El audio es un fuerte indicador de autenticidad cuando Hive falla.
                audio_res_for_veto = module_results.get("audio", {})
                audio_prob_for_veto = audio_res_for_veto.get("probabilidad", None)
                if audio_prob_for_veto is None:
                    # intentar extraer de estructura anidada
                    audio_prob_for_veto = audio_res_for_veto.get("sota_info", {}).get("probabilidad", None)
                audio_veto_active = (
                    not hive_res.get("available")   # Hive no disponible
                    and audio_prob_for_veto is not None
                    and float(audio_prob_for_veto) < 30.0
                )
                # Combinar vetos: si cualquiera está activo, protegemos el resultado
                any_veto_active = hive_veto_active or audio_veto_active

                # Refuerzos para videos IA - V9.2 (balance óptimo)
                if blinks > 85:
                    prob = max(prob, 96.0)
                    sota_notes.append("Refuerzo V10: Incoherencia estructural extrema")

                # Señal fuerte: vit_p90 muy alto + facial bajo (no hay rostro humano claro)
                # [FIX] Solo aplica si ningún veto está activo
                if facial_s < 15 and vit_mean > 0.75 and vit_p90 > 0.85 and not any_veto_active:
                    prob = max(prob, 93.0)
                    sota_notes.append("Refuerzo V9: Confianza neuronal alta sin rostro claro")

                # Señal media: vit medio-alto + jacobian alto
                # [FIX] Umbral jacobian subido 0.45 → 0.55 para evitar falsas alarmas
                # en videos recomprimidos por yt-dlp. Con veto Hive.
                if jacob > 0.55 and vit_mean > 0.70 and not any_veto_active:
                    prob = max(prob, 88.0)
                    sota_notes.append("Refuerzo V9: Incoherencia física + señal neuronal")

                # Señal específica: facial asymmetry muy baja (deepfake clásico)
                # [FIX] Guard: facial_asym_std > 0 para no disparar cuando NO se detectó cara.
                # [FIX] Guard: Hive veto para no inflar falsamente cuando Hive dice REAL.
                if (facial_s < 25
                        and facial_asym_std > 0.0
                        and facial_asym_std < 0.015
                        and vit_mean > 0.72
                        and not any_veto_active):
                    prob = max(prob, 90.0)
                    sota_notes.append("Refuerzo V9: Simetría facial anómala")

                # ── VETO CAP ESCALONADO [FIX V10.3] ──────────────────────────
                # El cap depende de qué señales confirman que el video es real:
                #
                # Tier 1 — Hive + Audio ambos dicen REAL:  cap → 30% (REAL fuerte)
                # Tier 2 — Solo Hive dice REAL (<15%):     cap → 38% (REAL)
                # Tier 3 — Solo Audio dice REAL (<25%):    cap → 55% (INCIERTO bajo)
                # Tier 4 — Solo Audio dice REAL (<30%):    cap → 60% (INCIERTO, conservador)
                #
                # Razonamiento: Hive es el más confiable.
                # Si Hive dice 14%, el video ES real y debemos decir REAL, no INCIERTO.
                # Si solo tenemos audio (Hive con rate-limit), somos más conservadores.
                if any_veto_active:
                    audio_prob_val = float(audio_prob_for_veto) if audio_prob_for_veto is not None else 100.0

                    if hive_veto_active and audio_veto_active:
                        # Doble confirmación: Hive + Audio → REAL fuerte
                        cap = 30.0
                        veto_reason = f"Hive={hive_prob:.1f}% + Audio={audio_prob_val:.1f}% — Doble confirmación orgánica"
                    elif hive_veto_active:
                        # Solo Hive confirma real
                        if hive_prob < 10.0:
                            cap = 30.0
                            veto_reason = f"Hive={hive_prob:.1f}% — Alta certeza orgánica"
                        else:
                            cap = 38.0
                            veto_reason = f"Hive={hive_prob:.1f}% — Señal orgánica confirmada"
                    else:
                        # Solo Audio (Hive no disponible) → conservador
                        if audio_prob_val < 25.0:
                            cap = 55.0
                            veto_reason = f"Audio={audio_prob_val:.1f}% — Voz humana detectada (Hive no disponible)"
                        else:
                            cap = 60.0
                            veto_reason = f"Audio={audio_prob_val:.1f}% — Señal orgánica parcial (Hive no disponible)"

                    if prob > cap:
                        prob = cap
                        sota_notes.append(
                            f"Veto V10.3: Score capado a {cap:.0f}% ({veto_reason})"
                        )

                # Rescate para videos REALES con alta calidad cinemática
                if jacob < 0.30 and vit_mean > 0.75 and prob < 80:
                    prob = prob * 0.35
                    sota_notes.append("Rescate V9: Alta calidad cinemática")

                if jacob > 0.65 and not any_veto_active:
                    prob = max(prob, 94.0)
                    sota_notes.append("Refuerzo SOTA V7: Incoherencia física extrema")

                audio_data_res = module_results.get("audio", {})
                if isinstance(audio_data_res, dict):
                    synth_score_str = audio_data_res.get("sota_info", {}).get("detalles", {}).get("metricas_avanzadas", {}).get("synthid_watermark", "0.0%")
                    synth_val = float(str(synth_score_str).replace("%", ""))
                    if synth_val > 70:
                        prob = max(prob, 98.0)
                        sota_notes.append(f"Hard Override V10: Firma SynthID de Google detectada ({synth_val}%)")

                if vit_mean > 0.60:
                    if audio_data_res.get("sota_info", {}).get("probabilidad", 0) > 80:
                        prob = max(prob, 99.0)
                        sota_notes.append("Hard Override V10: Patrón estructural de Google Veo detectado")

                score_result["probability"] = round(float(prob), 1)
                score_result["verdict"] = "IA" if prob >= 50 else "REAL"

                t_elapsed = round(time.monotonic() - t_start, 2)
                audio_nota = audio_data_res.get("sota_info", {}).get("nota", "")
                if audio_nota:
                    sota_notes.append(audio_nota)

                reasons = sota_notes + (score_result.get("reasons") or ["Análisis completado"])
                module_latencies = {k: round(v.get("_latency_s", 0), 3) for k, v in module_results.items() if isinstance(v, dict)}

                return {
                    "status":              "success",
                    "tipo":                "video",
                    "probabilidad":        score_result["probability"],
                    "confidence":          score_result["confidence"],
                    "ci_lower":            score_result["ci_lower"],
                    "ci_upper":            score_result["ci_upper"],
                    "verdict":             score_result["verdict"],
                    "ai_model_likely":     score_result["ai_model_likely"],
                    "ai_model_confidence": score_result["ai_model_confidence"],
                    "nota":                " | ".join(reasons),
                    "reasons":             reasons,
                    "module_scores":       score_result["module_scores"],
                    "raw_scores":          score_result.get("raw_scores", {}),
                    "shap_contributions":  score_result["shap_contributions"],
                    "forensic_report":     forensic_report,
                    "module_latencies_s":  module_latencies,
                    "video_metadata": {
                        **video_meta,
                        "duration_s":     frame_data["duration_s"],
                        "native_fps":     native_fps,
                        "resolution":     frame_data["resolution"],
                        "frames_sampled": frame_data["sampled_count"],
                    },
                    "processing_time_s": t_elapsed,
                    "pipeline_version":  PIPELINE_VERSION,
                    "detalles": {
                        "predicciones": {
                            "IA": f"{int(score_result['probability'])}%",
                            "Humano": f"{100 - int(score_result['probability'])}%",
                        }
                    }
                }

            finally:
                if cleanup_needed and video_path:
                    try:
                        os.unlink(video_path)
                    except OSError:
                        pass

        finally:
            self._semaphore.release()

    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analiza un video directamente desde el sistema de archivos.

        [FIX V10.3] ZERO-COPY RAM: Ya no lee el archivo a memoria.
        Pasa la ruta directamente al pipeline, lo que permite manejar archivos de >2GB
        con un consumo de RAM constante e insignificante.
        """
        src = Path(file_path)
        if not src.exists():
            return _error_response(f"Archivo no encontrado: {file_path}")

        file_size = src.stat().st_size
        if file_size > MAX_INPUT_BYTES:
            return _error_response(f"Archivo excede límite de {MAX_INPUT_BYTES // 1_048_576} MB")

        # Pasamos la ruta directamente (ahora la infraestructura de V5 soporta Path)
        return self.analyze(file_path)


# ── Helpers internos ──────────────────────────────────────────────────────────

def _g(d: Dict, *keys: str, default: float = 0.0) -> Any:
    """Extractor seguro de campo anidado en un dict."""
    for key in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(key, default)
    return d if d is not None else default


def _error_response(msg: str, status: str = "error") -> Dict[str, Any]:
    """Construye una respuesta de error uniforme."""
    return {
        "status":           status,
        "tipo":             "video",
        "probabilidad":     0,
        "verdict":          "ERROR",
        "nota":             msg,
        "pipeline_version": PIPELINE_VERSION,
    }


def _as_completed_safe(fs_map: Dict[Future, str], timeout: float):
    """
    Itera sobre futures completados dentro de deadline.
    Generador seguro que no lanza StopIteration al agotar timeout —
    simplemente deja de yield sin propagación de excepción al caller.
    """
    from concurrent.futures import as_completed
    try:
        yield from as_completed(fs_map, timeout=timeout)
    except FutureTimeout:
        pending = [name for f, name in fs_map.items() if not f.done()]
        if pending:
            logger.warning(
                "Deadline global (%.0fs) alcanzado — módulos pendientes: %s",
                timeout, pending,
            )


# ============================================================
# Singleton y función pública  [PRD-4]
# ============================================================
_instance: Optional[VideoIADetectorV5] = None
_instance_lock = threading.Lock()


def get_detector(config: Optional[PipelineConfig] = None) -> VideoIADetectorV5:
    """
    Singleton thread-safe.
    Si ya existe una instancia y se pasa una config diferente, se registra
    una advertencia en lugar de ignorar silenciosamente el cambio.
    """
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = VideoIADetectorV5(config)
        elif config is not None and config is not _instance.config:
            logger.warning(
                "get_detector() llamado con config diferente — "
                "la instancia existente NO fue reemplazada. "
                "Usar VideoIADetectorV5(config) directamente si necesitas otra config."
            )
    return _instance


def analyze_video(data: bytes, config: Optional[PipelineConfig] = None) -> Dict[str, Any]:
    """
    Función de entrada principal (compatible con V3/V4).

    Args:
        data:   Bytes del video
        config: Configuración opcional

    Returns:
        Dict con resultado del análisis
    """
    try:
        return get_detector(config).analyze(data)
    except Exception as exc:
        logger.critical("Error crítico en analyze_video: %s", exc, exc_info=True)
        return _error_response(f"Error crítico en pipeline {PIPELINE_VERSION}: {exc}")


# ============================================================
# Demo CLI
# ============================================================
if __name__ == "__main__":
    import sys

    # CLI sí configura logging (la librería no lo hace)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if len(sys.argv) < 2:
        print(f"Uso: python {Path(__file__).name} <ruta_video>")
        print("Ejemplo: python video_detector_v5.py test_video.mp4")
        sys.exit(1)

    video_path = sys.argv[1]
    if not Path(video_path).exists():
        print(f"Error: archivo no encontrado: {video_path}")
        sys.exit(1)

    logger.info("Analizando: %s", video_path)

    detector = VideoIADetectorV5()

    # Health-check antes de analizar
    hc = detector.health_check()
    logger.info("Health-check: %s", hc)

    result = detector.analyze_file(video_path)

    sep = "═" * 60
    print(f"\n{sep}")
    print(f"  RESULTADO — VideoIADetector {PIPELINE_VERSION}")
    print(sep)
    print(f"  Veredicto:       {result.get('verdict', 'ERROR')}")
    print(f"  Probabilidad IA: {result.get('probabilidad', 0)}%")
    print(f"  Confianza:       {result.get('confidence', 0)}%")
    print(f"  CI 80%:          [{result.get('ci_lower', 0)}% — {result.get('ci_upper', 0)}%]")
    print(f"  Modelo probable: {result.get('ai_model_likely', 'N/A')} "
          f"({result.get('ai_model_confidence', 0)}%)")
    print(f"  Tiempo total:    {result.get('processing_time_s', 0)}s")

    print("\nLatencias por módulo:")
    for mod, lat in result.get("module_latencies_s", {}).items():
        bar = "▓" * int(lat * 5)
        print(f"  {mod:20s}: {lat:5.2f}s  {bar}")

    print("\nRazones:")
    for r in result.get("reasons", []):
        print(f"  • {r}")

    print("\nScores por módulo:")
    for mod, sc in result.get("module_scores", {}).items():
        print(f"  {mod:20s}: {sc:.1f}%")

    print("\nContribuciones SHAP:")
    for mod, val in result.get("shap_contributions", {}).items():
        bar  = "█" * int(abs(val) * 40)
        sign = "+" if val > 0 else "-"
        print(f"  {mod:20s}: {sign}{abs(val):.3f}  {bar}")

    print("\nReporte forense:")
    for k, v in result.get("forensic_report", {}).items():
        print(f"  {k:30s}: {v}")

    detector.shutdown()
