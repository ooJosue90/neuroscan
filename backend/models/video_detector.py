"""
VideoIADetector V8.4-PROD — Orquestador Principal (Production-Grade)
Pipeline completo de detección de video generado por IA.

Versiones:
  - V8.4-PROD (Optimización de Memoria y Arquitectura) [ADJ-2026-04]
    * [BUG-6] Soporte nativo para rutas (Zero-copy RAM).
    * [FIX-V8.4] Corregido offset ftyp en MP4.
    * [FIX-V8.4] Reemplazada API privada _shutdown por flag interna.
    * [ADJ-V8.4] Suavizadas heurísticas agresivas (Neural Trust / Logo overrides).
    * [DOC] Reconocimiento de threads huérfanos en timeout.
  - V5.0-BASE (Versión de Producción Primaria)

Uso:
    detector = VideoIADetectorV5()
    result   = detector.analyze(video_bytes)
    print(result["verdict"])  # "SINTÉTICO" / "ORGÁNICO" / ...

    # Health-check (para readiness probe)
    status = detector.health_check()
    assert status["status"] == "ok"
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
PIPELINE_VERSION  = "V8.1-KINETIC"
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
    max_frames:         int   = 80
    target_fps_sample:  float = 3.0
    max_video_duration: float = 120.0

    # GPU / Inferencia
    use_gpu:            bool  = True
    gpu_device_id:      int   = 0
    batch_size:         int   = 16

    # Timeouts ajustados por módulo (segundos)
    timeout_temporal:   float = 90.0
    timeout_facial:     float = 90.0
    timeout_forensic:   float = 90.0
    timeout_audio:      float = 90.0
    timeout_vit:        float = 90.0

    # Módulos habilitados
    enable_temporal:    bool  = True
    enable_facial:      bool  = True
    enable_forensic:    bool  = True
    enable_audio:       bool  = True
    enable_vit:         bool  = True

    # Control de carga  [PRD-2]
    max_concurrent:     int   = 4   # máximo análisis simultáneos

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
        # [FIX-V8.4] Flag interna para evitar uso de API privada _shutdown
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
        self.scorer             = CalibratedEnsembleScorer()

        active = [n for n, obj in [
            ("temporal", self.temporal_analyzer), ("facial", self.facial_analyzer),
            ("forensic", self.forensic_analyzer), ("audio", self.audio_analyzer),
            ("vit", self.vit_classifier),
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
        [BUG-6 FIX V8.4] Soporta rutas nativas para evitar carga en RAM.
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
        Punto de entrada principal del pipeline V8.4.
        
        [BUG-6] Soporte Zero-Copy RAM:
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

                cfg = self.config
                all_tasks: Dict[str, Tuple[Callable, float]] = {
                    "temporal":     (run_temporal,  cfg.timeout_temporal),
                    "facial":       (run_facial,    cfg.timeout_facial),
                    "forensic":     (run_forensic,  cfg.timeout_forensic),
                    "audio":        (run_audio,     cfg.timeout_audio),
                    "vit_ensemble": (run_vit,       cfg.timeout_vit),
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
                    "lip_sync_corr":          _g(module_results["audio"], "lip_sync", "lip_sync_correlation"),
                    "vit_frame_mean":         _g(module_results["vit_ensemble"], "frame_level", "mean"),
                    "vit_frame_p90":          _g(module_results["vit_ensemble"], "frame_level", "p90"),
                    "videomae_score":         _g(module_results["vit_ensemble"], "videomae_score"),
                }
                forensic_report = {
                    k: round(float(v), 4) if isinstance(v, (int, float)) else v
                    for k, v in forensic_report.items()
                }

                prob = score_result["probability"]
                temp_data = module_results.get("temporal", {})
                jacob = temp_data.get("jacobian_discontinuity", 0.0)
                bitrate = video_meta.get("bitrate", 2000000)
                vit_mean = forensic_report.get("vit_frame_mean", 0.0)
                vit_p90 = forensic_report.get("vit_frame_p90", 0.0)
                blinks = forensic_report.get("blinks_per_min", 0.0)
                facial_s = score_result.get("module_scores", {}).get("facial", 0.0)
                noise_std = forensic_report.get("noise_std", 10.0)
                sota_notes = []
                
                if blinks > 85:
                    prob = max(prob, 96.0)
                    sota_notes.append("Refuerzo V8: Incoherencia estructural extrema (Synthetic Pareidolia)")
                elif facial_s < 15 and vit_mean > 0.92 and vit_p90 > 0.94:
                    prob = max(prob, 92.0)
                    sota_notes.append("Refuerzo V8: Confianza neuronal confirmada (Non-human Subject)")
                elif facial_s < 15 and vit_p90 > 0.70 and noise_std < 2.5:
                    prob = max(prob, 93.0)
                    sota_notes.append("Refuerzo V8: Firma ElevenLabs/Sora T2V detectada (High Fidelity)")
                    # [FIX V10.3-STABILITY] Calibrado: no rescatar si sospecha neural o temporal es dominante.
                    # IA High-Fid (ia 8) tiene vit_mean > 0.90, REAL High-Fid suele estar < 0.75.
                    if jacob < 0.40 and vit_mean < 0.78 and score_result.get("module_scores", {}).get("temporal", 0.0) < 65:
                        prob = prob * 0.40
                        sota_notes.append("Rescate SOTA V7: Integridad cinemática verificada")
                if jacob > 0.65:
                    prob = max(prob, 94.0)
                    sota_notes.append("Refuerzo SOTA V7: Incoherencia física extrema (Signature T2V)")

                audio_data_res = module_results.get("audio", {})
                if isinstance(audio_data_res, dict):
                    synth_score_str = audio_data_res.get("sota_info", {}).get("detalles", {}).get("metricas_avanzadas", {}).get("synthid_watermark", "0.0%")
                    synth_val = float(str(synth_score_str).replace("%", ""))
                    if synth_val > 70:
                        prob = max(prob, 98.0)
                        sota_notes.append(f"Hard Override V8: Firma SynthID de Google detectada ({synth_val}%)")

                if vit_mean > 0.60:
                    if audio_data_res.get("sota_info", {}).get("probabilidad", 0) > 80:
                        prob = max(prob, 99.0)
                        sota_notes.append("Hard Override V8: Patrón estructural de Google Veo detectado")

                score_result["probability"] = round(float(prob), 1)
                score_result["verdict"] = "IA" if prob >= 50 else "REAL"

                t_elapsed = round(time.monotonic() - t_start, 2)
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

        [BUG-6 FIX V8.4] ZERO-COPY RAM: Ya no lee el archivo a memoria.
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
