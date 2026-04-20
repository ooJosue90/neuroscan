import logging
import numpy as np
import warnings
import os
import tempfile
import subprocess
from typing import List, Dict, Optional, Tuple
import sys
from pathlib import Path

# Configurar logger para este módulo
logger = logging.getLogger(__name__)

# Importar el detector SOTA de audio
try:
    from models.audio_detector import analyze_audio
except ImportError:
    # Fallback si el path no está configurado
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from audio_detector import analyze_audio


class AudioAnalyzer:
    """
    Análisis de audio avanzado para detección de TTS sintético.
    Targets: ElevenLabs, Play.ht, Bark, VoiceCloning, HeyGen audio.
    """

    def __init__(self):
        self._parselmouth_available = False
        self._librosa_available = False
        self._wav2vec_model = None
        self._wav2vec_processor = None
        self._check_dependencies()

    def _check_dependencies(self):
        try:
            import parselmouth
            self._parselmouth_available = True
            print(">>> [AudioAnalyzer] parselmouth disponible (Jitter/Shimmer/HNR)")
        except ImportError:
            print(">>> [AudioAnalyzer] parselmouth no disponible, usando estimaciones librosa")

        try:
            import librosa
            self._librosa_available = True
            print(">>> [AudioAnalyzer] librosa disponible")
        except ImportError:
            self._librosa_available = False
            print(">>> [AudioAnalyzer] librosa NO disponible — análisis de audio deshabilitado")

    def _load_audio(self, video_path: str, max_duration: float = 90.0) -> Tuple[Optional[np.ndarray], int]:
        """Carga audio del video usando una extracción previa a WAV temporal."""
        if not self._librosa_available:
            return None, 0
        
        temp_wav = None
        try:
            import librosa
            
            # [FIX V10.3] Extraer audio a WAV usando ffmpeg para máxima compatibilidad
            # Usamos un nombre de archivo limpio para evitar problemas en Windows
            temp_wav = os.path.join(tempfile.gettempdir(), f"talos_audio_{os.getpid()}.wav")
            
            cmd = [
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                temp_wav
            ]
            
            # Ejecutar y capturar errores de subprocess
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning(f"ffmpeg falló (code {result.returncode}). Intentando carga directa.")
                # Fallback: intentar cargar directamente si ffmpeg falla
                y, sr = librosa.load(video_path, sr=16000, duration=max_duration, mono=True)
                return y, sr
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y, sr = librosa.load(temp_wav, sr=16000, duration=max_duration)
            
            if len(y) < 1000:
                return None, 0
            return y, sr
            
        except Exception as e:
            logger.error(f">>> [AudioAnalyzer] Error cargando audio: {e}")
            return None, 0
        finally:
            if temp_wav and os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except:
                    pass

    # ------------------------------------------------------------------
    # Jitter, Shimmer, HNR via Parselmouth (Praat Python binding)
    # ------------------------------------------------------------------
    def _analyze_prosody_parselmouth(self, y: np.ndarray, sr: int) -> Dict:
        """
        Extrae features prosódicas de alta precisión con Praat.
        
        Valores normales en voz humana:
          - Jitter local: 0.2-1.0% → TTS: < 0.1% (demasiado regular)
          - Shimmer local: 1-5% → TTS: < 0.5%
          - HNR: 15-20 dB → TTS: > 22 dB (demasiado limpio)
        """
        try:
            import parselmouth
            from parselmouth.praat import call

            sound = parselmouth.Sound(y, sampling_frequency=float(sr))

            # Pitch track
            pitch = call(sound, "To Pitch", 0.0, 75.0, 500.0)
            n_voiced = call(pitch, "Count voiced frames")

            if n_voiced < 10:
                return {"jitter": 0.0, "shimmer": 0.0, "hnr_db": 0.0, "voiced_ratio": 0.0, "suspicion": 0.5, "available": False, "method": "parselmouth_short"}

            # Jitter
            point_process = call(sound, "To PointProcess (periodic, cc)", 75.0, 500.0)
            jitter_local  = call(point_process, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3)

            # Shimmer
            shimmer_local = call([sound, point_process], "Get shimmer (local)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)

            # Harmonicity (HNR)
            harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75.0, 0.1, 1.0)
            hnr = call(harmonicity, "Get mean", 0.0, 0.0)

            # Pitch statistics
            total_frames = call(pitch, "Get number of frames")
            voiced_ratio = n_voiced / max(total_frames, 1)

            # Estadísticas de pitch
            f0_mean = call(pitch, "Get mean", 0.0, 0.0, "Hertz")
            f0_std  = call(pitch, "Get standard deviation", 0.0, 0.0, "Hertz")

            # Evaluar sospecha
            suspicion = 0.0

            # Jitter: TTS < 0.001 (0.1%), humano 0.002-0.010 (0.2-1%)
            if jitter_local is not None and not (isinstance(jitter_local, float) and np.isnan(jitter_local)):
                if jitter_local < 0.001:
                    suspicion += 0.35
                elif jitter_local < 0.002:
                    suspicion += 0.15

            # Shimmer: TTS < 0.005 (0.5%), humano 0.01-0.05 (1-5%)
            if shimmer_local is not None and not (isinstance(shimmer_local, float) and np.isnan(shimmer_local)):
                if shimmer_local < 0.005:
                    suspicion += 0.30
                elif shimmer_local < 0.01:
                    suspicion += 0.10

            # HNR: TTS > 22 dB, humano 15-20 dB
            if hnr is not None and not (isinstance(hnr, float) and np.isnan(hnr)):
                if hnr > 25:
                    suspicion += 0.25
                elif hnr > 22:
                    suspicion += 0.10

            # F0 variación: TTS tiene pitch más monótono
            if f0_std is not None and not np.isnan(f0_std) and f0_std < 10:
                suspicion += 0.15

            return {
                "jitter":       round(float(jitter_local) if jitter_local else 0.0, 5),
                "shimmer":      round(float(shimmer_local) if shimmer_local else 0.0, 5),
                "hnr_db":       round(float(hnr) if hnr else 0.0, 2),
                "f0_mean":      round(float(f0_mean) if f0_mean else 0.0, 1),
                "f0_std":       round(float(f0_std) if f0_std else 0.0, 1),
                "voiced_ratio": round(float(voiced_ratio), 3),
                "suspicion":    round(min(1.0, suspicion), 3),
                "method":       "parselmouth"
            }

        except Exception as e:
            print(f">>> [AudioAnalyzer] Parselmouth error: {e}")
            return self._analyze_prosody_librosa(y, sr)

    def _analyze_prosody_librosa(self, y: np.ndarray, sr: int) -> Dict:
        """Estimación de features prosódicas con librosa (menos preciso que Praat)."""
        try:
            import librosa

            # Fundamental frequency con librosa
            f0, voiced_flag, voiced_prob = librosa.pyin(
                y, fmin=75, fmax=500, sr=sr,
                frame_length=2048, hop_length=512
            )
            f0_voiced = f0[voiced_flag & (f0 > 0)]

            if len(f0_voiced) < 10:
                # No hay suficiente voz humana para evaluar deepfake de audio. (Sora, T2V, música)
                return {
                    "jitter": 0.0, "shimmer": 0.0, "hnr_db": 0.0, 
                    "suspicion": 0.5, "method": "librosa", "available": False
                }

            # Jitter aproximado como variación relativa de F0
            jitter_approx = float(np.mean(np.abs(np.diff(f0_voiced))) / (np.mean(f0_voiced) + 1e-8))

            # ZCR y MFCC con filtro de silencios casero (VAD)
            rms = librosa.feature.rms(y=y)[0]
            voiced_mask = (rms >= np.mean(rms) * 0.1)

            zcr       = librosa.feature.zero_crossing_rate(y)[0]
            zcr_voiced = zcr[voiced_mask] if np.sum(voiced_mask) > 10 else zcr
            zcr_std   = float(np.std(zcr_voiced))
            
            mfccs     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            mfccs_voiced = mfccs[:, voiced_mask] if np.sum(voiced_mask) > 10 else mfccs
            mfcc_var  = float(np.var(mfccs_voiced))

            # Spectral flatness
            flatness     = librosa.feature.spectral_flatness(y=y)[0]
            flatness_voiced = flatness[voiced_mask] if np.sum(voiced_mask) > 10 else flatness
            flatness_var = float(np.var(flatness_voiced))

            suspicion = 0.0
            if jitter_approx < 0.005:
                suspicion += 0.30
            if zcr_std < 0.02:
                suspicion += 0.15
            if mfcc_var < 60:
                suspicion += 0.20
            if flatness_var < 5e-5:
                suspicion += 0.15

            return {
                "jitter":        round(jitter_approx, 5),
                "shimmer":       0.0,
                "hnr_db":        0.0,
                "f0_mean":       round(float(np.mean(f0_voiced)), 1),
                "f0_std":        round(float(np.std(f0_voiced)), 1),
                "zcr_std":       round(zcr_std, 5),
                "mfcc_variance": round(mfcc_var, 2),
                "flatness_var":  round(flatness_var, 8),
                "suspicion":     round(min(1.0, suspicion), 3),
                "method":        "librosa"
            }
        except Exception as e:
            return {"suspicion": 0.3, "error": str(e), "method": "failed"}

    # ------------------------------------------------------------------
    # Breathiness, Creaky Voice, Onset Irregularity
    # ------------------------------------------------------------------
    def _analyze_naturalness_markers(self, y: np.ndarray, sr: int) -> Dict:
        """
        Detecta marcadores de voz natural ausentes en TTS:
        - Breathiness: componente de ruido de respiración mezclado con voz
        - Creaky voice (glottalization): irregularidad glótica natural
        - Onset irregularity: las pausas TTS son perfectamente rectangulares
        - Micro-silencios: humanos insertan micro-pausas (20-80ms) naturales
        """
        try:
            import librosa

            # --- RMS envelope para onset analysis ---
            rms         = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
            silence_mask = (rms < np.mean(rms) * 0.1).astype(int)
            voice_mask   = (silence_mask == 0)

            # --- Spectral Flatness temporal ---
            flatness    = librosa.feature.spectral_flatness(y=y)[0]
            flatness_voiced = flatness[voice_mask] if np.sum(voice_mask) > 10 else flatness
            flat_var    = float(np.var(flatness_voiced))  # Alta varianza = más natural
            flat_mean   = float(np.mean(flatness_voiced))

            # Detectar transiciones silencio ↔ voz
            transitions  = np.diff(silence_mask)
            onset_frames = np.where(transitions == -1)[0]  # Silencio → voz
            offset_frames= np.where(transitions == 1)[0]   # Voz → silencio

            # Intervalos entre onsets
            onset_irregularity = 0.0
            if len(onset_frames) > 3:
                intervals = np.diff(onset_frames)
                onset_irregularity = float(np.std(intervals))  # Alta std = natural

            # Duración de silencios: TTS tiene silencios exactamente iguales
            silence_durations = []
            if len(onset_frames) > 1 and len(offset_frames) > 0:
                for i, off in enumerate(offset_frames):
                    # Buscar el siguiente onset
                    next_onsets = onset_frames[onset_frames > off]
                    if len(next_onsets) > 0:
                        dur = int(next_onsets[0]) - int(off)
                        silence_durations.append(dur)

            silence_dur_std = float(np.std(silence_durations)) if len(silence_durations) > 2 else 0.0

            # --- Breathiness proxy: energía de alta frecuencia en zonas vocalizadas ---
            # Voz con breathiness: espectrograma tiene energía difusa en HF durante vocales
            stft        = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
            n_freqs     = stft.shape[0]
            hf_ratio    = float(np.mean(stft[n_freqs//2:, :]) / (np.mean(stft[:n_freqs//2, :]) + 1e-8))

            suspicion = 0.0
            # Flatness sin varianza: TTS tiene textura espectral uniforme
            if flat_var < 1e-5:
                suspicion += 0.30
            # Onsets perfectamente regulares (TTS)
            if onset_irregularity < 3.0 and len(onset_frames) > 3:
                suspicion += 0.25
            # Silencios de duración idéntica
            if silence_dur_std < 2.0 and len(silence_durations) > 2:
                suspicion += 0.20

            return {
                "flatness_variance":      round(flat_var, 8),
                "flatness_mean":          round(flat_mean, 4),
                "onset_irregularity":     round(onset_irregularity, 2),
                "silence_duration_std":   round(silence_dur_std, 2),
                "silence_count":          len(silence_durations),
                "hf_energy_ratio":        round(hf_ratio, 4),
                "suspicion":              round(min(1.0, suspicion), 3)
            }
        except Exception as e:
            return {"suspicion": 0.2, "error": str(e)}

    # ------------------------------------------------------------------
    # Environment & Breathing Forensics (V10.3 Upgrade)
    # ------------------------------------------------------------------
    def _analyze_environment_noise(self, y: np.ndarray, sr: int) -> Dict:
        """
        Analiza si el ruido de fondo es coherente con un ambiente físico.
        Los TTS tienen ruido inyectado (blanco/rosa) que es matemáticamente uniforme.
        """
        try:
            import librosa
            # 1. Extraer segmentos de "silencio"
            S = np.abs(librosa.stft(y))
            rms = librosa.feature.rms(S=S)[0]
            # Umbral para detectar ruido de fondo puro (silencios entre palabras)
            threshold = np.percentile(rms, 15)
            noise_frames = S[:, rms < threshold]
            
            if noise_frames.shape[1] < 5:
                # No hay suficiente silencio para evaluar ambiente
                return {"coherence": 1.0, "suspicion": 0.2, "detail": "snr_too_high"}

            # 2. Análisis de varianza espectral del ruido
            # Ruido real físico varía por acústica; ruido digital es constante.
            spectral_variance = float(np.var(np.mean(noise_frames, axis=0)))
            # Entropía espectral del ruido
            noise_flatness = float(np.mean(librosa.feature.spectral_flatness(S=noise_frames)))
            
            suspicion = 0.0
            # Si la varianza del silencio es extremadamente baja (< 1e-7), es ruido inyectado.
            if spectral_variance < 1e-7:
                suspicion += 0.45 
            elif spectral_variance < 1e-6:
                suspicion += 0.20
                
            return {
                "noise_spectral_variance": round(spectral_variance, 9),
                "noise_flatness": round(noise_flatness, 4),
                "suspicion": round(suspicion, 3)
            }
        except Exception:
            return {"suspicion": 0.0}

    def _detect_breathing_signatures(self, y: np.ndarray, sr: int) -> Dict:
        """
        Busca firmas de respiración humana (inhale/exhale) antes de hablar.
        """
        try:
            import librosa
            # Buscamos energía en la banda de 300Hz-3000Hz durante onsets iniciales
            # La respiración es un ruido de banda ancha suave.
            S = np.abs(librosa.stft(y))
            rms = librosa.feature.rms(S=S)[0]
            
            # Buscamos transiciones silencio -> voz
            threshold = np.mean(rms) * 0.2
            is_voice = rms > threshold
            
            breaths_detected = 0
            # Analizar el pre-enrollment de cada segmento de voz (200ms antes)
            for i in range(1, len(is_voice)):
                if is_voice[i] and not is_voice[i-1]:
                    # Posible respiración antes de este frame
                    pre_idx = max(0, i-6) # ~200ms antes a 30fps spectral
                    pre_segment = S[:, pre_idx:i]
                    if pre_segment.shape[1] > 2:
                        # La respiración tiene un ratio de energía HF/LF específico
                        energy_ratio = np.mean(pre_segment[20:100, :]) / (np.mean(pre_segment[:20, :]) + 1e-8)
                        if 0.1 < energy_ratio < 0.8:
                            breaths_detected += 1
            
            # Si el audio dura > 10s y no hay ni una respiración detectable
            # ElevenLabs rara vez simula la entrada de aire.
            duration = len(y) / sr
            suspicion = 0.0
            if duration > 8.0 and breaths_detected == 0:
                suspicion = 0.40
            elif duration > 15.0 and breaths_detected < 2:
                suspicion = 0.25
                
            return {
                "breaths_detected": breaths_detected,
                "breath_frequency_ratio": round(duration / (breaths_detected + 1), 2),
                "suspicion": suspicion
            }
        except Exception:
            return {"suspicion": 0.0}

    # ------------------------------------------------------------------
    # Lip Sync Detection
    # ------------------------------------------------------------------
    def _analyze_lip_sync(self, y: np.ndarray, sr: int, mouth_aperture_seq: List[float]) -> Dict:
        """
        Correlaciona envelope de audio con apertura labial (MAR: Mouth Aspect Ratio).
        
        Humanos:
          - Correlación moderada (0.4-0.75): hay habla coherente pero imperfecta
        
        Deepfake/HeyGen:
          - Correlación muy alta > 0.9 (lip sync artificial perfecto)
          - O correlación muy baja < 0.2 (audio no sincronizado)
        """
        if not mouth_aperture_seq or len(mouth_aperture_seq) < 10:
            return {"lip_sync_correlation": 0.0, "suspicion": 0.3, "available": False}

        try:
            import librosa

            # Envelope de audio (RMS)
            hop = 512
            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]

            # Resamplear RMS al número de frames del video
            n_video_frames = len(mouth_aperture_seq)
            audio_resampled = np.interp(
                np.linspace(0, len(rms)-1, n_video_frames),
                np.arange(len(rms)),
                rms
            )

            mar_array  = np.array(mouth_aperture_seq)
            audio_norm = (audio_resampled - np.mean(audio_resampled)) / (np.std(audio_resampled) + 1e-8)
            mar_norm   = (mar_array - np.mean(mar_array)) / (np.std(mar_array) + 1e-8)

            # Correlación cruzada con margen de lag amplio (±10 frames)
            # Tolera videos reales desincronizados artificialmente por compresión/streaming
            correlations = []
            lag_range = 10
            for lag in range(-lag_range, lag_range + 1):
                if lag == 0:
                    corr = float(np.corrcoef(audio_norm, mar_norm)[0, 1])
                else:
                    a = audio_norm[max(0,-lag):len(audio_norm)-max(0,lag)] if lag > 0 else audio_norm[max(0,lag):len(audio_norm)+min(0,lag)]
                    m = mar_norm[max(0,lag):len(mar_norm)-max(0,lag)] if lag > 0 else mar_norm[max(0,-lag):]
                    min_len = min(len(a), len(m))
                    if min_len > 5:
                        corr = float(np.corrcoef(a[:min_len], m[:min_len])[0, 1])
                    else:
                        corr = 0.0
                if not np.isnan(corr):
                    correlations.append(corr)

            max_corr = float(max(correlations)) if correlations else 0.0
            best_lag = int(np.argmax(correlations)) - lag_range

            # Correlación perfecta → TTS lip sync artificial
            # Correlación cero → audio no coincide con labios
            suspicion = 0.0
            if max_corr > 0.92:
                suspicion += 0.45  # Demasiado sincronizado = artificial
            elif max_corr > 0.85:
                suspicion += 0.20
            elif max_corr < 0.20:
                suspicion += 0.35  # Desincronización total
            elif max_corr < 0.30:
                suspicion += 0.15

            return {
                "lip_sync_correlation": round(max_corr, 3),
                "best_lag_frames":      best_lag,
                "suspicion":            round(min(1.0, suspicion), 3),
                "available":            True
            }
        except Exception as e:
            return {"lip_sync_correlation": 0.0, "suspicion": 0.3, "available": False, "error": str(e)}

    # ------------------------------------------------------------------
    # Deep-Sync: Extracción de features para sincronización Omega V5
    # ------------------------------------------------------------------
    def _extract_sync_features(self, y: np.ndarray, sr: int, fps: float, n_frames: int) -> List[Dict]:
        """
        Extrae envolventes de energía (RMS) y sibilancia (HF) para sincronización.
        """
        if not self._librosa_available or y is None:
            return [{"rms": 0, "hf": 0}] * n_frames

        try:
            import librosa
            # Tamaño de ventana para coincidir con FPS del video
            win_size = int(sr / fps)
            
            # RMS (Energía global - vocales)
            rms = librosa.feature.rms(y=y, frame_length=win_size, hop_length=win_size)[0]
            
            # High-Frequency Energy (Sibilantes 's', 'f', 't')
            # Usamos un filtro simple o la parte alta del espectro
            spec = np.abs(librosa.stft(y, n_fft=win_size, hop_length=win_size))
            freqs = librosa.fft_frequencies(sr=sr, n_fft=win_size)
            hf_mask = freqs > 4500
            hf_energy = np.mean(spec[hf_mask, :], axis=0) if np.any(hf_mask) else np.zeros_like(rms)
            
            # Normalización suave
            rms_norm = rms / (np.max(rms) + 1e-8)
            hf_norm  = hf_energy / (np.max(hf_energy) + 1e-8)
            
            res = []
            for i in range(n_frames):
                idx = min(i, len(rms_norm) - 1)
                res.append({
                    "rms": float(rms_norm[idx]),
                    "hf":  float(hf_norm[idx])
                })
            return res
        except Exception:
            return [{"rms": 0, "hf": 0}] * n_frames

    # ------------------------------------------------------------------
    # Método principal
    # ------------------------------------------------------------------
    def analyze(self, video_path: str, mouth_aperture_seq: Optional[List[float]] = None, fps: float = 30.0, n_video_frames: int = 0) -> Dict:
        """
        Análisis de audio completo.
        
        Args:
            video_path: Ruta al archivo de video
            mouth_aperture_seq: Secuencia de MAR (Mouth Aspect Ratio) por frame
            fps: Frames per second del video
            n_video_frames: Número total de frames del video
        
        Returns:
            Dict con análisis prosódico, naturalness y lip sync
        """
        y, sr = self._load_audio(video_path)

        if y is None:
             return {"suspicion": 0.4, "available": False, "error": "no audio found"}
        
        if len(y) < sr * 0.5: # Check for short audio after initial None check
            return {
                "suspicion": 0.2,
                "available": False,
                "detail": "sin_audio_o_muy_corto"
            }

        # ── Análisis SOTA (PROD-2026 / V10.3) ──
        # Ejecutamos esto SIEMPRE, incluso si no hay voz humana clara.
        try:
            # [OPT V10.3] Pasar path directamente para evitar cargar video gigante en RAM 
            # y aplicar límite de duración interna para prevenir timeouts.
            sota_res = analyze_audio(video_path, duration=60.0)
        except Exception as e:
            print(f">>> [AudioAnalyzer] Error en motor SOTA: {e}")
            sota_res = {"probabilidad": 0, "detalles": {"metricas_avanzadas": {}}}

        # Ejecutar análisis prosódico/forense base
        if self._parselmouth_available:
            prosody_result = self._analyze_prosody_parselmouth(y, sr)
        else:
            prosody_result = self._analyze_prosody_librosa(y, sr)

        naturalness_result = self._analyze_naturalness_markers(y, sr)
        environment_result = self._analyze_environment_noise(y, sr)
        breathing_result   = self._detect_breathing_signatures(y, sr)

        lip_sync_result = {"suspicion": 0.2, "available": False}
        if mouth_aperture_seq:
            lip_sync_result = self._analyze_lip_sync(y, sr, mouth_aperture_seq)
            
        # Si no hay voz humana (prosody_result unavailable), 
        # pero detectamos SynthID, el módulo debe estar disponible.
        synth_score_str = sota_res.get("detalles", {}).get("metricas_avanzadas", {}).get("synthid_watermark", "0.0%")
        synth_val = float(synth_score_str.replace("%", ""))
        has_synth = synth_val > 50

        if prosody_result.get("available", True) is False and not has_synth:
            return {
                "suspicion": 0.5,
                "available": False,
                "detail": "ausencia_de_voz_valida_y_sin_synthid"
            }

        # Aggregación
        sub_suspicions = {
            "prosody":    prosody_result.get("suspicion", 0.5),
            "naturalness": naturalness_result.get("suspicion", 0.5),
            "lip_sync":   lip_sync_result.get("suspicion", 0.5),
            "sota_synth": sota_res.get("probabilidad", 0) / 100.0
        }

        weights = {
            "prosody": 0.05, 
            "naturalness": 0.05,
            "environment": 0.15,  # Nueva señal de peso relevante
            "breathing": 0.15,    # Nueva señal de peso relevante
            "lip_sync": 0.10,
            "sota_synth": 0.50 
        }
        
        # Consolidar sospechas
        sub_suspicions["environment"] = environment_result.get("suspicion", 0.0)
        sub_suspicions["breathing"] = breathing_result.get("suspicion", 0.0)
        
        total = sum(sub_suspicions[k] * weights[k] for k in sub_suspicions)

        # Hard Override: si SOTA detectó SynthID con alta confianza, forzar sospecha muy alta
        if synth_val > 70:
            total = max(total, 0.98)

        return {
            "suspicion":            round(min(1.0, total), 3),
            "suspicion_components": sub_suspicions,
            "prosody":              prosody_result,
            "naturalness":          naturalness_result,
            "environment":          environment_result,
            "breathing":            breathing_result,
            "lip_sync":             lip_sync_result,
            "sota_info":            sota_res,
            "audio_duration_s":     round(len(y) / sr, 1),
            "sample_rate":          sr,
            "available":            True
        }
