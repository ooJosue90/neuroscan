"""
Deepfake Audio Detection — Sistema Híbrido Avanzado V4
=======================================================
Mejoras sobre V3:
  • Sistema anti-falsos-positivos: requiere evidencia corroborada múltiple
  • Clasificación previa del tipo de audio (voz / música / ambiente)
  • Puntuación basada en evidencias independientes (evidence-gating)
  • Detección de codecs tradicionales (MP3/AAC) vs vocoders neuronales
  • Cobertura de motores: ElevenLabs, PlayHT, Bark, Coqui/VITS,
    Tortoise-TTS, RVC, Microsoft Azure TTS, Google TTS, Murf, Resemble
  • Análisis de prosodia y ritmo silábico (distingue lectura TTS vs habla espontánea)
  • Seguimiento de formantes F1/F2 (detección de interpolación artificial)
  • Intervalos de confianza: se reporta incertidumbre, no solo binario
  • Recalibración completa de umbrales y pesos (reducción de sesgo hacia IA)
  • Score final balanceado: sin corrección unless ≥3 señales independientes
"""

import os
import io
import logging
import tempfile
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import librosa
import scipy.signal as signal
from transformers import pipeline

# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("deepfake_audio_v4")

# ─────────────────────────────────────────────────────────────────────────────
# Constantes del modelo
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME         = "Hemgg/Deepfake-audio-detection"
SAMPLE_RATE        = 16_000
SILENCE_TOP_DB     = 30
MIN_AUDIO_DURATION = 0.5   # subido a 0.5s — muestras más cortas son poco confiables

# ─────────────────────────────────────────────────────────────────────────────
# Umbrales V2 — revisados para reducir falsos positivos
# ─────────────────────────────────────────────────────────────────────────────
ZCR_HIGH_NOISE          = 0.12   # subido (micro de ambiente tiene ZCR alto)
ZCR_MED_NOISE           = 0.06
MFCC_LOW_VARIANCE       = 400.0  # bajado — el umbral anterior era demasiado agresivo
SPECTRAL_FLUX_THRESH    = 0.010
PITCH_STD_HUMAN_MIN     = 10.0
FLATNESS_IA_LIMIT       = 0.06
PHASE_DISSONANCE_THRESH = 0.40   # subido — evita FP en audio comprimido

# ─────────────────────────────────────────────────────────────────────────────
# Umbrales V3 — recalibrados
# ─────────────────────────────────────────────────────────────────────────────
JITTER_IA_LIMIT       = 0.002  # bajado: < 0.2% → periodo casi perfecto (TTS)
SHIMMER_IA_LIMIT      = 0.020  # bajado: < 2% → amplitud glótica sintética
BREATH_RATE_MIN       = 0.06   # bajado ligeramente
COART_SMOOTH_THRESH   = 0.50   # bajado — coarticulación humana varía mucho
CODEC_ARTIFACT_THRESH = 0.20
AMR_IA_THRESH         = 0.15
UV_TRANSITION_THRESH  = 0.025
SUBBAND_RATIO_THRESH  = 4.5    # subido — MP3/AAC también desbalancea sub-bandas

# ─────────────────────────────────────────────────────────────────────────────
# Umbrales V4 — nuevas métricas
# ─────────────────────────────────────────────────────────────────────────────
# Ritmo silábico: TTS produce isocronia (sílabas de duración uniforme)
SYLLABLE_ISOCHRONY_THRESH = 0.15  # Desviación estándar normalizada del ritmo

# Formantes: en TTS los formantes se interpolan linealmente entre fonemas
FORMANT_LINEARITY_THRESH  = 0.72  # r² de regresión lineal sobre trayectoria F1

# Pausas: TTS inserta micro-pausas regulares y predecibles
PAUSE_REGULARITY_THRESH   = 0.20  # Std normalizada de duración de pausas

# Relación señal/artefacto (distingue MP3 de vocoder neuronal)
CODEC_TYPE_FREQ_CUTOFF    = 6_500  # Hz — MP3/AAC filtra a partir de aquí
NEURAL_ARTIFACT_BAND_LOW  = 7_000  # Hz — ringing de HiFi-GAN/DAC
NEURAL_ARTIFACT_BAND_HIGH = 7_800  # Hz

# Mínimo de evidencias independientes requeridas para penalizar
EVIDENCE_GATE = 2   # al menos 2 señales de IA para aplicar corrección

# ─────────────────────────────────────────────────────────────────────────────
# Pesos recalibrados — mucho más conservadores
# ─────────────────────────────────────────────────────────────────────────────
PENALTY_STRONG       = 12
PENALTY_MODERATE     = 8
PENALTY_SLIGHT       = 4

BONUS_PHASE_ATTACK   = 22   # antes 40 — demasiado agresivo
BONUS_JITTER_SHIMMER = 16   # antes 25
BONUS_CODEC_ART      = 14   # antes 22
BONUS_BREATH_ABSENT  = 12   # antes 18
BONUS_COART          = 10   # antes 15
BONUS_SUSPICIOUS     = 12   # antes 20
BONUS_STRONG         = 18   # antes 30
BONUS_AMR            = 8    # antes 12
BONUS_PROSODY        = 14   # nuevo: ritmo silábico isócrono
BONUS_FORMANT_LINEAR = 12   # nuevo: formantes lineales
BONUS_PAUSE_REGULAR  = 8    # nuevo: pausas demasiado regulares

# ─────────────────────────────────────────────────────────────────────────────
# Singleton del modelo
# ─────────────────────────────────────────────────────────────────────────────
_audio_pipe: Optional[pipeline] = None


def get_audio_pipeline() -> pipeline:
    """Singleton de la pipeline. Detecta CUDA > MPS (Apple) > CPU."""
    global _audio_pipe
    if _audio_pipe is None:
        if torch.cuda.is_available():
            device: int | str = 0
            accel = "CUDA"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"   # Apple Silicon M1/M2/M3
            accel  = "MPS (Apple Silicon)"
        else:
            device = -1
            accel  = "CPU"
        logger.info("Cargando modelo '%s' en %s …", MODEL_NAME, accel)
        _audio_pipe = pipeline("audio-classification", model=MODEL_NAME, device=device)
        logger.info("Modelo cargado en %s.", accel)
    return _audio_pipe


# ─────────────────────────────────────────────────────────────────────────────
# Contenedor centralizado de matrices costosas
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class BaseFeatures:
    """
    Matrices pesadas calculadas una sola vez y compartidas por todos los
    extractores. Incluye el tipo de audio inferido para condicionar el análisis.
    """
    y:          np.ndarray
    sr:         int
    n_fft:      int
    hop_length: int
    stft_mag:   np.ndarray   # (F, T)
    stft_phase: np.ndarray   # (F, T)
    freqs:      np.ndarray   # (F,)
    rms:        np.ndarray   # (T,)  — 1-D
    f0:         np.ndarray   # (T,)  — 1-D
    centroid:   np.ndarray   # (T,)  — 1-D
    audio_type: str          # "speech" | "music" | "ambient"


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses de resultado
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class AcousticFeatures:
    """V2 + V3 + V4 features."""
    # ── V2 ──
    zcr:               float = 0.0
    mfcc_variance:     float = 0.0
    spectral_flux:     float = 0.0
    spectral_flatness: float = 0.0
    spectral_centroid: float = 0.0
    pitch_std:         float = 0.0
    rms_energy:        float = 0.0
    harmonic_ratio:    float = 0.0
    phase_dissonance:  float = 0.0
    digital_silence:   bool  = False
    duration_seconds:  float = 0.0
    # ── V3 ──
    jitter:                  float = 0.0
    shimmer:                 float = 0.0
    breath_energy_ratio:     float = 0.0
    coarticulation_score:    float = 0.0
    codec_artifact_score:    float = 0.0
    amr_regularity:          float = 0.0
    uv_transition_sharpness: float = 0.0
    subband_imbalance:       float = 0.0
    elevenlabs_score:        float = 0.0
    # ── V4 ──
    syllable_isochrony:  float = 0.0          # Uniformidad del ritmo silábico
    formant_linearity:   float = 0.0          # Linealidad de trayectorias F1
    pause_regularity:    float = 0.0          # Regularidad de micro-pausas
    codec_type:          str   = "unknown"    # "mp3_aac" | "neural_vocoder" | "clean"
    speech_confidence:   float = 0.0          # Probabilidad de que el audio sea habla
    evidence_count:      int   = 0            # Señales de IA independientes detectadas
    confidence_interval: tuple = field(default_factory=lambda: (0.0, 0.0))
    engine_scores:       dict  = field(default_factory=dict)  # Scores por motor TTS


@dataclass
class AnalysisResult:
    status:              str   = "success"
    probabilidad:        int   = 0
    nota:                str   = ""
    tipo:                str   = "audio"
    audio_type:          str   = "speech"
    correccion_aplicada: bool  = False
    prob_modelo_base:    int   = 0
    penalizacion_total:  int   = 0
    features:            AcousticFeatures = field(default_factory=AcousticFeatures)
    raw_predictions:     dict  = field(default_factory=dict)
    error_detail:        Optional[str] = None
    motor_detectado:     str   = "Desconocido"
    confianza_analisis:  str   = "normal"   # "alta" | "normal" | "baja"

    def to_dict(self) -> dict:
        ci_low, ci_high = self.features.confidence_interval
        base = {
            "status":          self.status,
            "probabilidad":    self.probabilidad,
            "confianza_rango": f"{ci_low}%–{ci_high}%",
            "confianza_nivel": self.confianza_analisis,
            "nota":            self.nota,
            "tipo":            self.tipo,
            "audio_tipo":      self.audio_type,
            "motor_detectado": self.motor_detectado,
            "detalles": {
                "modelo":            f"Híbrido Avanzado V4 ({MODEL_NAME})",
                "prob_modelo_bruta": f"{self.prob_modelo_base}%",
                "ajuste_acustico": (
                    f"{'-' if self.penalizacion_total < 0 else '+'}"
                    f"{abs(self.penalizacion_total)}%"
                ),
                "evidencias_ia":       self.features.evidence_count,
                "gate_minimo":         EVIDENCE_GATE,
                "correccion_aplicada": self.correccion_aplicada,
                "predicciones": {
                    "IA":     f"{self.probabilidad}%",
                    "Humano": f"{100 - self.probabilidad}%",
                },
                "scores_por_motor": self.features.engine_scores,
                "features_v2": {
                    "zcr":              f"{self.features.zcr:.4f}",
                    "mfcc_varianza":    f"{self.features.mfcc_variance:.2f}",
                    "spectral_flux":    f"{self.features.spectral_flux:.4f}",
                    "pitch_std_hz":     f"{self.features.pitch_std:.2f}",
                    "disonancia_fase":  f"{self.features.phase_dissonance:.4f}",
                    "silencio_digital": "Sí" if self.features.digital_silence else "No",
                    "duracion_s":       f"{self.features.duration_seconds:.2f}",
                },
                "features_v3": {
                    "jitter":               f"{self.features.jitter:.5f}",
                    "shimmer":              f"{self.features.shimmer:.5f}",
                    "breath_energy_ratio":  f"{self.features.breath_energy_ratio:.4f}",
                    "coarticulation_score": f"{self.features.coarticulation_score:.4f}",
                    "codec_artifact_score": f"{self.features.codec_artifact_score:.4f}",
                    "amr_regularity":       f"{self.features.amr_regularity:.4f}",
                    "uv_transition_sharp":  f"{self.features.uv_transition_sharpness:.4f}",
                    "subband_imbalance":    f"{self.features.subband_imbalance:.2f}",
                    "elevenlabs_score":     f"{self.features.elevenlabs_score:.1f}",
                },
                "features_v4": {
                    "syllable_isochrony": f"{self.features.syllable_isochrony:.4f}",
                    "formant_linearity":  f"{self.features.formant_linearity:.4f}",
                    "pause_regularity":   f"{self.features.pause_regularity:.4f}",
                    "codec_type":         self.features.codec_type,
                    "speech_confidence":  f"{self.features.speech_confidence:.3f}",
                },
                "predicciones_crudas": self.raw_predictions,
            },
        }
        if self.error_detail:
            base["error_detail"] = self.error_detail  # type: ignore[assignment]
        return base


# ═══════════════════════════════════════════════════════════════════════════════
# CLASIFICADOR DE TIPO DE AUDIO — NUEVO EN V4
# Evita analizar música o audio ambiental como si fuera voz sintetizada.
# ═══════════════════════════════════════════════════════════════════════════════

def classify_audio_type(
    y: np.ndarray, sr: int, stft_mag: np.ndarray
) -> tuple[str, float]:
    """
    Clasifica el audio en 'speech', 'music' o 'ambient' antes del análisis
    forense. Si no se detecta habla humana, se omite toda la corrección
    acústica.

    Usa 7 señales independientes para ser robusta ante voz cantada y música
    con letra (el caso más difícil de distinguir de habla):

    1. voiced_ratio    — fracción de frames con F0 en rango de habla (80–350 Hz)
    2. centroid_mean   — centroide espectral medio
    3. zcr_mean        — tasa de cruces por cero
    4. percussive_ratio— energía percusiva relativa (HPSS) → música tiene más
    5. rolloff_mean    — rolloff espectral al 85% → música usa rango más amplio
    6. bandwidth_mean  — anchura espectral → música es más ancha
    7. rms_cv          — coef. de variación del RMS → música tiene más dinámica
    """
    try:
        # ── Feature 1: Voiced ratio ───────────────────────────────────────
        f0 = librosa.yin(y, fmin=60.0, fmax=400.0, sr=sr)
        voiced = f0[(f0 > 80) & (f0 < 350)]
        voiced_ratio = len(voiced) / max(len(f0), 1)

        # ── Feature 2 & 3: Centroide y ZCR ────────────────────────────────
        centroid_mean = float(np.mean(
            librosa.feature.spectral_centroid(S=stft_mag, sr=sr)[0]
        ))
        zcr_mean = float(np.mean(librosa.feature.zero_crossing_rate(y)))

        # ── Feature 4: Ratio percusivo (HPSS) ──────────────────────────────
        # Música tiene dominante percusión (batería, ritmo); habla prácticamente no.
        _, y_perc = librosa.effects.hpss(y)
        total_e = float(np.mean(y ** 2)) + 1e-10
        percussive_ratio = float(np.mean(y_perc ** 2)) / total_e

        # ── Feature 5 & 6: Rolloff y Bandwidth ────────────────────────────
        # Música usa todo el espectro; habla se concentra en 200–4000 Hz.
        rolloff_mean = float(np.mean(
            librosa.feature.spectral_rolloff(S=stft_mag, sr=sr, roll_percent=0.85)[0]
        ))
        bandwidth_mean = float(np.mean(
            librosa.feature.spectral_bandwidth(S=stft_mag, sr=sr)[0]
        ))

        # ── Feature 7: Dinámica RMS ────────────────────────────────────
        # Música tiene variaciones de volumen más amplias y suaves.
        rms = librosa.feature.rms(S=stft_mag)[0]
        rms_cv = float(np.std(rms) / (np.mean(rms) + 1e-10))

        # ── Score de voz (0–1) ────────────────────────────────────────
        speech_score = 0.0

        # Indicaçiones de VOZ
        speech_score += min(voiced_ratio * 2.5, 0.40)   # máximo 0.40
        if 300 < centroid_mean < 3500:
            speech_score += 0.15
        if 0.01 < zcr_mean < 0.10:
            speech_score += 0.10
        if bandwidth_mean < 2500:            # anchura espectral estrecha → voz
            speech_score += 0.10

        # Indicaciones de MÚS ICA (restan del score de voz)
        music_score = 0.0
        if percussive_ratio > 0.12:          # percusión significativa
            music_score += 0.30
        if rolloff_mean > 5_000:             # usa frecuencias altas
            music_score += 0.20
        if bandwidth_mean > 3_000:           # espectro ancho
            music_score += 0.20
        if rms_cv > 0.55:                    # dinámica amplia y fluida
            music_score += 0.15
        if voiced_ratio > 0.70 and percussive_ratio > 0.08:
            # Voz cantada: mucho F0 Y percusión notable → música con letra
            music_score += 0.25

        music_score = float(np.clip(music_score, 0.0, 1.0))
        speech_score = float(np.clip(speech_score - music_score * 0.5, 0.0, 1.0))

        logger.debug(
            "classify_audio_type: speech=%.2f music=%.2f "
            "voiced_r=%.2f perc=%.2f rolloff=%.0f bw=%.0f rms_cv=%.2f",
            speech_score, music_score, voiced_ratio,
            percussive_ratio, rolloff_mean, bandwidth_mean, rms_cv,
        )

        # ── Decisión final ──────────────────────────────────────────
        if music_score >= 0.45:              # suficientes señales de música
            return "music", speech_score
        if speech_score >= 0.40:            # umbral bajado de 0.45 a 0.40
            return "speech", speech_score
        # Si ni música ni voz clara → ambiente
        return "ambient", speech_score

    except Exception as exc:
        logger.warning("classify_audio_type error: %s — fallback speech", exc)
        return "speech", 0.5   # fallback conservador


# ═══════════════════════════════════════════════════════════════════════════════
# EXTRACCIÓN V2 — sin cambios de lógica, usa BaseFeatures
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_v2_features(base: BaseFeatures, feat: AcousticFeatures) -> None:
    feat.duration_seconds = len(base.y) / base.sr
    feat.zcr = float(np.mean(librosa.feature.zero_crossing_rate(base.y)))

    mfcc = librosa.feature.mfcc(
        S=librosa.power_to_db(base.stft_mag ** 2), sr=base.sr, n_mfcc=13
    )
    feat.mfcc_variance = float(np.mean(np.var(mfcc, axis=1)))

    phase_diff = np.diff(np.unwrap(base.stft_phase, axis=0), axis=1)
    feat.phase_dissonance = float(np.std(phase_diff))

    flux = np.mean(np.sqrt(np.sum(np.diff(base.stft_mag, axis=1) ** 2, axis=0)))
    feat.spectral_flux = float(flux)

    feat.spectral_flatness = float(
        np.mean(librosa.feature.spectral_flatness(S=base.stft_mag))
    )
    feat.spectral_centroid = float(np.mean(base.centroid))

    valid_f0 = base.f0[(base.f0 > 50) & (base.f0 < 2000)]
    if len(valid_f0) > 1:
        q1, q3 = np.percentile(valid_f0, [25, 75])
        iqr = q3 - q1
        clean_f0 = valid_f0[
            (valid_f0 >= q1 - 1.5 * iqr) & (valid_f0 <= q3 + 1.5 * iqr)
        ]
        feat.pitch_std = float(np.std(clean_f0)) if len(clean_f0) > 0 else 0.0
    else:
        feat.pitch_std = 0.0

    feat.rms_energy    = float(np.mean(base.rms))
    feat.digital_silence = bool(np.any(base.y == 0.0))

    y_harmonic, _ = librosa.effects.hpss(base.y)
    feat.harmonic_ratio = float(np.mean(y_harmonic ** 2)) / (
        float(np.mean(base.y ** 2)) + 1e-10
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MÉTRICAS V3 — recalibradas
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_jitter_shimmer(base: BaseFeatures) -> tuple[float, float]:
    try:
        valid = base.f0[(base.f0 > 60) & (base.f0 < 400)]
        if len(valid) < 20:   # muestra mínima subida
            return 0.0, 0.0
        periods = 1.0 / valid
        jitter = float(
            np.mean(np.abs(np.diff(periods))) / (np.mean(periods) + 1e-10)
        )
        valid_rms = base.rms[base.rms > 1e-5]
        if len(valid_rms) < 2:
            return jitter, 0.0
        shimmer = float(
            np.mean(np.abs(np.diff(valid_rms))) / (np.mean(valid_rms) + 1e-10)
        )
        return jitter, shimmer
    except Exception:
        return 0.0, 0.0


def _compute_breath_energy(base: BaseFeatures) -> float:
    """
    Fracción de energía en la banda 100–400 Hz durante pausas.
    Los TTS omiten o simulan mal las respiraciones entre palabras.

    Usa ``librosa.util.frame`` (stride tricks) para crear vistas sin copia
    de memoria (zero-copy). Evita el np.stack que duplicaba el audio en RAM.
    """
    try:
        y, sr = base.y, base.sr
        sos = signal.butter(4, [100, 400], btype="band", fs=sr, output="sos")
        y_breath: np.ndarray = np.asarray(signal.sosfilt(sos, y), dtype=np.float32)
        y_f32:    np.ndarray = np.asarray(y, dtype=np.float32)

        frame_len: int = int(sr * 0.025)
        hop_len:   int = frame_len // 2

        # librosa.util.frame crea vistas strided sin copiar datos
        # shape: (frame_len, n_frames)
        frames_v  = librosa.util.frame(y_f32,   frame_length=frame_len, hop_length=hop_len)
        breaths_v = librosa.util.frame(y_breath, frame_length=frame_len, hop_length=hop_len)

        if frames_v.shape[1] == 0:
            return 0.0

        # Energía media por frame — axis=0 porque la forma es (frame_len, n_frames)
        e_frame:  np.ndarray = np.mean(frames_v  ** 2, axis=0)
        e_breath: np.ndarray = np.mean(breaths_v ** 2, axis=0)

        p20 = np.percentile(e_frame, 20)
        p60 = np.percentile(e_frame, 60)
        mask: np.ndarray = (e_frame > p20) & (e_frame < p60)
        total_e: float = float(np.sum(e_frame[mask]))
        if total_e < 1e-10:
            return 0.0
        return float(np.sum(e_breath[mask])) / total_e
    except Exception:
        return 0.0


def _compute_coarticulation(base: BaseFeatures) -> float:
    try:
        if len(base.centroid) < 3:
            return 0.0
        c_norm = (base.centroid - np.mean(base.centroid)) / (
            np.std(base.centroid) + 1e-7
        )
        accel = np.diff(c_norm, n=2)
        return float(np.clip(float(np.var(accel)) / 0.5, 0, 1))
    except Exception:
        return 0.0


def _compute_codec_artifacts(base: BaseFeatures) -> tuple[float, str]:
    """
    Diferencia entre compresión tradicional (MP3/AAC → corte de frecuencias)
    y artefactos de vocoder neuronal (HiFi-GAN/DAC → ringing en 7–7.8 kHz).

    Retorna (score, tipo_codec).
    """
    try:
        nyquist = base.sr / 2.0

        # ── Detección de corte MP3/AAC ──────────────────────────────────────
        cutoff_mask = base.freqs > CODEC_TYPE_FREQ_CUTOFF
        full_mask   = base.freqs > 200
        if not np.any(cutoff_mask) or not np.any(full_mask):
            return 0.0, "clean"

        e_above_cutoff = float(np.mean(base.stft_mag[cutoff_mask, :] ** 2))
        e_full         = float(np.mean(base.stft_mag[full_mask,   :] ** 2)) + 1e-10

        if e_above_cutoff / e_full < 0.01 and nyquist > CODEC_TYPE_FREQ_CUTOFF:
            return 0.0, "mp3_aac"   # compresión normal, no marca como IA

        # ── Detección de ringing neuronal (banda 7–7.8 kHz) ────────────────
        if nyquist < NEURAL_ARTIFACT_BAND_LOW:
            return 0.0, "clean"

        mask_neural = (
            (base.freqs >= NEURAL_ARTIFACT_BAND_LOW) &
            (base.freqs <= NEURAL_ARTIFACT_BAND_HIGH)
        )
        mask_mid = (base.freqs >= 1_000) & (base.freqs <= 5_000)

        if not np.any(mask_neural) or not np.any(mask_mid):
            return 0.0, "clean"

        e_neural = float(np.mean(base.stft_mag[mask_neural, :] ** 2))
        e_mid    = float(np.mean(base.stft_mag[mask_mid,    :] ** 2)) + 1e-10

        score = float(np.clip((e_neural / e_mid) / CODEC_ARTIFACT_THRESH, 0, 1))
        return (score, "neural_vocoder") if score > 0.5 else (score, "clean")

    except Exception:
        return 0.0, "unknown"


def _compute_amr(base: BaseFeatures) -> float:
    try:
        if len(base.rms) < 16:
            return 0.0
        env_fft  = np.abs(np.fft.rfft(base.rms - np.mean(base.rms)))
        env_fft /= (np.sum(env_fft) + 1e-10)
        return float(np.clip(float(np.max(env_fft[1:])) / 0.3, 0, 1))
    except Exception:
        return 0.0


def _compute_uv_transition_sharpness(base: BaseFeatures) -> float:
    try:
        threshold   = float(np.percentile(base.rms, 30))
        voiced      = (base.rms > threshold).astype(np.float32)
        transitions = np.where(np.diff(voiced) != 0)[0]
        if len(transitions) < 2:
            return 0.0
        sharpness_vals = []
        for t in transitions:
            start  = max(0, t - 3)
            end    = min(len(base.rms) - 1, t + 3)
            window = base.rms[start:end + 1]
            if len(window) > 1:
                sharpness_vals.append(float(np.max(np.abs(np.diff(window)))))
        return float(np.mean(sharpness_vals)) if sharpness_vals else 0.0
    except Exception:
        return 0.0


def _compute_subband_imbalance(base: BaseFeatures) -> float:
    try:
        stft_power = base.stft_mag ** 2
        total      = float(np.sum(stft_power)) + 1e-10
        bands      = np.array_split(np.arange(len(base.freqs)), 4)
        band_energies = [
            float(np.sum(stft_power[idx, :])) / total
            for idx in bands if len(idx) > 0
        ]
        if len(band_energies) < 2:
            return 0.0
        expected  = 1.0 / len(band_energies)
        deviation = float(np.mean([(e - expected) ** 2 for e in band_energies]))
        return float(deviation * 100)
    except Exception:
        return 0.0


def _compute_elevenlabs_signature(feat: AcousticFeatures) -> float:
    score = 0.0
    if feat.jitter < JITTER_IA_LIMIT:
        score += 35 * (1.0 - feat.jitter / JITTER_IA_LIMIT)
    if feat.shimmer < SHIMMER_IA_LIMIT:
        score += 25 * (1.0 - feat.shimmer / SHIMMER_IA_LIMIT)
    if feat.breath_energy_ratio < BREATH_RATE_MIN:
        score += 20 * (1.0 - feat.breath_energy_ratio / BREATH_RATE_MIN)
    score += 15 * min(feat.codec_artifact_score, 1.0)
    if feat.amr_regularity > AMR_IA_THRESH:
        score += 5 * min(feat.amr_regularity / AMR_IA_THRESH, 1.0)
    return float(np.clip(score, 0, 100))


# ═══════════════════════════════════════════════════════════════════════════════
# MÉTRICAS V4 — nuevas
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_syllable_isochrony(base: BaseFeatures) -> float:
    """
    Detecta si el ritmo silábico es demasiado uniforme (isócrono), rasgo
    característico de TTS que lee texto sin variación prosódica natural.

    Coeficiente de variación bajo de intervalos inter-silábicos → TTS.
    """
    try:
        env = base.rms
        if len(env) < 30:
            return 0.0

        smoothed  = np.convolve(env, np.ones(5) / 5, mode="same")
        threshold = np.percentile(smoothed, 55)
        above     = (smoothed > threshold).astype(int)
        onsets    = np.where(np.diff(above) == 1)[0]

        if len(onsets) < 4:
            return 0.0

        intervals = np.diff(onsets).astype(float)
        cv = float(np.std(intervals) / (np.mean(intervals) + 1e-6))
        # CV humano típico: 0.3–0.9; TTS: < 0.15
        return float(np.clip(1.0 - cv / 0.5, 0, 1))
    except Exception:
        return 0.0


def _compute_formant_linearity(base: BaseFeatures) -> float:
    """
    Estima si las trayectorias de formantes son lineales (interpolación TTS).
    Proxy: r² medio de regresión lineal sobre el centroide en ventanas de 20 frames.
    Un r² alto indica transiciones demasiado suaves → TTS neuronal.

    Implementación 100% vectorizada (sin bucle for ni np.polyfit):
    usa álgebra de matrices para resolver OLS simultáneamente sobre todas
    las ventanas. Reducción de tiempo τ ∧3× respecto al bucle original.
    """
    try:
        c = base.centroid
        if len(c) < 40:
            return 0.0

        window: int = 20
        hop:    int = window // 2
        n_wins: int = (len(c) - window) // hop
        if n_wins < 1:
            return 0.0

        # Construir matriz de segmentos (n_wins, window) via stride tricks
        starts  = np.arange(n_wins) * hop
        idx_row = starts[:, None] + np.arange(window)[None, :]   # (n_wins, window)
        segs    = c[idx_row]   # (n_wins, window) — vista sin copia

        # Regresión lineal OLS vectorizada: y = m*x + b
        # x es siempre [0, 1, ..., window-1] — igual para todas las ventanas
        x   = np.arange(window, dtype=np.float64)
        xm  = x - x.mean()             # x centrado: (window,)
        ym  = segs - segs.mean(axis=1, keepdims=True)   # (n_wins, window)

        # slope m = sum(xm * ym, axis=1) / sum(xm^2)
        m   = (ym @ xm) / (np.dot(xm, xm) + 1e-10)    # (n_wins,)
        # fit = m[:, None] * xm[None, :] (broadcast)
        fit = m[:, None] * xm[None, :]                  # (n_wins, window)

        ss_res = np.sum((ym - fit) ** 2, axis=1)        # (n_wins,)
        ss_tot = np.sum(ym ** 2, axis=1) + 1e-10        # (n_wins,)
        r2     = 1.0 - ss_res / ss_tot                  # (n_wins,)

        return float(np.clip(np.mean(r2), 0, 1))
    except Exception:
        return 0.0


def _compute_pause_regularity(base: BaseFeatures) -> float:
    """
    Los TTS insertan micro-pausas de duración muy uniforme.
    La voz humana tiene pausas irregulares (hesitaciones, respiración, duda).

    Coeficiente de variación invertido: 1 = muy regular = TTS.
    """
    try:
        threshold = float(np.percentile(base.rms, 15))
        silent    = (base.rms < threshold).astype(int)
        changes   = np.diff(silent)

        starts = np.where(changes == 1)[0]
        ends   = np.where(changes == -1)[0]

        if len(starts) == 0 or len(ends) == 0:
            return 0.0
        if ends[0] < starts[0]:
            ends = ends[1:]
        n = min(len(starts), len(ends))
        starts, ends = starts[:n], ends[:n]
        if n < 3:
            return 0.0

        durations = (ends - starts).astype(float)
        cv = float(np.std(durations) / (np.mean(durations) + 1e-6))
        # CV humano: > 0.6; TTS: < 0.2
        return float(np.clip(1.0 - cv / 0.7, 0, 1))
    except Exception:
        return 0.0


def _compute_engine_scores(feat: AcousticFeatures) -> dict:
    """
    Calcula scores 0–100 para cada motor TTS conocido.
    """
    scores: dict = {}

    # ── ElevenLabs / PlayHT / Murf ──────────────────────────────────────────
    el = feat.elevenlabs_score
    if feat.syllable_isochrony > 0.6:
        el = min(el + 10, 100)
    scores["ElevenLabs/PlayHT/Murf"] = float(round(el, 1))

    # ── Bark (Suno) ──────────────────────────────────────────────────────────
    bark = 0.0
    if feat.codec_type == "neural_vocoder":
        bark += 40
    if feat.subband_imbalance > 3.0:
        bark += 25
    if feat.amr_regularity > 0.18:
        bark += 20
    if feat.digital_silence:
        bark += 15
    scores["Bark/Suno"] = float(round(min(bark, 100), 1))

    # ── Coqui / VITS / Piper ─────────────────────────────────────────────────
    vits = 0.0
    if feat.formant_linearity > FORMANT_LINEARITY_THRESH:
        vits += 35
    if feat.pause_regularity > PAUSE_REGULARITY_THRESH * 1.5:
        vits += 30
    if feat.breath_energy_ratio < BREATH_RATE_MIN * 0.3:
        vits += 20
    if feat.mfcc_variance < MFCC_LOW_VARIANCE * 0.8:
        vits += 15
    scores["Coqui/VITS/Piper"] = float(round(min(vits, 100), 1))

    # ── Tortoise-TTS / Vall-E ────────────────────────────────────────────────
    tortoise = 0.0
    if feat.jitter < JITTER_IA_LIMIT and feat.shimmer < SHIMMER_IA_LIMIT:
        tortoise += 30
    if feat.syllable_isochrony > 0.5:
        tortoise += 25
    if feat.codec_artifact_score > 0.4:
        tortoise += 25
    if feat.phase_dissonance > PHASE_DISSONANCE_THRESH * 0.8:
        tortoise += 20
    scores["Tortoise-TTS/Vall-E"] = float(round(min(tortoise, 100), 1))

    # ── RVC (Real-Time Voice Cloning) ────────────────────────────────────────
    rvc = 0.0
    if feat.uv_transition_sharpness > UV_TRANSITION_THRESH:
        rvc += 35
    if feat.pause_regularity > PAUSE_REGULARITY_THRESH:
        rvc += 25
    if feat.codec_type == "neural_vocoder":
        rvc += 25
    if feat.jitter > JITTER_IA_LIMIT * 1.5:   # RVC hereda algo del jitter original
        rvc += 15
    scores["RVC/Voice-Cloning"] = float(round(min(rvc, 100), 1))

    # ── Microsoft Azure TTS / Google Cloud TTS ───────────────────────────────
    corp = 0.0
    if feat.breath_energy_ratio < BREATH_RATE_MIN * 0.2:
        corp += 30
    if feat.syllable_isochrony > 0.65:
        corp += 30
    if feat.pitch_std < PITCH_STD_HUMAN_MIN * 0.7:
        corp += 25
    if feat.digital_silence:
        corp += 15
    scores["Azure TTS/Google TTS"] = float(round(min(corp, 100), 1))

    return scores


# ═══════════════════════════════════════════════════════════════════════════════
# EXTRACTOR PRINCIPAL V4
# ═══════════════════════════════════════════════════════════════════════════════

def extract_acoustic_features(y: np.ndarray, sr: int) -> AcousticFeatures:
    """
    Normaliza la señal, construye BaseFeatures una sola vez y delega en los
    extractores V2, V3 y V4.
    """
    feat = AcousticFeatures()

    # ── Normalización de pico ────────────────────────────────────────────────
    max_amp = np.max(np.abs(y))
    if max_amp > 0:
        y = y / max_amp

    # ── Cálculos pesados centralizados ──────────────────────────────────────
    n_fft      = 2048
    hop_length = 256

    stft_complex = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    stft_mag     = np.abs(stft_complex)
    stft_phase   = np.angle(stft_complex)
    freqs        = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    rms = librosa.feature.rms(
        S=stft_mag, frame_length=n_fft, hop_length=hop_length
    )[0]

    f0 = librosa.yin(
        y, fmin=60.0, fmax=2100.0, sr=sr,
        frame_length=n_fft, hop_length=hop_length
    )

    centroid = librosa.feature.spectral_centroid(
        S=stft_mag, sr=sr, n_fft=n_fft, hop_length=hop_length
    )[0]

    audio_type, speech_conf = classify_audio_type(y, sr, stft_mag)
    feat.speech_confidence  = speech_conf

    base = BaseFeatures(
        y, sr, n_fft, hop_length,
        stft_mag, stft_phase, freqs,
        rms, f0, centroid,
        audio_type,
    )

    # ── Extracción V2 + V3 ───────────────────────────────────────────────────
    _extract_v2_features(base, feat)

    feat.jitter, feat.shimmer    = _compute_jitter_shimmer(base)
    feat.breath_energy_ratio     = _compute_breath_energy(base)
    feat.coarticulation_score    = _compute_coarticulation(base)

    codec_score, codec_type      = _compute_codec_artifacts(base)
    feat.codec_artifact_score    = codec_score
    feat.codec_type              = codec_type

    feat.amr_regularity          = _compute_amr(base)
    feat.uv_transition_sharpness = _compute_uv_transition_sharpness(base)
    feat.subband_imbalance       = _compute_subband_imbalance(base)
    feat.elevenlabs_score        = _compute_elevenlabs_signature(feat)

    # ── Extracción V4 ────────────────────────────────────────────────────────
    feat.syllable_isochrony = _compute_syllable_isochrony(base)
    feat.formant_linearity  = _compute_formant_linearity(base)
    feat.pause_regularity   = _compute_pause_regularity(base)
    feat.engine_scores      = _compute_engine_scores(feat)

    return feat


# ═══════════════════════════════════════════════════════════════════════════════
# CORRECCIÓN HÍBRIDA V4 — Evidence-Gated
# ═══════════════════════════════════════════════════════════════════════════════

def compute_hybrid_correction(
    prob_ia_base: float,
    feat: AcousticFeatures,
) -> tuple[int, bool, str, int]:
    """
    Sistema de puntuación basado en evidencias independientes.

    Cada señal positiva de IA incrementa ``evidence_count``.
    La corrección solo se aplica si evidence_count >= EVIDENCE_GATE.
    Esto evita falsos positivos por una única métrica anómala.

    Retorna (ajuste_puntos, se_aplicó, motor_inferido, evidence_count).
    """
    ajuste   = 0
    evidence = 0
    razones: list[str] = []

    # ── PENALIZACIONES — indicios sólidos de humanidad ──────────────────────

    if feat.zcr > ZCR_HIGH_NOISE and feat.harmonic_ratio > 0.8:
        ajuste -= PENALTY_STRONG
        razones.append("Ruido ambiental real con estructura vocal orgánica.")
    elif feat.zcr > ZCR_MED_NOISE and feat.harmonic_ratio > 0.8:
        ajuste -= PENALTY_MODERATE
        razones.append("Fondo de micrófono humano.")

    if (feat.pitch_std > PITCH_STD_HUMAN_MIN * 2.5
            and prob_ia_base > 40
            and feat.harmonic_ratio > 0.85):
        ajuste -= PENALTY_SLIGHT
        razones.append("Dinamismo tonal biológico alto.")

    if feat.breath_energy_ratio > BREATH_RATE_MIN * 1.5:
        ajuste -= PENALTY_MODERATE
        razones.append(
            f"Respiraciones naturales (ratio={feat.breath_energy_ratio:.3f})."
        )

    if feat.coarticulation_score > 0.80:
        ajuste -= PENALTY_SLIGHT
        razones.append(
            f"Coarticulación rica y variada ({feat.coarticulation_score:.2f})."
        )

    if feat.syllable_isochrony < 0.15:
        ajuste -= PENALTY_SLIGHT
        razones.append(
            f"Ritmo silábico irregular (isocronia={feat.syllable_isochrony:.2f})."
        )

    # ── EVIDENCIAS IA ────────────────────────────────────────────────────────

    # 1. Fase — artefacto masivo de vocoder
    if feat.phase_dissonance > PHASE_DISSONANCE_THRESH:
        ajuste   += BONUS_PHASE_ATTACK
        evidence += 1
        razones.append(
            f"Artefactos masivos en matriz de fase "
            f"(disonancia={feat.phase_dissonance:.4f})."
        )

    # 2. Pitch plano
    if feat.pitch_std < PITCH_STD_HUMAN_MIN:
        ajuste   += BONUS_STRONG
        evidence += 1
        razones.append(
            f"Monotonía de afinación robótica (std={feat.pitch_std:.2f} Hz)."
        )

    # 3. Silencios digitales puros
    if feat.digital_silence:
        ajuste   += BONUS_SUSPICIOUS // 2
        evidence += 1
        razones.append("Silencios digitales puros.")

    # 4. MFCC plano
    if feat.mfcc_variance < MFCC_LOW_VARIANCE:
        ajuste   += BONUS_SUSPICIOUS // 2
        evidence += 1
        razones.append(
            f"Firma espectral MFCC artificialmente plana "
            f"(var={feat.mfcc_variance:.1f})."
        )

    # 5. Jitter / Shimmer robóticos
    if feat.jitter < JITTER_IA_LIMIT and feat.shimmer < SHIMMER_IA_LIMIT:
        ajuste   += BONUS_JITTER_SHIMMER
        evidence += 1
        razones.append(
            f"Jitter={feat.jitter:.4f} y Shimmer={feat.shimmer:.4f}: "
            "voz sintéticamente perfecta."
        )
    elif feat.jitter < JITTER_IA_LIMIT:
        ajuste   += BONUS_JITTER_SHIMMER // 2
        evidence += 1
        razones.append(f"Jitter robótico ({feat.jitter:.4f}).")
    elif feat.shimmer < SHIMMER_IA_LIMIT:
        ajuste   += BONUS_JITTER_SHIMMER // 2
        evidence += 1
        razones.append(f"Shimmer robótico ({feat.shimmer:.4f}).")

    # 6. Sin respiraciones
    if feat.breath_energy_ratio < BREATH_RATE_MIN * 0.4:
        ajuste   += BONUS_BREATH_ABSENT
        evidence += 1
        razones.append(
            f"Sin respiraciones detectadas (ratio={feat.breath_energy_ratio:.4f})."
        )

    # 7. Artefactos de vocoder neuronal (no penalizar MP3/AAC)
    if feat.codec_type == "neural_vocoder":
        if feat.codec_artifact_score > 0.6:
            ajuste   += BONUS_CODEC_ART
            evidence += 1
            razones.append(
                f"Ringing neuronal HiFi-GAN/DAC >7 kHz "
                f"(score={feat.codec_artifact_score:.2f})."
            )
        elif feat.codec_artifact_score > 0.35:
            ajuste   += BONUS_CODEC_ART // 2
            evidence += 1
            razones.append(
                f"Artefactos de vocoder neuronal moderados "
                f"(score={feat.codec_artifact_score:.2f})."
            )
    elif feat.codec_type == "mp3_aac":
        razones.append(
            "Compresión MP3/AAC detectada — artefactos de alta frecuencia "
            "descartados del análisis de IA."
        )

    # 8. Coarticulación demasiado suave
    if feat.coarticulation_score < COART_SMOOTH_THRESH:
        ajuste   += BONUS_COART
        evidence += 1
        razones.append(
            f"Transiciones formánticas interpoladas "
            f"(coart={feat.coarticulation_score:.3f})."
        )

    # 9. AMR periódica
    if feat.amr_regularity > AMR_IA_THRESH * 1.5:
        ajuste   += BONUS_AMR
        evidence += 1
        razones.append(
            f"Envolvente de amplitud periódica (AMR={feat.amr_regularity:.3f})."
        )

    # 10. Transiciones V/UV abruptas
    if feat.uv_transition_sharpness > UV_TRANSITION_THRESH:
        ajuste   += BONUS_COART // 2
        evidence += 1
        razones.append(
            f"Transiciones voz/silencio bruscas como TTS "
            f"(sharp={feat.uv_transition_sharpness:.4f})."
        )

    # 11. Sub-band imbalance (EnCodec)
    if feat.subband_imbalance > SUBBAND_RATIO_THRESH:
        ajuste   += BONUS_SUSPICIOUS // 3
        evidence += 1
        razones.append(
            f"Desequilibrio de sub-bandas EnCodec "
            f"(imb={feat.subband_imbalance:.2f})."
        )

    # 12. Isocronia silábica
    if feat.syllable_isochrony > SYLLABLE_ISOCHRONY_THRESH * 2:
        ajuste   += BONUS_PROSODY
        evidence += 1
        razones.append(
            f"Ritmo silábico isócrono (isocronia={feat.syllable_isochrony:.3f})."
        )
    elif feat.syllable_isochrony > SYLLABLE_ISOCHRONY_THRESH:
        ajuste   += BONUS_PROSODY // 2
        evidence += 1
        razones.append(
            f"Ritmo silábico algo uniforme "
            f"(isocronia={feat.syllable_isochrony:.3f})."
        )

    # 13. Formantes lineales
    if feat.formant_linearity > FORMANT_LINEARITY_THRESH:
        ajuste   += BONUS_FORMANT_LINEAR
        evidence += 1
        razones.append(
            f"Trayectorias de formantes lineales "
            f"(r²={feat.formant_linearity:.3f}) → interpolación TTS."
        )

    # 14. Pausas regulares
    if feat.pause_regularity > PAUSE_REGULARITY_THRESH * 2:
        ajuste   += BONUS_PAUSE_REGULAR
        evidence += 1
        razones.append(
            f"Micro-pausas demasiado regulares "
            f"(reg={feat.pause_regularity:.3f})."
        )

    # 15. ElevenLabs score compuesto
    if feat.elevenlabs_score > 75:
        ajuste   += 12
        evidence += 1
        razones.append(
            f"Firma ElevenLabs/PlayHT compuesta muy alta "
            f"(score={feat.elevenlabs_score:.1f})."
        )
    elif feat.elevenlabs_score > 55:
        ajuste += 6
        razones.append(
            f"Firma ElevenLabs/PlayHT moderada "
            f"(score={feat.elevenlabs_score:.1f})."
        )

    logger.debug(
        "Evidencias IA: %d | Ajuste: %+d | %s",
        evidence, ajuste, " | ".join(razones),
    )

    # ── Gate de evidencias ───────────────────────────────────────────────────
    correccion_aplicada = evidence >= EVIDENCE_GATE
    if not correccion_aplicada:
        ajuste = max(ajuste, 0)   # solo dejamos penalizaciones (humano)

    # ── Inferencia del motor predominante ────────────────────────────────────
    engine_scores: dict[str, float] = feat.engine_scores
    best_engine   = max(engine_scores, key=engine_scores.get) if engine_scores else None  # type: ignore[arg-type]
    best_score: float    = engine_scores.get(best_engine, 0.0) if best_engine else 0.0  # type: ignore[arg-type]

    motor = "Desconocido"
    if evidence < EVIDENCE_GATE:
        motor = "Sin firma clara de IA"
    elif best_score >= 55 and best_engine:
        confianza = "alta" if best_score >= 75 else "moderada"
        motor = f"{best_engine} (confianza {confianza})"
    elif (feat.phase_dissonance > PHASE_DISSONANCE_THRESH
          and feat.codec_type == "neural_vocoder"):
        motor = "Motor Vocoder Neuronal (HiFi-GAN / DAC)"
    elif feat.digital_silence and feat.mfcc_variance < MFCC_LOW_VARIANCE:
        motor = "TTS Concatenativo / Clonación"
    elif evidence >= EVIDENCE_GATE:
        motor = "IA — Motor no identificado"

    return ajuste, correccion_aplicada, motor, evidence


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_audio(data: bytes) -> dict:
    """
    Analiza un buffer de audio (bytes) y retorna el resultado como dict.

    Optimización V4.1: lee el audio desde ``io.BytesIO`` directamente
    en memoria, evitando escribir y leer un archivo temporal en disco.
    """
    result = AnalysisResult()

    try:
        audio_buf = io.BytesIO(data)
        y, sr = librosa.load(audio_buf, sr=SAMPLE_RATE, mono=True)

        if len(y) / sr < MIN_AUDIO_DURATION:
            result.status = "error"
            result.nota   = (
                f"Audio demasiado corto ({len(y)/sr:.2f}s). "
                f"Mínimo requerido: {MIN_AUDIO_DURATION}s."
            )
            return result.to_dict()

        y_trim, _ = librosa.effects.trim(y, top_db=SILENCE_TOP_DB)
        if len(y_trim) / sr < MIN_AUDIO_DURATION:
            logger.warning("Audio muy corto tras recorte — usando señal original.")
            y_trim = y

        # ── Pre-clasificación del tipo de audio ──────────────────────────────
        stft_quick = np.abs(librosa.stft(y_trim, n_fft=2048, hop_length=512))
        audio_type, speech_conf = classify_audio_type(y_trim, sr, stft_quick)
        result.audio_type = audio_type

        if audio_type != "speech":
            logger.info(
                "Audio clasificado como '%s' (speech_conf=%.2f) — "
                "análisis forense limitado.",
                audio_type, speech_conf,
            )

        # ── Inferencia HuggingFace ────────────────────────────────────────────
        pipe = get_audio_pipeline()
        raw_results = pipe({"array": y_trim, "sampling_rate": sr})
        result.raw_predictions = {
            r["label"]: round(r["score"] * 100, 2) for r in raw_results
        }
        prob_ia_base = result.raw_predictions.get("AIVoice", 0.0)
        result.prob_modelo_base = int(prob_ia_base)

        # ── Extracción de características V4 ─────────────────────────────────
        feat = extract_acoustic_features(y_trim, sr)
        feat.speech_confidence = speech_conf
        result.features = feat

        # ── Para audio no-voz: omitir análisis forense completamente ─────────
        # Si el clasificador detectó música o ambiente, ninguna métrica
        # acústica de voz (jitter, respiraciones, etc.) es válida.
        # Solo se reporta la predicción del modelo neural base.
        if audio_type != "speech":
            result.probabilidad        = int(prob_ia_base)
            result.penalizacion_total  = 0
            result.correccion_aplicada = False
            result.motor_detectado     = f"Análisis limitado ({audio_type})"
            result.confianza_analisis  = "baja"
            result.nota = (
                f"Audio detectado como '{audio_type}' "
                f"(speech_conf={speech_conf:.2f}). "
                "El análisis forense está diseñado exclusivamente para voz "
                "humana; se reporta solo la predicción del modelo neural base."
            )
            feat.confidence_interval = (
                max(0, int(prob_ia_base) - 25),
                min(100, int(prob_ia_base) + 25),
            )
            return result.to_dict()

        # ── Fusión híbrida V4 ─────────────────────────────────────────────────
        ajuste, correccion, motor, evidence = compute_hybrid_correction(
            prob_ia_base, feat
        )
        feat.evidence_count        = evidence
        result.penalizacion_total  = ajuste
        result.correccion_aplicada = correccion
        result.motor_detectado     = motor

        prob_final          = int(np.clip(prob_ia_base + ajuste, 0, 100))
        result.probabilidad = prob_final

        # ── Intervalo de confianza ────────────────────────────────────────────
        uncertainty = max(5, 20 - evidence * 3)
        feat.confidence_interval = (
            max(0,   prob_final - uncertainty),
            min(100, prob_final + uncertainty),
        )

        # ── Nivel de confianza ────────────────────────────────────────────────
        if evidence >= 4 or (correccion and abs(ajuste) >= 30):
            result.confianza_analisis = "alta"
        elif evidence >= EVIDENCE_GATE:
            result.confianza_analisis = "normal"
        else:
            result.confianza_analisis = "baja"

        # ── Nota final ────────────────────────────────────────────────────────
        partes: list[str] = []

        if feat.elevenlabs_score > 70:
            partes.append(
                f"⚠ Firma ElevenLabs/PlayHT detectada "
                f"({feat.elevenlabs_score:.0f}/100)"
            )

        engine_scores: dict[str, float] = feat.engine_scores
        best_engine = (
            max(engine_scores, key=engine_scores.get)  # type: ignore[arg-type]
            if engine_scores else None
        )
        best_score: float = engine_scores.get(best_engine, 0.0) if best_engine else 0.0
        if best_engine and best_score >= 60:
            partes.append(
                f"Motor más probable: {best_engine} ({best_score:.0f}/100)"
            )

        if not correccion:
            if evidence == 0:
                partes.append(
                    "Sin señales acústicas de IA — clasificación basada "
                    "solo en el modelo neural."
                )
            else:
                partes.append(
                    f"Solo {evidence} señal(es) acústica(s) de IA "
                    f"(mínimo {EVIDENCE_GATE}) — corrección no aplicada."
                )
        else:
            partes.append(
                f"{evidence} evidencias de síntesis detectadas — "
                "análisis forense biológico aplicado."
            )
            if prob_final >= 60:
                partes.append("Alta probabilidad de clonación o síntesis de voz.")

        result.nota = " | ".join(partes)

    except Exception as exc:
        logger.exception("Error inesperado al auditar audio.")
        result.status       = "error"
        result.nota         = "Error del motor de análisis espectral."
        result.error_detail = str(exc)

    return result.to_dict()