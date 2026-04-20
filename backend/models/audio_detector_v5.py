"""
audio_detector_v5.py — DEPRECATED (alias a V6)
================================================
Este módulo fue reemplazado por el motor V6-PROD-2026.
Este archivo existe solo para compatibilidad en scripts legacy (verify_v5.py).

Para el sistema en producción, ver: audio_detector.py (V6-PROD-2026)
"""
from models.audio_detector import (
    analyze_audio as analyze_audio_v5,
    get_audio_pipelines,
    AcousticFeatures,
    AnalysisResult,
)

__all__ = ["analyze_audio_v5", "get_audio_pipelines", "AcousticFeatures", "AnalysisResult"]
