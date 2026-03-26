import nbformat

notebook_path = r'c:\Users\JOEL\OneDrive\Desktop\nueroscan\prueba 2.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

code = """import librosa
import numpy as np
import torch
import warnings
from transformers import pipeline

warnings.filterwarnings('ignore')

# 1. Configuración
MODEL_NAME = "Hemgg/Deepfake-audio-detection"
pipe = pipeline("audio-classification", model=MODEL_NAME, device=0 if torch.cuda.is_available() else -1)

# 2. Lógica del Motor Forense (ACTUALIZADA con código de audio_detector.py)
def probar_hibrido(audio_path):
    print(f"\\n{'='*50}\\nAnalizando archivo: {audio_path}")
    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
    except Exception as e:
        print(f"Error cargando audio: {e}")
        return
        
    # Inferencia de IA
    raw_results = pipe({"array": y, "sampling_rate": sr})
    prob_ia_base = next(r['score'] for r in raw_results if r['label'] == 'AIVoice') * 100
    
    # Análisis Acústico (Extraído del nuevo motor)
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    
    # Harmonic Ratio
    y_harmonic, _ = librosa.effects.hpss(y)
    harmonic_energy = float(np.mean(y_harmonic ** 2))
    total_energy = float(np.mean(y ** 2)) + 1e-10
    harmonic_ratio = harmonic_energy / total_energy
    
    # Pitch Std
    try:
        f0 = librosa.yin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr)
        valid_f0 = f0[f0 > 0]
        pitch_std = float(np.std(valid_f0)) if len(valid_f0) > 1 else 0.0
    except:
        pitch_std = 0.0
        
    # Silence and MFCC
    digital_silence = bool(np.any(y == 0.0))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_variance = float(np.mean(np.var(mfcc, axis=1)))

    PITCH_STD_HUMAN_MIN = 12.0
    MFCC_LOW_VARIANCE = 600.0
    
    ajuste = 0
    razones = []
    
    # --- SECCIÓN A: PENALIZACIONES (Factores humanos) ---
    if zcr > 0.08 and harmonic_ratio > 0.8:
        ajuste -= 15
        razones.append("- Ruido ambiental real con voz estructurada (ZCR alto + ratio armónico alto)")
    elif zcr > 0.04 and harmonic_ratio > 0.8:
        ajuste -= 10
        razones.append("- Fondo ruidoso con voz estructurada")

    if pitch_std > PITCH_STD_HUMAN_MIN * 2.0 and prob_ia_base > 40 and harmonic_ratio > 0.85:
        ajuste -= 5
        razones.append("- Variación tonal orgánica alta")

    # --- SECCIÓN B: BONIFICACIONES (Factores IA) ---
    if pitch_std < PITCH_STD_HUMAN_MIN:
        ajuste += 30
        razones.append("+ Monotonía de pitch sospechosa (tono muy robótico)")

    if digital_silence:
        ajuste += 20
        razones.append("+ Detección de silencios digitales puros (IA)")

    if mfcc_variance < MFCC_LOW_VARIANCE:
        ajuste += 20
        razones.append("+ Firma espectral artificialmente estable")
    
    prob_final = max(0, min(100, prob_ia_base + ajuste))
    
    print(f"Prob IA (Base Modelo): {prob_ia_base:.2f}%")
    if razones:
         print("Ajustes Forenses Detectados:")
         for r in razones: print(f"  {r}")
    else:
         print("Ajuste Forense Aplicado: 0%")
         
    print(f"PROBABILIDAD FINAL: {prob_final:.2f}%")
    print(f"Métricas => ZCR: {zcr:.4f} | PitchStd: {pitch_std:.2f} | HarmonicRatio: {harmonic_ratio:.2f} | MFCC_Var: {mfcc_variance:.2f}")

# 3. Ejecutar Pruebas
audios_test = [
    "C:/Users/JOEL/OneDrive/Desktop/nueroscan/backend/data/audio 1.mp3", # Real 
    "C:/Users/JOEL/OneDrive/Desktop/nueroscan/backend/data/audio ia 1.mp3", # Falso IA
    "C:/Users/JOEL/OneDrive/Desktop/nueroscan/backend/data/audio 3.mp3", # Real
]
for a in audios_test:
    probar_hibrido(a)
"""

# Reemplazar la última celda de código (la que tiene el motor viejo) con el nuevo código
nb.cells[-1].source = code

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Notebook actualizado con éxito.")
