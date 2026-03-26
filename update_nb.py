import nbformat

notebook_path = r'c:\Users\JOEL\OneDrive\Desktop\nueroscan\prueba 2.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

markdown_cell = nbformat.v4.new_markdown_cell("### Comparativa Completa\nTabla comparativa de los modelos de audio y el sistema híbrido, mostrando los porcentajes de probabilidad de que el audio sea IA.")

code = """import os
import time
import librosa
import numpy as np
import torch
import pandas as pd
from transformers import pipeline
from IPython.display import display

AUDIO_DIR = r"C:\\Users\\JOEL\\OneDrive\\Desktop\\nueroscan\\backend\\data"
audios = [
    "audio 1.mp3",
    "audio 2.mp3",
    "audio 3.mp3",
    "audio ia 1.mp3"
]

model_urls = {
    "MelodyMachine": "MelodyMachine/Deepfake-audio-detection-V2",
    "Hemgg Base": "Hemgg/Deepfake-audio-detection"
}

pipelines = {}
device = 0 if torch.cuda.is_available() else -1

print("Cargando Modelos...")
for name, repo in model_urls.items():
    try:
        pipelines[name] = pipeline("audio-classification", model=repo, device=device)
    except Exception as e:
        print(e)
print("Modelos cargados.")

def hybrid_score(audio_path, base_score):
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    f0 = librosa.yin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr)
    f0_valid = f0[f0 > 0]
    pitch_std = float(np.std(f0_valid)) if len(f0_valid) > 1 else 0.0
    
    ajuste = 0
    if zcr > 0.08: ajuste -= 50
    elif zcr > 0.04: ajuste -= 30
    if pitch_std < 12.0 and zcr < 0.04: ajuste += 10
    
    return max(0, min(100, base_score + ajuste)), zcr, pitch_std

results = []

for audio in audios:
    path = os.path.join(AUDIO_DIR, audio)
    if not os.path.exists(path):
        continue
    
    print(f"Procesando {audio}...")
    row = {"Archivo": audio}
    
    # MelodyMachine
    res1 = pipelines["MelodyMachine"](path)
    ia_score_1 = next((r['score']*100 for r in res1 if r['label'] in ['AIVoice', 'fake', 'Fake', 'spoof', 'AI']), 0)
    if ia_score_1 == 0 and len(res1) > 0:
         human_score = next((r['score']*100 for r in res1 if r['label'] in ['human', 'Real', 'real', 'bonafide']), 0)
         if human_score > 0: ia_score_1 = 100 - human_score
    row["MelodyMachine % IA"] = f"{ia_score_1:.2f}%"
    
    # Hemgg Base
    res2 = pipelines["Hemgg Base"](path)
    ia_score_base = next((r['score']*100 for r in res2 if r['label'] == 'AIVoice'), 0)
    row["Hemgg Base % IA"] = f"{ia_score_base:.2f}%"
    
    # Hybrid
    hyb_score, zcr, pitch_std = hybrid_score(path, ia_score_base)
    row["Híbrido % IA"] = f"{hyb_score:.2f}%"
    row["Ruido (ZCR)"] = round(zcr, 4)
    row["Pitch Std"] = round(pitch_std, 2)
    
    results.append(row)

df = pd.DataFrame(results)
display(df)"""

code_cell = nbformat.v4.new_code_cell(code)

nb.cells.extend([markdown_cell, code_cell])

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Notebook actualizado con la celda comparativa.")
