import sys
sys.path.append('c:\\Users\\JOEL\\OneDrive\\Desktop\\nueroscan\\backend\\models')
import audio_detector
import os

# Busquemos un archivo de audio real si hay alguno
audio_files = []
for root, _, files in os.walk('c:\\Users\\JOEL\\OneDrive\\Desktop\\nueroscan'):
    for f in files:
        if f.endswith('.wav') or f.endswith('.mp3'):
            audio_files.append(os.path.join(root, f))
            break
    if audio_files: break

if audio_files:
    print("Test con:", audio_files[0])
    with open(audio_files[0], 'rb') as f:
        data = f.read()
    print(audio_detector.analyze_audio(data))
else:
    print("Generando audio mudo para test...")
    import numpy as np
    import io, wave
    file = io.BytesIO()
    with wave.open(file, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes((np.random.randn(16000*3)*1000).astype(np.int16).tobytes())
    file.seek(0)
    data = file.read()
    print("Audio generado. Analizando...")
    print(audio_detector.analyze_audio(data))
