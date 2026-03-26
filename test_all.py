import os
import subprocess

videos = [
    "backend/data/video ia.mp4",
    "backend/data/video real 2.mp4",
    "backend/data/video real 3.mp4",
    "backend/data/video real 4.mp4",
    "backend/data/video real 5.mp4",
    "backend/data/video real 6.mp4",
    "backend/data/video real.mp4"
]

print("=== REPORTE FINAL DE CALIBRACIÓN V4.1 ===")
for v in videos:
    print(f"\nProcesando: {v}")
    try:
        # Run python process and capture output as utf-8
        result = subprocess.run(["python", "backend/models/video_detector.py", v], capture_output=True, text=True, encoding="cp1252")
        lines = result.stdout.splitlines()
        
        veredicto = "UNKNOWN"
        prob = "UNKNOWN"
        for line in lines:
            if "Veredicto:" in line:
                veredicto = line.strip()
            if "Probabilidad IA:" in line:
                prob = line.strip()
                
        print(f"  -> {veredicto}")
        print(f"  -> {prob}")
        
    except Exception as e:
        print(f"Error procesando {v}: {e}")
