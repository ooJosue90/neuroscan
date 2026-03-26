import os
import subprocess

videos = [
    "backend/data/video ia.mp4",
    "backend/data/video real 2.mp4",
    "backend/data/video real 3.mp4",
    "backend/data/video real.mp4"
]

print("=== RESULTADOS GLOBALES DE CALIBRACIÓN V4.1 ===")
for v in videos:
    print(f"\nProcesando: {v}")
    try:
        # Run python process and capture output as utf-8 (force ignore errors)
        result = subprocess.run(
            ["python", "backend/models/video_detector.py", v],
            capture_output=True,
            timeout=30,  # 30 seconds max!
            text=True,
            encoding="utf-8",
            errors="ignore"
        )
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
        
    except subprocess.TimeoutExpired:
        print(f"  -> ERROR: Timeout! Video inválido o demasiado grande.")
    except Exception as e:
        print(f"  -> ERROR proc: {e}")
