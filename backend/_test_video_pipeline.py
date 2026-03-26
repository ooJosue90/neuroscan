"""
Diagnóstico rápido del pipeline de video V4.1
Ejecutar desde la carpeta backend:
    python _test_video_pipeline.py <ruta_video.mp4>
"""
import sys
import time
import os

sys.path.insert(0, ".")

print("=" * 55)
print("DIAGNÓSTICO VideoIADetector V4.1")
print("=" * 55)

# ── 1. Verificar config ────────────────────────────────────
from models.video_detector import PipelineConfig
cfg = PipelineConfig()
cfg.USE_GPU = False   # Forzar CPU para el test
cfg.MAX_FRAMES = 20   # Menos frames para ir rápido
print(f"  USE_GPU        : {cfg.USE_GPU}")
print(f"  BATCH_SIZE     : {cfg.BATCH_SIZE}")
print(f"  TIMEOUT_TEMPORAL: {cfg.TIMEOUT_TEMPORAL}s")
print(f"  TIMEOUT_VIT    : {cfg.TIMEOUT_VIT}s")
print(f"  MAX_FRAMES     : {cfg.MAX_FRAMES}")

# ── 2. Inicializar detector ────────────────────────────────
print("\nInicializando detector...")
t0 = time.time()
from models.video_detector import VideoIADetectorV4
detector = VideoIADetectorV4(cfg)
print(f"  Inicializado en {time.time()-t0:.1f}s")

# ── 3. Verificar que RAFT NO se cargó en CPU ───────────────
raft = detector.temporal_analyzer._raft_model
print(f"\n  RAFT model     : {'CARGADO (GPU)' if raft else 'None (Farneback en CPU) ✔'}")

# ── 4. Analizar video si se pasó como argumento ────────────
if len(sys.argv) >= 2:
    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print(f"\nERROR: archivo no encontrado: {video_path}")
        sys.exit(1)

    print(f"\nAnalizando: {video_path}")
    print("-" * 55)

    t1 = time.time()
    result = detector.analyze_file(video_path)
    elapsed = time.time() - t1

    print(f"\n{'='*55}")
    print(f"  RESULTADO")
    print(f"{'='*55}")
    print(f"  Status     : {result.get('status')}")
    print(f"  Veredicto  : {result.get('verdict')}")
    print(f"  Prob IA    : {result.get('probabilidad')}%")
    print(f"  Confianza  : {result.get('confidence')}%")
    print(f"  Tiempo     : {elapsed:.1f}s")

    print(f"\n  Scores por módulo:")
    for mod, sc in result.get("module_scores", {}).items():
        bar = "█" * int(sc / 5)
        print(f"    {mod:20s}: {sc:5.1f}%  {bar}")

    print(f"\n  Razones:")
    for r in result.get("reasons", []):
        print(f"    • {r}")

    if result.get("status") == "error":
        print(f"\n  NOTA: {result.get('nota')}")
else:
    print("\n  (Pasa un video como argumento para analizarlo)")
    print("  Ejemplo: python _test_video_pipeline.py video.mp4")
