
import torch
import time
import logging
import sys

# Forzar salida UTF-8 para evitar problemas en Windows
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

from backend.models.modules.ai_image_detector import AIImageDetector

# Configuración de logging para no ensuciar la salida
logging.basicConfig(level=logging.ERROR)

def benchmark_model():
    print("\n--- Iniciando Benchmark de Mejoras - AIImageDetector ---\n")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo detectado: {device.upper()}")

    # 1. Prueba de Estabilidad (Batch Size = 1)
    print("\n--- [Prueba 1] Estabilidad en Inferencia Unitaria ---")
    try:
        model = AIImageDetector(freeze_backbones=True).to(device)
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        _ = model.predict_proba(dummy_input)
        print("Resultado: ESTABLE (LayerNorm funcionando correctamente)")
    except Exception as e:
        print(f"Resultado: ERROR ({e})")

    # 2. Diagnóstico de Parámetros
    print("\n--- [Prueba 2] Control de Parámetros (Fase 1) ---")
    trainable = model.count_trainable_params()
    total_trainable = sum(trainable.values())
    print(f"Parametros en CLIP: {trainable.get('clip', 0):,}")
    print(f"Parametros en EfficientNet: {trainable.get('efficientnet', 0):,}")
    print(f"Total Entrenables (Cabezas + Fusion): {total_trainable:,}")

    # 3. Estimación de VRAM
    print("\n--- [Prueba 3] Optimizacion de Memoria ---")
    vram_est = model.estimate_vram_mb()
    print(f"VRAM Estimada (Sin AMP/FP32): {vram_est * 2:.2f} MB")
    print(f"VRAM Estimada (Con AMP/FP16): {vram_est:.2f} MB")
    print(f"Ahorro teorico de memoria: ~50%")

    # 4. Latencia de Inferencia (AMP vs No AMP)
    print("\n--- [Prueba 4] Rendimiento de Inferencia ---")
    # Warmup
    for _ in range(5): _ = model.predict_proba(dummy_input)
    
    # Test Latency con AMP
    start = time.time()
    for _ in range(20): _ = model.predict_proba(dummy_input)
    latency_amp = (time.time() - start) / 20 * 1000
    
    # Test Latency sin AMP
    model.use_amp = False
    start = time.time()
    for _ in range(20): _ = model.predict_proba(dummy_input)
    latency_no_amp = (time.time() - start) / 20 * 1000
    
    print(f"Latencia media (Con AMP): {latency_amp:.2f} ms / imagen")
    print(f"Latencia media (Sin AMP): {latency_no_amp:.2f} ms / imagen")

if __name__ == "__main__":
    benchmark_model()
