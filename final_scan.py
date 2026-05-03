
import os
import torch
from PIL import Image
from backend.models.modules.ai_image_detector import AIImageDetector, get_processor
import logging

logging.basicConfig(level=logging.ERROR)

def run_final_scan():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*60)
    print(" INFORME FINAL DE ESCANEO - TALOS AI DETECTOR ".center(60, "="))
    print("="*60 + "\n")

    # 1. Cargar modelo entrenado
    model = AIImageDetector(freeze_backbones=False).to(device)
    weights_path = "backend/models/weights/detector_ia_v1.pth"
    
    if not os.path.exists(weights_path):
        print(f"Error: No se encontró el archivo de pesos en {weights_path}")
        return

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    processor = get_processor()

    # 2. Escanear carpeta data
    data_dir = "backend/data"
    valid_extensions = ('.png', '.jpg', '.jpeg', '.webp')
    
    results = []
    
    files = [f for f in os.listdir(data_dir) if f.lower().endswith(valid_extensions)]
    
    print(f"{'ARCHIVO':<35} | {'REAL':<8} | {'PREDICCIÓN':<10} | {'CONFIDENCIA'}")
    print("-" * 75)

    with torch.no_grad():
        for file in files:
            img_path = os.path.join(data_dir, file)
            
            # Etiqueta real basada en nombre
            real_label = "IA" if 'ia' in file.lower() else "REAL"
            
            try:
                image = Image.open(img_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt")["pixel_values"].to(device)
                
                # Inferencia con nuestro nuevo sistema calibrado
                output = model.predict_proba(inputs)
                prob = output["prob_ai"].item()
                pred_label = "IA" if prob > 0.5 else "REAL"
                confidence = prob if prob > 0.5 else (1 - prob)
                
                results.append((file, real_label, pred_label, confidence))
                
                # Formateado de color (simple texto)
                status = "CORRECTO" if real_label == pred_label else "ERROR"
                print(f"{file[:34]:<35} | {real_label:<8} | {pred_label:<10} | {confidence*100:6.1f}% [{status}]")
                
            except Exception as e:
                continue

    # Resumen
    correct = sum(1 for r in results if r[1] == r[2])
    total = len(results)
    accuracy = (correct / total * 100) if total > 0 else 0
    
    print("\n" + "="*60)
    print(f"RESUMEN: {correct}/{total} ACACIERTOS | PRECISIÓN REAL: {accuracy:.1f}%")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_final_scan()
