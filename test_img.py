import sys
import os

sys.path.append(os.path.abspath("."))

from backend.models.image_detector import get_engine
from PIL import Image
import io
import cv2
import numpy as np

def debug_image(image_path):
    print(f"\n--- DEBUGGING {image_path} ---")
    try:
        with open(image_path, "rb") as f:
            data = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    engine = get_engine()
    
    img_pil = Image.open(io.BytesIO(data)).convert("RGB")
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    has_real_metadata, has_ai_metadata = engine._check_exif(img_pil)
    print(f"METADATA: real={has_real_metadata}, ai={has_ai_metadata}")
    
    res_s1 = engine.detector_img1(img_pil) if engine.detector_img1 else None
    print(f"Raw Organika: {res_s1}")
    s1 = engine._get_score(engine.detector_img1, img_pil)
    print(f"Computed s1 (prob AI): {s1}")
    
    clip_score = engine._get_clip_score(img_pil)
    print(f"Computed clip_score (prob AI): {clip_score}")
    
    f_metrics = engine._analyze_forensics(img_cv, gray)
    print("Forensics Metrics:")
    for k, v in f_metrics.items():
        print(f"  {k}: {v}")
        
    result = engine.process_image(data)
    print(f"FINAL RESULT: probability={result['probabilidad']}, reasons={result['nota']}")

if __name__ == '__main__':
    debug_image(r"c:\Users\JOEL\OneDrive\Desktop\nueroscan\backend\data\foto real 2.jpg")
