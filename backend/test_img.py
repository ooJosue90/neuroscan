import sys
import os
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.image_detector import analyze_image

def test_image(image_path):
    print(f"Testing image: {image_path}")
    try:
        with open(image_path, "rb") as f:
            data = f.read()
        
        res = analyze_image(data)
        print(json.dumps(res, indent=2))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    img_path = r"c:\Users\JOEL\OneDrive\Desktop\nueroscan\backend\data\foto ia 4.png"
    test_image(img_path)
