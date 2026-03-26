import sys
sys.path.append('c:\\Users\\JOEL\\OneDrive\\Desktop\\nueroscan\\backend')
from models.video_detector import detector
import json

with open('c:\\Users\\JOEL\\OneDrive\\Desktop\\nueroscan\\backend\\data\\video ia.mp4', 'rb') as f:
    data = f.read()

res = detector.analyze_comprehensive(data)
print(json.dumps(res, indent=2))
