import os
import sys

from backend.models.video_detector import VideoIADetectorV4

def print_diagnostics():
    detector = VideoIADetectorV4()
    
    videos = [
        "backend/data/video ia.mp4",
        "backend/data/video real 6.mp4"
    ]
    
    print("=== DIAGNÓSTICO FINAL DE ESTABILIDAD ===")
    for v in videos:
        print(f"\nProcesando: {v} ...")
        try:
            with open(v, "rb") as f:
                data = f.read()
            
            res = detector.analyze(data)
            print(f"  Veredicto:       {res.get('verdict')}")
            print(f"  Probabilidad IA: {res.get('probabilidad')}%")
            
            # Sub-scores
            m_scores = res.get("module_scores", {})
            print(f"  [Scores] ViT: {m_scores.get('vit_ensemble')}% | Temporal: {m_scores.get('temporal')}% | Forensic: {m_scores.get('forensic')}%")
            
            # Reporte detallado
            rep = res.get("forensic_report", {})
            print(f"  [Metrics] PRNU Corr: {rep.get('prnu_correlation')} | Flow Div: {rep.get('flow_divergence')} | ViT P90: {rep.get('vit_frame_p90')}")
        except Exception as e:
            print(f"  X ERROR: {e}")

if __name__ == "__main__":
    print_diagnostics()
