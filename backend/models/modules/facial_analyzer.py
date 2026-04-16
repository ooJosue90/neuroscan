"""
MÓDULO: Facial Biometrics Analyzer V4
Detecta anomalías faciales con 478 landmarks MediaPipe:
  - Eye Aspect Ratio (EAR) y cinemática de parpadeo
  - Microexpresiones (Action Units FACS)
  - Asimetría facial dinámica
  - rPPG (Detección de pulso cardíaco remoto)
  - Mesh Sliding (Detección de flotación de máscara)
  - Detección de piel sintética (crominancia YCbCr)
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import deque
import concurrent.futures


class FacialBiometricsAnalyzer:
    """
    Análisis facial avanzado basado en MediaPipe FaceMesh 478 puntos.
    Detecta deepfakes clásicos (FaceSwap) y avatares sintéticos (HeyGen).
    """

    # --- Índices MediaPipe FaceMesh ---
    # Ojo izquierdo (desde perspectiva del sujeto)
    LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    # Ojo derecho
    RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    # EAR específico (6 puntos por ojo)
    LEFT_EAR_PTS  = [362, 385, 387, 263, 373, 380]
    RIGHT_EAR_PTS = [33,  160, 158, 133, 153, 144]
    # Boca
    LIPS_OUTER = [61, 84, 17, 314, 405, 321, 375, 291, 61, 185, 40, 39, 37, 0, 267, 269, 270, 409]
    LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
    # Puntos de referencia para asimetría
    SYMMETRY_PAIRS = [
        (33, 263),   # Comisuras ojos
        (61, 291),   # Comisuras boca
        (234, 454),  # Mejillas
        (93, 323),   # Mandíbula
        (70, 300),   # Cejas
    ]
    # Nariz (para estabilidad de tracking)
    NOSE_TIP = 4
    # Contorno facial (jawline + frente)
    FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    
    # Puntos para rPPG (Frente)
    FOREHEAD_PATCH = [67, 109, 10, 338, 297, 332, 284, 251, 21, 54, 103]
    # Puntos estables para referencia de movimiento (Nariz)
    STABLE_POINTS = [1, 2, 4, 5, 6, 168, 197] 

    def __init__(self):
        self._face_mesh = None
        self._face_detector = None
        self._load_models()

    def _load_models(self):
        try:
            import mediapipe as mp
            self._mp = mp
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,  # [OPT] Mejor para frames muestreados (no-continuos)
                max_num_faces=1,
                refine_landmarks=True,   # 478 puntos incluyendo iris
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print(">>> [FacialAnalyzer] MediaPipe FaceMesh 478pts cargado")
        except Exception as e:
            print(f">>> [FacialAnalyzer] MediaPipe no disponible: {e}")
            self._face_mesh = None

    # ------------------------------------------------------------------
    # Eye Aspect Ratio (EAR)
    # ------------------------------------------------------------------
    @staticmethod
    def _ear(landmarks: list, pts_indices: List[int]) -> float:
        """
        Calcula Eye Aspect Ratio.
        EAR = (||p1-p5|| + ||p2-p4||) / (2 * ||p0-p3||)
        Ojo cerrado: EAR < 0.2
        """
        pts = np.array([(landmarks[i].x, landmarks[i].y) for i in pts_indices[:6]])
        A = np.linalg.norm(pts[1] - pts[5])
        B = np.linalg.norm(pts[2] - pts[4])
        C = np.linalg.norm(pts[0] - pts[3])
        return float((A + B) / (2.0 * C + 1e-8))

    def _analyze_blink_kinematics(self, ear_sequence: np.ndarray, fps: float = 30.0) -> Dict:
        """
        Analiza la cinemática del parpadeo frame a frame.
        
        Humanos:
          - 15-25 parpadeos/minuto
          - Cierre más rápido que apertura (ratio ≈ 1.5-2.5)
          - Duración total: 150-400ms
        
        Deepfakes/Avatares:
          - Parpadeos "cuadrados" (simétricos)
          - 0 parpadeos (avatares HeyGen) o parpadeos robóticos
        """
        EAR_THRESH = 0.21
        CONSEC_MIN = 2  # Mínimo frames con ojo cerrado = parpadeo válido

        blink_events = []
        below = False
        start_idx = 0

        for i, ear in enumerate(ear_sequence):
            if ear < EAR_THRESH and not below:
                below = True
                start_idx = i
            elif ear >= EAR_THRESH and below:
                below = False
                duration = i - start_idx
                if duration >= CONSEC_MIN:
                    # Encontrar mínimo (pico del parpadeo)
                    seg = ear_sequence[start_idx:i]
                    min_idx_rel = int(np.argmin(seg))
                    blink_events.append({
                        "start":    start_idx,
                        "min":      start_idx + min_idx_rel,
                        "end":      i,
                        "duration": duration,
                        "closing":  min_idx_rel,
                        "opening":  duration - min_idx_rel
                    })

        duration_s = len(ear_sequence) / fps
        blinks_per_min = len(blink_events) / max(duration_s, 1.0) * 60.0

        if not blink_events:
            # [FIX] Si el FPS es bajo (<10), es estadísticamente probable perder parpadeos. 
            # No penalizamos la ausencia en videos muestreados.
            suspicion = 0.3 if (duration_s < 4 or fps < 10) else 0.7
            return {
                "blink_count":        0,
                "blinks_per_min":     0.0,
                "asymmetry_ratio":    None,
                "duration_ms_mean":   None,
                "suspicion":          suspicion,
                "detail":             "sin_parpadeos"
            }

        closing_rates = [b["closing"] for b in blink_events]
        opening_rates = [b["opening"] for b in blink_events]
        durations_ms  = [b["duration"] / fps * 1000 for b in blink_events]

        # Ratio asimetría: apertura / cierre (humanos: 1.5-2.5)
        asymmetry_ratio = float(np.mean(opening_rates)) / (float(np.mean(closing_rates)) + 1e-6)

        # Duración media parpadeo (humanos: 150-400ms)
        mean_dur_ms = float(np.mean(durations_ms))

        suspicion = 0.0
        # Frecuencia anormal
        if blinks_per_min < 5 or blinks_per_min > 50:
            suspicion += 0.35
        # Parpadeo perfectamente simétrico (IA)
        if asymmetry_ratio < 1.2 or asymmetry_ratio > 4.0:
            suspicion += 0.30
        # Duración irrealista
        if mean_dur_ms < 60 or mean_dur_ms > 600:
            suspicion += 0.20
        # Varianza nula en duración (parpadeos robóticos)
        if len(blink_events) > 2 and np.std(durations_ms) < 10:
            suspicion += 0.20

        return {
            "blink_count":        len(blink_events),
            "blinks_per_min":     round(blinks_per_min, 2),
            "asymmetry_ratio":    round(asymmetry_ratio, 3),
            "duration_ms_mean":   round(mean_dur_ms, 1),
            "duration_ms_std":    round(float(np.std(durations_ms)), 1),
            "suspicion":          min(1.0, suspicion)
        }

    # ------------------------------------------------------------------
    # Microexpresiones
    # ------------------------------------------------------------------
    def _analyze_microexpressions(self, landmark_sequences: List[list], fps: float = 30.0) -> Dict:
        """
        Detecta presencia/ausencia de microexpresiones.
        Duración microexpresión: 40-200ms → 1-6 frames @ 30fps.
        
        HeyGen / avatares: microexpressions_density ≈ 0
        Humanos: 0.3-2.0 microexpresiones/segundo en conversación
        """
        if len(landmark_sequences) < 10:
            return {"microexp_density": 0.0, "suspicion": 0.4, "available": False}

        # Calcular delta de movimiento facial frame a frame
        # Usar puntos clave: boca, ojos, cejas, nariz
        key_points = [0, 17, 61, 291, 33, 263, 70, 300, 4, 168, 397, 172]

        au_deltas = []
        for i in range(1, len(landmark_sequences)):
            prev = landmark_sequences[i-1]
            curr = landmark_sequences[i]
            try:
                # Alineamiento básico por traslación (punta de la nariz, idx 4)
                # Mitiga falsos positivos por movimientos de cabeza globales
                pn_x, pn_y = prev[4].x, prev[4].y
                cn_x, cn_y = curr[4].x, curr[4].y

                delta = np.mean([
                    np.sqrt(
                        ((curr[j].x - cn_x) - (prev[j].x - pn_x))**2 + 
                        ((curr[j].y - cn_y) - (prev[j].y - pn_y))**2
                    )
                    for j in key_points if j < len(curr) and j < len(prev) and j != 4
                ])
                au_deltas.append(float(delta))
            except (IndexError, AttributeError):
                continue

        if len(au_deltas) < 5:
            return {"microexp_density": 0.0, "suspicion": 0.4, "available": False}

        au_array = np.array(au_deltas)
        mean_delta = float(np.mean(au_array))
        std_delta  = float(np.std(au_array))

        # Detectar picos cortos (microexpresiones)
        try:
            from scipy.signal import find_peaks
            threshold = mean_delta + 1.2 * std_delta
            peaks, props = find_peaks(
                au_array,
                height=threshold,
                width=(1, int(fps * 0.25)),  # 1 a 7 frames (40-250ms)
                prominence=std_delta * 0.5
            )
            microexp_count = len(peaks)
        except ImportError:
            # Detección manual sin scipy
            threshold = mean_delta + 1.2 * std_delta
            microexp_count = 0
            in_peak = False
            peak_len = 0
            for val in au_array:
                if val > threshold:
                    if not in_peak:
                        in_peak = True
                        peak_len = 1
                    else:
                        peak_len += 1
                else:
                    if in_peak and 1 <= peak_len <= int(fps * 0.25):
                        microexp_count += 1
                    in_peak = False
                    peak_len = 0

        duration_s = len(au_deltas) / fps
        microexp_density = microexp_count / max(duration_s, 1.0)

        # HeyGen: density ≈ 0-0.1, humano: 0.3-2.0
        suspicion = max(0.0, 1.0 - microexp_density / 0.5) if microexp_density < 0.5 else 0.0
        # Penalizar también variabilidad nula (cara completamente estática/robótica)
        if std_delta < mean_delta * 0.03 and mean_delta < 0.005:
            suspicion = min(1.0, suspicion + 0.3)

        return {
            "microexp_density":  round(microexp_density, 3),
            "microexp_count":    microexp_count,
            "au_delta_mean":     round(mean_delta, 5),
            "au_delta_std":      round(std_delta, 5),
            "duration_s":        round(duration_s, 1),
            "suspicion":         round(min(1.0, suspicion), 3),
            "available":         True
        }

    # ------------------------------------------------------------------
    # Asimetría facial dinámica
    # ------------------------------------------------------------------
    def _analyze_facial_asymmetry(self, landmark_sequences: List[list]) -> Dict:
        """
        Las caras reales tienen asimetría natural Y variable.
        Los deepfakes tienden a simetría artificial o asimetría constante.
        """
        if len(landmark_sequences) < 5:
            return {"asymmetry_mean": 0.0, "asymmetry_std": 0.0, "suspicion": 0.3}

        asymmetry_scores = []

        for lms in landmark_sequences:
            if len(lms) < 468:
                continue
            frame_asym = []
            for (li, ri) in self.SYMMETRY_PAIRS:
                lpt = np.array([lms[li].x, lms[li].y])
                rpt = np.array([lms[ri].x, lms[ri].y])
                # Distancia al eje central (nariz)
                nose = np.array([lms[self.NOSE_TIP].x, lms[self.NOSE_TIP].y])
                d_left  = np.linalg.norm(lpt - nose)
                d_right = np.linalg.norm(rpt - nose)
                asym = abs(d_left - d_right) / (max(d_left, d_right) + 1e-8)
                frame_asym.append(asym)
            asymmetry_scores.append(float(np.mean(frame_asym)))

        if not asymmetry_scores:
            return {"asymmetry_mean": 0.0, "asymmetry_std": 0.0, "suspicion": 0.3}

        asym_mean = float(np.mean(asymmetry_scores))
        asym_std  = float(np.std(asymmetry_scores))

        # Demasiado simétrico: deepfake
        suspicion = 0.0
        if asym_mean < 0.01:
            suspicion += 0.4
        # Asimetría perfectamente constante: deepfake (no varía con expresiones)
        if asym_std < 0.002 and len(asymmetry_scores) > 10:
            suspicion += 0.35
        # Asimetría extrema y constante: face-swap mal alineado
        if asym_mean > 0.15:
            suspicion += 0.25

        return {
            "asymmetry_mean": round(asym_mean, 4),
            "asymmetry_std":  round(asym_std, 4),
            "suspicion":      round(min(1.0, suspicion), 3)
        }

    # ------------------------------------------------------------------
    # Análisis de piel sintética (crominancia YCbCr)
    # ------------------------------------------------------------------
    def _analyze_skin_chrominance(self, frames_bgr: List[np.ndarray], landmark_sequences: List[list]) -> Dict:
        """
        Face-Swap clásico: la piel pegada tiene distribución de crominancia diferente.
        Canal Cb de YCbCr es el más discriminativo.
        
        Método: comparar distribución de Cb en máscara facial poligonal vs fondo cercano/cuello.
        DeepFaceLab: diferencia Cb > 8 entre cara y cuello.
        """
        cb_face_means  = []
        cb_bg_means    = []
        cb_transitions = []  # Gradiente en borde de cara

        for i, (frame, lms) in enumerate(zip(frames_bgr, landmark_sequences)):
            if not lms:
                continue
                
            h_img, w_img = frame.shape[:2]
            
            # Construir polígono de la cara (FACE_OVAL)
            face_pts = np.array([
                [int(lms[idx].x * w_img), int(lms[idx].y * h_img)]
                for idx in self.FACE_OVAL
            ], dtype=np.int32)
            
            # Máscara estricta para la piel facial
            face_mask = np.zeros((h_img, w_img), dtype=np.uint8)
            cv2.fillPoly(face_mask, [face_pts], 255)

            ycbcr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            cb = ycbcr[:, :, 2].astype(float)

            # Región facial (solo píxeles de piel pura)
            face_cb_pixels = cb[face_mask == 255]
            if len(face_cb_pixels) == 0:
                continue
            cb_face_means.append(float(np.mean(face_cb_pixels)))

            # Región de "cuello" (debajo del mentón para skin natural)
            chin_y = int(lms[152].y * h_img)
            chin_x = int(lms[152].x * w_img)
            
            neck_y1 = min(chin_y + 5, h_img)
            neck_y2 = min(chin_y + max(20, h_img // 10), h_img)
            neck_x1 = max(0, chin_x - 30)
            neck_x2 = min(w_img, chin_x + 30)
            
            neck_cb = cb[neck_y1:neck_y2, neck_x1:neck_x2]
            if neck_cb.size > 0:
                cb_bg_means.append(float(np.mean(neck_cb)))

            # Gradiente en borde inferior de cara (mentón/zona de splice)
            if chin_y + 5 < h_img:
                border_zone = cb[max(0, chin_y-3):min(h_img, chin_y+3), neck_x1:neck_x2]
                if border_zone.size > 0:
                    grad = float(np.std(np.gradient(border_zone, axis=0)))
                    cb_transitions.append(grad)

        if not cb_face_means:
            return {"cb_face_mean": 0.0, "cb_diff": 0.0, "suspicion": 0.2}

        avg_face_cb = float(np.mean(cb_face_means))
        avg_bg_cb   = float(np.mean(cb_bg_means)) if cb_bg_means else avg_face_cb
        cb_diff     = abs(avg_face_cb - avg_bg_cb)
        avg_grad    = float(np.mean(cb_transitions)) if cb_transitions else 0.0

        # Face-swap: Ahora que usamos polígonos estrictos sin pelo ni fondo,
        # la diferencia real en Cb rara vez supera 5-6 inclusive en fakes.
        # Diferencia > 5 es sospechosa (sombras fuertes), > 9 es alta certeza de splice injertado.
        suspicion = 0.0
        if cb_diff > 9.0:
            suspicion += 0.6
        elif cb_diff > 5.0:
            suspicion += 0.3
            
        # Gradiente brusco (línea dura del mentón en empalmes)
        if avg_grad > 10.0:
            suspicion += 0.3
        elif avg_grad > 6.0:
            suspicion += 0.15

        return {
            "cb_face_mean":  round(avg_face_cb, 2),
            "cb_bg_mean":    round(avg_bg_cb, 2),
            "cb_diff":       round(cb_diff, 2),
            "border_grad":   round(avg_grad, 2),
            "suspicion":     round(min(1.0, suspicion), 3)
        }

    # ------------------------------------------------------------------
    # rPPG: Detección de pulso remoto (Heartbeat) - Omega V5 
    # ------------------------------------------------------------------
    def _analyze_rppg_pulse(self, frames_bgr: List[np.ndarray], landmark_sequences: List[list], fps: float = 30.0) -> Dict:
        """
        Detecta variaciones cromáticas rítmicas compatibles con pulso humano.
        Los fakes de IA tienen ruido aleatorio o señal plana (SNR < 0.1).
        """
        if len(frames_bgr) < 30:
            return {"pulse_snr": 0.0, "suspicion": 0.5, "available": False}
        
        green_means = []
        for frame, lms in zip(frames_bgr, landmark_sequences):
            h, w = frame.shape[:2]
            pts = np.array([[int(lms[i].x * w), int(lms[i].y * h)] for i in self.FOREHEAD_PATCH])
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)
            # Canal G es el óptimo para captura de oximetría por reflexión
            mean_val = cv2.mean(frame[:,:,1], mask=mask)[0]
            green_means.append(mean_val)
        
        # Procesamiento de señal: FFT
        signal = np.array(green_means)
        signal = signal - np.mean(signal)
        
        try:
            from scipy.fft import rfft, rfftfreq
            n = len(signal)
            yf = rfft(signal)
            xf = rfftfreq(n, 1/fps)
            
            # Buscamos en el rango 0.7 - 3.0 Hz (42 - 180 BPM)
            mask_h = (xf >= 0.7) & (xf <= 3.0)
            if not np.any(mask_h):
                 return {"pulse_snr": 0.0, "suspicion": 0.5, "available": False}
                 
            power_in_range = np.abs(yf[mask_h])**2
            peak_idx = np.argmax(power_in_range)
            peak_bpm = float(xf[mask_h][peak_idx]) * 60
            
            # SNR: Ratio de la potencia del pico vs ruido ambiental en banda
            peak_power = power_in_range[peak_idx]
            noise_power = np.mean(power_in_range) + 1e-10
            snr = peak_power / noise_power
            
            suspicion = 0.0
            if snr < 1.1:   suspicion = 0.85 # Señal plana (IA)
            elif snr < 1.4: suspicion = 0.50 # Pulso muy débil / borroso
            elif snr > 2.5: suspicion = 0.0  # Firma vital clara
            
            return {
                "pulse_bpm": round(peak_bpm, 1),
                "pulse_snr": round(float(snr), 3),
                "suspicion": round(suspicion, 3),
                "available": True
            }
        except Exception:
            return {"pulse_snr": 0.0, "suspicion": 0.5, "available": False}

    # ------------------------------------------------------------------
    # Mesh Sliding: Detección de "flotación" de la malla facial
    # ------------------------------------------------------------------
    def _analyze_mesh_sliding(self, landmark_sequences: List[list]) -> Dict:
        """
        Detecta si la máscara facial se deforma o desliza sobre el cráneo real.
        Compara el movimiento de la nariz (ancla) vs el resto de la cara.
        """
        if len(landmark_sequences) < 10:
            return {"avg_drift": 0.0, "suspicion": 0.4}
            
        drifts = []
        for i in range(1, len(landmark_sequences)):
            prev = landmark_sequences[i-1]
            curr = landmark_sequences[i]
            
            # Movimiento del punto de anclaje (Nariz)
            prev_nose = np.array([prev[4].x, prev[4].y])
            curr_nose = np.array([curr[4].x, curr[4].y])
            nose_vec  = curr_nose - prev_nose
            
            # Drift en puntos móviles (mandíbula, cejas)
            key_pts = [152, 10, 234, 454, 70, 300]
            frame_drifts = []
            for j in key_pts:
                p_rel = np.array([prev[j].x, prev[j].y]) - prev_nose
                c_rel = np.array([curr[j].x, curr[j].y]) - curr_nose
                # El drift es la diferencia vectorial entre marcos de referencia
                # Si el punto j se mueve distinto a la nariz de forma incoherente:
                frame_drifts.append(np.linalg.norm(c_rel - p_rel))
            
            drifts.append(float(np.mean(frame_drifts)))
            
        avg_drift = float(np.mean(drifts))
        # IA tiene jitter de malla que causa drift > 0.003
        suspicion = 0.0
        if avg_drift > 0.005:   suspicion = 0.90 # Mesh sliding evidente (swap)
        elif avg_drift > 0.002: suspicion = 0.40
        
        return {
            "avg_drift": round(avg_drift, 5),
            "suspicion": round(suspicion, 3)
        }

    # ------------------------------------------------------------------
    # Viseme Extraction (Para Deep-Sync)
    # ------------------------------------------------------------------
    def _extract_visemes(self, landmark_sequences: List[list]) -> List[Dict]:
        """Extrae apertura y estiramiento de boca para sincronización."""
        visemes = []
        for lms in landmark_sequences:
            # Apertura vertical
            v_dist = np.linalg.norm(np.array([lms[13].x, lms[13].y]) - np.array([lms[14].x, lms[14].y]))
            # Estiramiento horizontal
            h_dist = np.linalg.norm(np.array([lms[61].x, lms[61].y]) - np.array([lms[291].x, lms[291].y]))
            visemes.append({"v": float(v_dist), "h": float(h_dist)})
        return visemes

    # ------------------------------------------------------------------
    # Detección de cara con MediaPipe (reemplaza HaarCascade)
    # ------------------------------------------------------------------
    def _detect_face_rect(self, frame_bgr: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
        """Detecta cara y retorna (x, y, w, h) o None."""
        try:
            import mediapipe as mp
            mp_detect = mp.solutions.face_detection
            with mp_detect.FaceDetection(min_detection_confidence=0.5) as det:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                res = det.process(rgb)
                if res.detections:
                    bbox = res.detections[0].location_data.relative_bounding_box
                    h, w = frame_bgr.shape[:2]
                    x = max(0, int(bbox.xmin * w))
                    y = max(0, int(bbox.ymin * h))
                    bw = min(int(bbox.width * w), w - x)
                    bh = min(int(bbox.height * h), h - y)
                    return (x, y, bw, bh)
        except Exception:
            pass
        # Fallback a HaarCascade
        try:
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
            if len(faces) > 0:
                return tuple(faces[0])
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Método principal
    # ------------------------------------------------------------------
    def analyze(self, frames_bgr: List[np.ndarray], fps: float = 30.0) -> Dict:
        """
        Análisis facial completo sobre secuencia de frames.
        
        Returns:
            Dict con scores de sub-análisis y suspicion global [0-1]
        """
        if not frames_bgr:
            return {"suspicion": 0.3, "available": False, "error": "sin frames"}

        if self._face_mesh is None:
            return {"suspicion": 0.3, "available": False, "error": "MediaPipe no disponible"}

        landmark_sequences = []
        ear_left_seq       = []
        ear_right_seq      = []
        face_rects         = []
        frames_with_face   = 0

        for frame in frames_bgr:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._face_mesh.process(rgb)

            if results.multi_face_landmarks:
                lms = results.multi_face_landmarks[0].landmark
                landmark_sequences.append(lms)
                frames_with_face += 1

                ear_l = self._ear(lms, self.LEFT_EAR_PTS)
                ear_r = self._ear(lms, self.RIGHT_EAR_PTS)
                ear_left_seq.append(ear_l)
                ear_right_seq.append(ear_r)

                # Face rect aproximado desde landmarks
                xs = [lms[i].x * frame.shape[1] for i in self.FACE_OVAL]
                ys = [lms[i].y * frame.shape[0] for i in self.FACE_OVAL]
                x, y = int(min(xs)), int(min(ys))
                w = int(max(xs)) - x
                h = int(max(ys)) - y
                face_rects.append((x, y, w, h))
            else:
                face_rects.append(None)

        if frames_with_face < 3:
            # Si el video es un paisaje IA (Sora), no penalizar el score global 
            # con un 0.4 neutral, sino apagar el módulo ("available": False)
            # para que el Score final se calcule 100% con ViT y Forense.
            return {
                "suspicion":      0.5,
                "available":      False,
                "faces_detected": frames_with_face,
                "detail":         "insuficientes_caras"
            }

        # --- Ejecutar sub-análisis en Paralelo ---
        ear_combined = np.array(ear_left_seq) * 0.5 + np.array(ear_right_seq) * 0.5

        # Preprocesar argumentos para chrominance
        cb_frames = [f for f, r in zip(frames_bgr, face_rects) if r is not None]
        cb_lms    = [lms for lms, r in zip(landmark_sequences, face_rects) if r is not None]

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            fut_blink    = executor.submit(self._analyze_blink_kinematics, ear_combined, fps)
            fut_micro    = executor.submit(self._analyze_microexpressions, landmark_sequences, fps)
            fut_asym     = executor.submit(self._analyze_facial_asymmetry, landmark_sequences)
            fut_chroma   = executor.submit(self._analyze_skin_chrominance, cb_frames, cb_lms)
            fut_pulse    = executor.submit(self._analyze_rppg_pulse, cb_frames, cb_lms, fps)
            fut_sliding  = executor.submit(self._analyze_mesh_sliding, landmark_sequences)

            blink_result       = fut_blink.result()
            microexp_result    = fut_micro.result()
            asymmetry_result   = fut_asym.result()
            chrominance_result = fut_chroma.result()
            pulse_result       = fut_pulse.result()
            sliding_result     = fut_sliding.result()

        # --- Agregación Omega V5 ---
        sub_suspicions = {
            "blink":       blink_result["suspicion"],
            "microexp":    microexp_result["suspicion"],
            "asymmetry":   asymmetry_result["suspicion"],
            "chrominance": chrominance_result["suspicion"],
            "pulse":       pulse_result["suspicion"],
            "sliding":     sliding_result["suspicion"]
        }

        weights = {
            "blink":       0.15,
            "microexp":    0.20,
            "asymmetry":   0.10,
            "chrominance": 0.15,
            "pulse":       0.25, # Peso alto: firma biológica crítica
            "sliding":     0.15
        }

        total = sum(sub_suspicions[k] * weights[k] for k in sub_suspicions)
        viseme_seq = self._extract_visemes(landmark_sequences)

        return {
            "suspicion":            round(min(1.0, total), 3),
            "suspicion_components": sub_suspicions,
            "faces_detected":       frames_with_face,
            "face_detection_ratio": round(frames_with_face / len(frames_bgr), 3),
            "blink_analysis":       blink_result,
            "microexpression":      microexp_result,
            "asymmetry":            asymmetry_result,
            "skin_chrominance":     chrominance_result,
            "pulse_rppg":           pulse_result,
            "mesh_sliding":         sliding_result,
            "viseme_sequence":      viseme_seq,
            "available":            True
        }
