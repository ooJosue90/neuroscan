"""
MÓDULO: Temporal Analyzer V4.1
Detecta inconsistencias físicas frame-a-frame:
  - RAFT optical flow con análisis cinemático completo en GPU (single-pass)
  - Filtro Gaussiano para mitigar artefactos de compresión H.264 en el flujo
  - Ghosting (Runway Gen-3 signature)
  - Incoherencia biomecánica de pose
  - Pixel mutation flicker
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import warnings


class TemporalAnalyzer:
    """
    Analizador temporal avanzado.
    Reemplaza el Farneback simple con análisis de campo vectorial completo.
    """

    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self._raft_model = None
        self._pose_model = None
        print(">>> [TemporalAnalyzer] Inicializado (GPU:", use_gpu, ")")

    # ------------------------------------------------------------------
    # RAFT Optical Flow (requiere torchvision >= 0.13)
    # ------------------------------------------------------------------
    def _load_raft(self):
        if self._raft_model is not None:
            return
        # RAFT en CPU es demasiado lento para uso en producción (2-5s por par de frames).
        # Solo se carga si hay GPU CUDA disponible.
        if not self.use_gpu:
            self._raft_model = None
            return
        try:
            import torch
            if not torch.cuda.is_available():
                self._raft_model = None
                return
            from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
            weights = Raft_Small_Weights.DEFAULT
            self._raft_model = raft_small(weights=weights).eval().cuda()
            self._raft_transforms = weights.transforms()
            print(">>> [TemporalAnalyzer] RAFT cargado en CUDA")
        except Exception as e:
            print(f">>> [TemporalAnalyzer] RAFT no disponible, usando Farneback: {e}")
            self._raft_model = None

    def _preprocess_raft(self, frame_bgr: np.ndarray):
        """Convierte frame BGR a tensor float32 normalizado para RAFT."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        return tensor.unsqueeze(0)

    # ------------------------------------------------------------------
    # Análisis cinemático GPU unificado (RAFT + Gaussiano + Jacobiano)
    # ------------------------------------------------------------------
    def _create_gaussian_kernel(self, device: torch.device) -> torch.Tensor:
        """Crea un kernel de suavizado Gaussiano 3x3 en la GPU."""
        kernel = torch.tensor([[
            [1/16, 1/8,  1/16],
            [1/8,  1/4,  1/8 ],
            [1/16, 1/8,  1/16]
        ]], dtype=torch.float32, device=device)
        return kernel.unsqueeze(0)  # [1, 1, 3, 3]

    def _compute_and_analyze_raft(
        self, f1_bgr: np.ndarray, f2_bgr: np.ndarray
    ) -> Optional[Dict]:
        """
        RAFT + análisis cinemático completo en un solo paso GPU.

        Ventajas frente al pipeline antiguo:
          - Filtro Gaussiano que limpia artefactos de compresión H.264
          - torch.gradient (diferencias centrales) más preciso que np.gradient
          - Solo 6 escalares pasan GPU→CPU (antes: matriz H×W completa)

        Retorna dict con métricas cinemáticas o None si RAFT no disponible.
        """
        self._load_raft()
        if self._raft_model is None:
            return None

        try:
            t1 = self._preprocess_raft(f1_bgr)
            t2 = self._preprocess_raft(f2_bgr)

            h, w = f1_bgr.shape[:2]
            new_h, new_w = (h // 8) * 8, (w // 8) * 8

            t1 = F.interpolate(t1, size=(new_h, new_w), mode='bilinear', align_corners=False)
            t2 = F.interpolate(t2, size=(new_h, new_w), mode='bilinear', align_corners=False)

            device = next(self._raft_model.parameters()).device
            t1, t2 = t1.to(device), t2.to(device)

            with torch.no_grad():
                flow_preds = self._raft_model(t1, t2)   # list de refinamientos

            # Último refinamiento RAFT: [2, H, W]
            flow_tensor = flow_preds[-1].squeeze(0)

            u = flow_tensor[0].unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
            v = flow_tensor[1].unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

            # 1. Suavizado Gaussiano (elimina ruido de compresión H.264)
            gk = self._create_gaussian_kernel(device)
            u_s = F.conv2d(u, gk, padding=1).squeeze()   # [H,W]
            v_s = F.conv2d(v, gk, padding=1).squeeze()   # [H,W]

            # 2. Gradientes con diferencias centrales
            du_dy, du_dx = torch.gradient(u_s, dim=(0, 1))
            dv_dy, dv_dx = torch.gradient(v_s, dim=(0, 1))

            # 3. Cinética del campo vectorial
            divergence  = du_dx + dv_dy
            curl        = dv_dx - du_dy
            magnitude   = torch.sqrt(u_s**2 + v_s**2)
            det_J       = du_dx * dv_dy - du_dy * dv_dx

            # 4. Reducción a escalares (solo 6 valores pasan a CPU)
            div_mean  = float(divergence.abs().mean().item())
            curl_mean = float(curl.abs().mean().item())
            mag_mean  = float(magnitude.mean().item())
            det_std   = float(det_J.std().item())
            det_mean  = det_J.mean()
            det_skew  = float(
                (((det_J - det_mean)**3).mean() / (det_std**3 + 1e-10)).item()
            )

            # IA Generativa estática es extremadamente suave
            smooth_suspicion = 0.0
            if div_mean < 0.22:
                smooth_suspicion = min(1.0, (0.22 - div_mean) * 5.0)

            naturalness          = 1.0 / (1.0 + div_mean + curl_mean)
            discontinuity_score  = float(min(1.0, det_std * 0.20 + abs(det_skew) * 0.05 + smooth_suspicion))

            return {
                "divergence_mean":       div_mean,
                "curl_mean":             curl_mean,
                "magnitude_mean":        mag_mean,
                "field_naturalness":     naturalness,
                "jacobian_discontinuity":discontinuity_score,
            }

        except Exception as e:
            print(f">>> [TemporalAnalyzer] RAFT GPU Analysis error: {e}")
            return None

    def _compute_cpu_flow(self, f1_gray: np.ndarray, f2_gray: np.ndarray) -> np.ndarray:
        """Fallback: DIS Optical Flow (mucho más rápido que Farneback)."""
        dis = cv2.DISOpticalFlow_create(cv2.DISOpticalFlow_PRESET_MEDIUM)
        return dis.calc(f1_gray, f2_gray, None)

    # ------------------------------------------------------------------
    # Análisis de campo vectorial
    # ------------------------------------------------------------------
    def _analyze_flow_field(self, flow: np.ndarray) -> Dict:
        """
        Calcula divergencia y rotacional del campo vectorial.
        
        Física real: movimiento humano → divergencia baja, rotacional suave.
        IA (Sora/Runway): discontinuidades bruscas en bordes semánticos.
        """
        u = flow[..., 0].astype(np.float64)
        v = flow[..., 1].astype(np.float64)

        # Gradientes numéricos
        du_dx = np.gradient(u, axis=1)
        du_dy = np.gradient(u, axis=0)
        dv_dx = np.gradient(v, axis=1)
        dv_dy = np.gradient(v, axis=0)

        divergence = du_dx + dv_dy          # Expansión/compresión del campo
        curl = dv_dx - du_dy                # Rotación local del campo

        magnitude = np.sqrt(u**2 + v**2)

        return {
            "divergence_mean":   float(np.mean(np.abs(divergence))),
            "divergence_std":    float(np.std(divergence)),
            "curl_mean":         float(np.mean(np.abs(curl))),
            "curl_std":          float(np.std(curl)),
            "magnitude_mean":    float(np.mean(magnitude)),
            "magnitude_std":     float(np.std(magnitude)),
            # Score de "naturalidad" del campo vectorial
            "field_naturalness": float(1.0 / (1.0 + np.mean(np.abs(divergence)) + np.mean(np.abs(curl))))
        }

    def _compute_flow_jacobian_discontinuity(self, flow: np.ndarray) -> float:
        """
        Detecta discontinuidades imposibles en el Jacobiano del flujo.
        Sora produce discontinuidades en bordes semánticos que violan física básica.
        """
        u = flow[..., 0].astype(np.float64)
        v = flow[..., 1].astype(np.float64)

        # Jacobiano: [[du_dx, du_dy], [dv_dx, dv_dy]]
        du_dx = np.gradient(u, axis=1)
        du_dy = np.gradient(u, axis=0)
        dv_dx = np.gradient(v, axis=1)
        dv_dy = np.gradient(v, axis=0)

        # Determinante del Jacobiano (conservación de volumen)
        det_J = du_dx * dv_dy - du_dy * dv_dx

        # En flujo físico real: det_J ≈ 1 (compresible pero continuo)
        # En IA generativa: det_J tiene spikes bruscos (discontinuidades de compresión)
        det_std = float(np.std(det_J))
        det_mean = float(np.mean(det_J))
        det_skew = float(np.mean((det_J - det_mean)**3) / (det_std**3 + 1e-10))

        # Alta desviación estándar + asimetría → incoherencia física
        discontinuity_score = min(1.0, (det_std * 0.20 + abs(det_skew) * 0.05))
        return discontinuity_score

    # ------------------------------------------------------------------
    # Ghosting detector (Runway Gen-3 signature)
    # ------------------------------------------------------------------
    def _detect_ghosting(self, frames_gray: List[np.ndarray]) -> Dict:
        """
        Runway Gen-3 deja 'fantasmas' de objetos desaparecidos.
        Se detecta comparando diferencias a distancia-1 y distancia-2.
        
        Si diff(t, t-2) < diff(t, t-1) en zonas con movimiento → ghosting.
        """
        ghost_scores = []

        for i in range(2, len(frames_gray)):
            f0 = frames_gray[i-2].astype(float)
            f1 = frames_gray[i-1].astype(float)
            f2 = frames_gray[i].astype(float)

            diff_1 = np.abs(f2 - f1)
            diff_2 = np.abs(f2 - f0)

            # Máscara de movimiento real
            motion_mask = (diff_1 > 12).astype(float)
            motion_area = np.sum(motion_mask) + 1e-6

            if motion_area < 500:  # Sin suficiente movimiento, saltar
                continue

            # Ghosting: el frame anterior-anterior es más similar al actual
            # que el frame anterior (objeto que "reaparece")
            ghost_pixels = np.sum((diff_2 < diff_1 * 0.65) * motion_mask)
            ghost_ratio = ghost_pixels / motion_area
            ghost_scores.append(ghost_ratio)

        if not ghost_scores:
            return {"ghosting_ratio": 0.0, "suspicion": 0.1}

        avg_ghosting = float(np.mean(ghost_scores))
        suspicion = min(1.0, avg_ghosting * 1.5)  # Umbral más tolerante

        return {
            "ghosting_ratio":    avg_ghosting,
            "ghosting_max":      float(np.max(ghost_scores)),
            "ghosting_samples":  len(ghost_scores),
            "suspicion":         suspicion
        }

    # ------------------------------------------------------------------
    # Análisis biomecánico con MediaPipe Pose
    # ------------------------------------------------------------------
    def _load_pose(self):
        if self._pose_model is not None:
            return
        try:
            import mediapipe as mp
            self._mp_pose = mp.solutions.pose
            self._pose_model = self._mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print(">>> [TemporalAnalyzer] MediaPipe Pose cargado")
        except Exception as e:
            print(f">>> [TemporalAnalyzer] MediaPipe Pose no disponible: {e}")
            self._pose_model = None

    def _analyze_pose_biomechanics(self, frames_bgr: List[np.ndarray]) -> Dict:
        """
        Analiza aceleraciones articulares.
        Cuerpo humano: aceleraciones suaves con inercia natural.
        IA (Sora): aceleraciones abruptas que violan biomecánica.
        """
        self._load_pose()
        if self._pose_model is None:
            return {"biomechanics_suspicion": 0.2, "available": False}

        joint_positions = []  # Lista de dicts por frame

        for frame in frames_bgr:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._pose_model.process(rgb)
            if results.pose_landmarks:
                lms = results.pose_landmarks.landmark
                # Extraer articulaciones clave: hombros, codos, muñecas, caderas
                joints = {
                    "l_shoulder": (lms[11].x, lms[11].y),
                    "r_shoulder": (lms[12].x, lms[12].y),
                    "l_elbow":    (lms[13].x, lms[13].y),
                    "r_elbow":    (lms[14].x, lms[14].y),
                    "l_wrist":    (lms[15].x, lms[15].y),
                    "r_wrist":    (lms[16].x, lms[16].y),
                    "l_hip":      (lms[23].x, lms[23].y),
                    "r_hip":      (lms[24].x, lms[24].y),
                }
                joint_positions.append(joints)

        if len(joint_positions) < 5:
            return {"biomechanics_suspicion": 0.2, "available": False, "pose_frames": 0}

        # Calcular velocidades y aceleraciones
        joint_names = list(joint_positions[0].keys())
        accel_magnitudes = []

        for jname in joint_names:
            positions = np.array([jp[jname] for jp in joint_positions if jname in jp])
            if len(positions) < 3:
                continue

            velocities     = np.diff(positions, axis=0)
            accelerations  = np.diff(velocities, axis=0)
            accel_mag      = np.linalg.norm(accelerations, axis=1)
            accel_magnitudes.extend(accel_mag.tolist())

        if not accel_magnitudes:
            return {"biomechanics_suspicion": 0.2, "available": True, "pose_frames": len(joint_positions)}

        accel_array = np.array(accel_magnitudes)
        # Aceleraciones muy altas o muy bajas (movimiento congelado) son sospechosas
        high_accel_ratio  = float(np.mean(accel_array > 0.05))  # >5% del frame = imposible
        zero_motion_ratio = float(np.mean(accel_array < 0.0005))

        suspicion = 0.0
        if high_accel_ratio > 0.1:
            suspicion += 0.5
        if zero_motion_ratio > 0.7:
            suspicion += 0.3

        return {
            "biomechanics_suspicion": min(1.0, suspicion),
            "high_accel_ratio":       high_accel_ratio,
            "zero_motion_ratio":      zero_motion_ratio,
            "mean_acceleration":      float(np.mean(accel_array)),
            "accel_std":              float(np.std(accel_array)),
            "available":              True,
            "pose_frames":            len(joint_positions)
        }

    # ------------------------------------------------------------------
    # Análisis de textura temporal (pixel mutation)
    # ------------------------------------------------------------------
    def _analyze_texture_consistency(self, frames_bgr: List[np.ndarray]) -> Dict:
        """
        Runway/Pika: las texturas de fondo cambian de forma independiente al movimiento.
        Detecta zonas con alta variación de textura sin movimiento correspondiente.
        """
        if len(frames_bgr) < 3:
            return {"texture_inconsistency": 0.0, "suspicion": 0.1}

        inconsistencies = []

        for i in range(1, len(frames_bgr) - 1):
            prev = cv2.cvtColor(frames_bgr[i-1], cv2.COLOR_BGR2GRAY).astype(float)
            curr = cv2.cvtColor(frames_bgr[i],   cv2.COLOR_BGR2GRAY).astype(float)
            nxt  = cv2.cvtColor(frames_bgr[i+1], cv2.COLOR_BGR2GRAY).astype(float)

            # LBP simplificado: varianza local de textura
            kernel_size = 5
            prev_tex = cv2.Laplacian(prev.astype(np.uint8), cv2.CV_64F)
            curr_tex = cv2.Laplacian(curr.astype(np.uint8), cv2.CV_64F)

            # Cambio de textura
            tex_change = np.abs(curr_tex - prev_tex)

            # Movimiento real (Farneback rápido para esta métrica)
            flow = cv2.calcOpticalFlowFarneback(
                prev.astype(np.uint8), curr.astype(np.uint8),
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            motion_mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2)

            # Inconsistencia: textura cambia mucho pero movimiento es bajo
            static_mask  = (motion_mag < 1.0).astype(float)
            inconsistent = np.sum(tex_change * static_mask > 20) / (np.sum(static_mask) + 1e-6)
            inconsistencies.append(float(inconsistent))

        avg_inc = float(np.mean(inconsistencies)) if inconsistencies else 0.0
        suspicion = min(1.0, avg_inc * 3.0)

        return {
            "texture_inconsistency": avg_inc,
            "suspicion":             suspicion,
            "samples":               len(inconsistencies)
        }

    # ------------------------------------------------------------------
    # Método principal
    # ------------------------------------------------------------------
    def analyze(self, frames_bgr: List[np.ndarray], sample_step: int = 3) -> Dict:
        """
        Análisis temporal completo.
        
        Args:
            frames_bgr: Lista de frames en BGR
            sample_step: Paso de muestreo para flujo óptico (mayor = más rápido)
        
        Returns:
            Dict con scores de todos los sub-análisis y score final [0-1]
        """
        if len(frames_bgr) < 4:
            return {"suspicion": 0.3, "error": "Insuficientes frames", "available": False}

        frames_gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames_bgr]

        # --- 1. Flujo óptico y análisis de campo vectorial ---
        # Limitar pares analizados: RAFT en GPU → máx 20, Farneback CPU → máx 15
        # Esto evita timeouts en hardware sin aceleración.
        has_gpu = self.use_gpu and self._raft_model is not None
        max_pairs = 20 if has_gpu else 15

        indices = list(range(0, len(frames_bgr) - 1, sample_step))
        if len(indices) > max_pairs:
            # Submuestreo uniforme para no exceder max_pairs
            step2 = max(1, len(indices) // max_pairs)
            indices = indices[::step2][:max_pairs]

        flow_field_metrics = []
        jacobian_scores    = []

        for i in indices:
            # ── Rama GPU: análisis cinemático unificado en CUDA ──────────────────
            if self._raft_model is not None:
                gpu_result = self._compute_and_analyze_raft(
                    frames_bgr[i], frames_bgr[i + 1]
                )
                if gpu_result is not None:
                    # El resultado GPU ya incluye field_naturalness y jacobian
                    flow_field_metrics.append({
                        "divergence_mean":  gpu_result["divergence_mean"],
                        "curl_mean":        gpu_result["curl_mean"],
                        "magnitude_mean":   gpu_result["magnitude_mean"],
                        "field_naturalness":gpu_result["field_naturalness"],
                        # campos extra que espera _aggregate (pueden ser 0)
                        "divergence_std":   0.0,
                        "curl_std":         0.0,
                        "magnitude_std":    0.0,
                    })
                    jacobian_scores.append(gpu_result["jacobian_discontinuity"])
                    continue   # skip CPU path

            # ── Rama CPU: DIS Optical Flow + análisis NumPy (fallback sin GPU) ──────
            flow    = self._compute_cpu_flow(frames_gray[i], frames_gray[i + 1])
            field_m = self._analyze_flow_field(flow)
            jac_s   = self._compute_flow_jacobian_discontinuity(flow)

            flow_field_metrics.append(field_m)
            jacobian_scores.append(jac_s)

        # Agregar métricas de flujo
        div_mean   = float(np.mean([m["divergence_mean"]   for m in flow_field_metrics]))
        curl_mean  = float(np.mean([m["curl_mean"]         for m in flow_field_metrics]))
        mag_mean   = float(np.mean([m["magnitude_mean"]    for m in flow_field_metrics]))
        naturalness= float(np.mean([m["field_naturalness"] for m in flow_field_metrics]))
        jac_score  = float(np.mean(jacobian_scores))

        # --- 2. Ghosting ---
        ghosting_result = self._detect_ghosting(frames_gray)

        # --- 3. Biomecánica ---
        # Usar subset de frames para no sobrecargar pose model
        pose_frames = frames_bgr[::max(1, len(frames_bgr)//20)]
        biomech_result = self._analyze_pose_biomechanics(pose_frames)

        # --- 4. Consistencia de textura ---
        tex_frames = frames_bgr[::max(1, len(frames_bgr)//15)]
        texture_result = self._analyze_texture_consistency(tex_frames)

        # --- 5. Agregación de sospecha temporal ---
        suspicion_components = {
            "flow_field":    min(1.0, max(0.0, (1.0 - naturalness) * 0.3)), # Reducido severamente para tolerar movimiento
            "jacobian":      jac_score,
            "ghosting":      ghosting_result["suspicion"],
            "biomechanics":  biomech_result.get("biomechanics_suspicion", 0.2),
            "texture":       texture_result["suspicion"]
        }

        # Pesos de cada componente
        # JACOBIAN: Máxima fiabilidad matemática contra Sora/T2V (Violación de conservación volumen)
        # FLOW: Fiabilidad alta pero con ruido posible.
        # GHOSTING: Reducido al mínimo (0.05) porque desenfoques de movimiento humano lo activan falso.
        weights = {
            "flow_field":   0.15,
            "jacobian":     0.50,
            "ghosting":     0.05,
            "biomechanics": 0.10,
            "texture":      0.20
        }

        # ----------------------------------------------------
        # OPCIÓN 2: Escalamiento Suave Temporal (Jacobiano Ponderado)
        # ----------------------------------------------------
        # El Jacobiano detecta discontinuidades físicas, pero la compresión H.264
        # en videos estáticos genera ruido que el detector confunde con IA.
        # Escalamos la importancia del Jacobiano según el movimiento real (mag_mean).
        # Si no hay movimiento, el Jacobiano casi no tiene peso.
        
        # mag_mean suele estar entre 0 (estático) y 50 (acción rápida).
        # Usamos un factor de escala suave: si mag_mean < 5.0, el peso se reduce proporcionalmente.
        jac_modifier = min(1.0, mag_mean / 5.0)
        suspicion_components["jacobian"] *= jac_modifier
        
        total_suspicion = sum(
            suspicion_components[k] * weights[k]
            for k in suspicion_components
        )

        # Penalización extrema para Generación de Video (Sora/Runway)
        # Solo penar si el video es "demasiado suave" habiendo movimiento.
        if div_mean < 0.25 and mag_mean > 2.0:
            smooth_penalty = min(1.0, (0.25 - div_mean) * 5.0)
            total_suspicion = max(total_suspicion, smooth_penalty)

        return {
            "suspicion":              float(min(1.0, total_suspicion)),
            "suspicion_components":   suspicion_components,
            "flow_divergence_mean":   div_mean,
            "flow_curl_mean":         curl_mean,
            "flow_naturalness":       naturalness,
            "jacobian_discontinuity": jac_score,
            "ghosting":               ghosting_result,
            "biomechanics":           biomech_result,
            "texture_consistency":    texture_result,
            "frames_analyzed":        len(frames_bgr),
            "available":              True
        }
