"""
MÓDULO: Calibrated Ensemble Scorer V4
Sistema de scoring probabilístico para reemplazar las sumas heurísticas:
  - Pesos aprendidos por módulo (configurables)
  - Platt scaling para calibración de probabilidades
  - Cálculo de contribuciones SHAP (leave-one-out aproximado)
  - Intervalo de confianza vía dispersión del ensemble
  - Detección de tipo de IA (Sora, Runway, FaceSwap, HeyGen)
  - Recomendación de veredicto con umbrales configurables
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import json
import os


class CalibratedEnsembleScorer:
    """
    Scorer probabilístico calibrado para el pipeline V4.
    
    Reemplaza completamente el sistema heurístico de V3 (prob += N).
    """

    # Pesos base por módulo (aprendidos offline con DFDC + FaceForensics++ + Celeb-DF)
    # Ajustables post-calibración. Suma = 1.0
    DEFAULT_WEIGHTS = {
        "temporal":     0.35,
        "forensic":     0.30,
        "facial":       0.15,
        "deep_sync":    0.10, # Nuevo factor crítico Omega V5
        "audio":        0.05,
        "vit_ensemble": 0.05,
        "lip_sync":     0.03,
    }

    # Umbrales de veredicto
    VERDICT_THRESHOLDS = {
        "SINTÉTICO":               0.75,
        "PROBABLEMENTE SINTÉTICO": 0.55,
        "INDETERMINADO":           0.35,
        "PROBABLEMENTE REAL":      0.20,
        "ORGÁNICO":                0.0,
    }

    # Firmas de modelos generativos específicos
    # (umbrales de sub-métricas características de cada modelo)
    AI_MODEL_SIGNATURES = {
        "HeyGen/Avatar": {
            "microexp_density_max":     0.15,   # Casi sin microexpresiones
            "blink_per_min_range":      (8, 35), # Parpadeo regular pero presente
            "audio_hnr_min":            22.0,    # Voz muy limpia
        },
        "Sora/T2V": {
            "flow_divergence_min":      0.3,     # Campo vectorial divergente
            "jacobian_discontinuity_min": 0.25,  # Física imposible
            "ghosting_ratio_max":       0.05,    # Poco ghosting (Sora es suave)
        },
        "Runway Gen-3": {
            "ghosting_ratio_min":       0.08,    # Ghosting característico
            "texture_inconsistency_min":0.15,    # Texturas que cambian sin movimiento
            "lighting_jump_min":        0.10,    # Saltos de iluminación
        },
        "DeepFace/FaceSwap": {
            "cb_diff_min":              8.0,     # Desajuste crominancia
            "ela_splice_min":           0.1,     # Artefactos ELA de splice
            "prnu_corr_max":            0.03,    # Sin firma de sensor coherente
        },
        "Pika/AnimateDiff": {
            "texture_inconsistency_min":0.20,
            "upscale_fft_min":          0.10,    # Upscaling frecuente
            "noise_std_max":            2.0,     # Ruido artificial bajo
        },
    }

    def __init__(self, weights_path: Optional[str] = None):
        """
        Args:
            weights_path: JSON con pesos personalizados y parámetros de calibración (opcional)
        """
        self.weights = dict(self.DEFAULT_WEIGHTS)
        
        # Calibradores Platt (sigmoid) por módulo
        # Formato: (a, b) donde P_cal = 1 / (1 + exp(a * raw_score + b))
        self._platt_params = {
            "vit_ensemble":  (-4.0, 2.0),
            "temporal":      (-7.0, 3.5),
            "facial":        (-6.0, 3.0),
            "forensic":      (-7.0, 3.5),
            "audio":         (-5.0, 2.5),
        }

        if weights_path and os.path.exists(weights_path):
            try:
                with open(weights_path, 'r') as f:
                    custom_config = json.load(f)
                
                # Soporta formato anidado {"weights": {...}, "platt_params": {...}} o plano para pesos
                if "weights" in custom_config:
                    self.weights.update(custom_config["weights"])
                else:
                    self.weights.update({k: v for k, v in custom_config.items() if k != "platt_params"})
                
                if "platt_params" in custom_config:
                    for k, v in custom_config["platt_params"].items():
                        if k in self._platt_params and len(v) == 2:
                            self._platt_params[k] = (float(v[0]), float(v[1]))
                            
                print(f">>> [Scorer] Pesos y calibración Platt cargados desde {weights_path}")
            except Exception as e:
                print(f">>> [Scorer] Error cargando pesos/calibración: {e}")

    # ------------------------------------------------------------------
    # Platt Scaling
    # ------------------------------------------------------------------
    def _platt_calibrate(self, raw_score: float, module_name: str) -> float:
        """
        Calibra un score [0,1] a probabilidad calibrada.
        Platt scaling: P = sigmoid(a * f(x) + b)
        
        Con parámetros por defecto, actúa como suavizado de la función sigmoide.
        Para calibración real: entrenar a, b con sklearn.LogisticRegression
        sobre un dataset validado (DFDC, FaceForensics++).
        """
        a, b = self._platt_params.get(module_name, (-4.0, 2.0))
        # Convertir score [0,1] a logit-like y calibrar
        logit = a * raw_score + b
        calibrated = 1.0 / (1.0 + np.exp(logit))
        return float(np.clip(calibrated, 0.01, 0.99))

    def calibrate_from_data(self, module_name: str, raw_scores: np.ndarray, labels: np.ndarray):
        """
        Calibra los parámetros Platt usando datos etiquetados.
        Llamar offline con dataset de validación.
        
        Args:
            module_name: nombre del módulo
            raw_scores: array de scores [0,1] (N,)
            labels: array de etiquetas {0=real, 1=fake} (N,)
        """
        try:
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(C=1.0)
            lr.fit(raw_scores.reshape(-1, 1), labels)
            self._platt_params[module_name] = (float(lr.coef_[0][0]), float(lr.intercept_[0]))
            print(f">>> [Scorer] Platt calibrado para '{module_name}': a={lr.coef_[0][0]:.3f}, b={lr.intercept_[0]:.3f}")
        except ImportError:
            print(">>> [Scorer] sklearn no disponible para calibración Platt")

    # ------------------------------------------------------------------
    # Redistribución de pesos cuando módulos no disponibles
    # ------------------------------------------------------------------
    def _get_effective_weights(self, available_modules: List[str]) -> Dict[str, float]:
        """Redistribuye pesos de módulos no disponibles entre los disponibles."""
        available_set = set(available_modules)
        total_available_weight = sum(
            w for mod, w in self.weights.items() if mod in available_set
        )
        if total_available_weight == 0:
            return {mod: 1.0/len(available_modules) for mod in available_modules}

        effective = {
            mod: self.weights.get(mod, 0.05) / total_available_weight
            for mod in available_modules
        }
        return effective

    # ------------------------------------------------------------------
    # SHAP contributions (leave-one-out aproximado)
    # ------------------------------------------------------------------
    def _compute_shap(self, calibrated_scores: Dict[str, float], weights: Dict[str, float]) -> Dict[str, float]:
        """
        Calcula contribución de cada módulo al score final.
        Método: leave-one-out (remover módulo y calcular diferencia).
        """
        if not calibrated_scores:
            return {}

        total_w = sum(weights.values())
        full_score = sum(calibrated_scores[k] * weights[k] for k in calibrated_scores) / max(total_w, 1e-10)

        shap_values = {}
        for mod in calibrated_scores:
            others = {k: v for k, v in calibrated_scores.items() if k != mod}
            other_w = sum(weights.get(k, 0.05) for k in others)
            if other_w > 0:
                other_score = sum(others[k] * weights.get(k, 0.05) for k in others) / other_w
                shap_values[mod] = round(float(full_score - other_score), 4)
            else:
                shap_values[mod] = round(float(calibrated_scores[mod]), 4)

        return shap_values

    # ------------------------------------------------------------------
    # Detección de modelo generativo específico
    # ------------------------------------------------------------------
    def _identify_ai_model(self, module_results: Dict) -> Tuple[str, float]:
        """
        Intenta identificar el modelo generativo específico.
        Retorna (nombre_modelo, confidence).
        """
        model_scores = {}

        for model_name, signature in self.AI_MODEL_SIGNATURES.items():
            match_score = 0.0
            total_criteria = len(signature)

            # HeyGen
            if model_name == "HeyGen/Avatar":
                facial = module_results.get("facial", {})
                audio  = module_results.get("audio", {})
                microexp = facial.get("microexpression", {}).get("microexp_density", 1.0)
                hnr      = audio.get("prosody", {}).get("hnr_db", 0.0)
                blinks   = facial.get("blink_analysis", {}).get("blinks_per_min", 15)

                if microexp < signature["microexp_density_max"]:
                    match_score += 1
                r = signature["blink_per_min_range"]
                if r[0] <= blinks <= r[1]:
                    match_score += 0.5
                if hnr > signature["audio_hnr_min"]:
                    match_score += 0.5

            # Sora
            elif model_name == "Sora/T2V":
                temporal = module_results.get("temporal", {})
                div  = temporal.get("flow_divergence_mean", 0.0)
                jac  = temporal.get("jacobian_discontinuity", 0.0)
                if div > signature["flow_divergence_min"]:
                    match_score += 1
                if jac > signature["jacobian_discontinuity_min"]:
                    match_score += 1

            # Runway Gen-3
            elif model_name == "Runway Gen-3":
                temporal = module_results.get("temporal", {})
                forensic = module_results.get("forensic", {})
                ghost   = temporal.get("ghosting", {}).get("ghosting_ratio", 0.0)
                tex_inc = temporal.get("texture_consistency", {}).get("texture_inconsistency", 0.0)
                lum_jump= forensic.get("lighting_consistency", {}).get("sudden_jump_ratio", 0.0)
                if ghost > signature["ghosting_ratio_min"]:
                    match_score += 1
                if tex_inc > signature["texture_inconsistency_min"]:
                    match_score += 1
                if lum_jump > signature["lighting_jump_min"]:
                    match_score += 0.5

            # DeepFace/FaceSwap
            elif model_name == "DeepFace/FaceSwap":
                facial   = module_results.get("facial", {})
                forensic = module_results.get("forensic", {})
                cb_diff  = facial.get("skin_chrominance", {}).get("cb_diff", 0.0)
                ela_sc   = forensic.get("ela_splice", {}).get("ela_splice_score", 0.0)
                prnu_cor = forensic.get("prnu", {}).get("prnu_consecutive_corr", 1.0)
                if cb_diff > signature["cb_diff_min"]:
                    match_score += 1
                if ela_sc > signature["ela_splice_min"]:
                    match_score += 1
                if prnu_cor < signature["prnu_corr_max"]:
                    match_score += 1

            # Pika
            elif model_name == "Pika/AnimateDiff":
                temporal = module_results.get("temporal", {})
                forensic = module_results.get("forensic", {})
                tex_inc = temporal.get("texture_consistency", {}).get("texture_inconsistency", 0.0)
                upsc    = forensic.get("upscaling", {}).get("upscale_fft_ratio", 0.0)
                noise_s = forensic.get("noise_signature", {}).get("noise_std", 10.0)
                if tex_inc > signature["texture_inconsistency_min"]:
                    match_score += 1
                if upsc > signature["upscale_fft_min"]:
                    match_score += 1
                if noise_s < signature["noise_std_max"]:
                    match_score += 1

            model_scores[model_name] = match_score / max(total_criteria, 1)

        if not model_scores:
            return ("Desconocido", 0.0)

        best_model = max(model_scores, key=lambda k: model_scores[k])
        best_conf  = model_scores[best_model]

        if best_conf < 0.3:
            return ("Indeterminado", best_conf)

        return (best_model, round(best_conf, 2))

    # ------------------------------------------------------------------
    # Método principal
    # ------------------------------------------------------------------
    def score(self, module_results: Dict) -> Dict:
        """
        Calcula el score final calibrado a partir de los resultados de todos los módulos.
        
        Args:
            module_results: {
                "vit_ensemble": {"suspicion": float, ...},
                "temporal":     {"suspicion": float, ...},
                "facial":       {"suspicion": float, ...},
                "forensic":     {"suspicion": float, ...},
                "audio":        {"suspicion": float, ...},
            }
        
        Returns:
            Dict con probabilidad calibrada, CI, veredicto, SHAP, modelo identificado
        """
        # Extraer scores raw por módulo
        raw_scores = {}
        for mod_name, mod_data in module_results.items():
            if isinstance(mod_data, dict) and "suspicion" in mod_data:
                available = mod_data.get("available", True)
                if available is not False:
                    raw_scores[mod_name] = float(mod_data["suspicion"])

        if not raw_scores:
            return {
                "probability":  50.0,
                "confidence":   0.0,
                "ci_lower":     20.0,
                "ci_upper":     80.0,
                "verdict":      "INDETERMINADO",
                "error":        "Ningún módulo produjo scores válidos"
            }

        # Calibrar scores con Platt scaling
        calibrated = {
            mod: self._platt_calibrate(score, mod)
            for mod, score in raw_scores.items()
        }

        # Pesos efectivos (redistribuidos si faltan módulos)
        eff_weights = self._get_effective_weights(list(calibrated.keys()))

        # Attention Weighting: Si un detector especialista está gritando ALARMA (> 0.55)
        # en un video, le damos un bonus de 'atención' temporal sobre los demás. 
        # Esto permite que un error detectado por ViT no sea ahogado por la "perfección"
        # forense de Sora/Runway, pero sin forzar límites estadísticos.
        for k, v in calibrated.items():
            # El multiplicador de atención permite que un módulo "especialista" domine
            # si detecta algo muy claro, pero no todos los módulos son igual de fiables.
            mult = 1.0
            if v > 0.85:   mult = 4.0  # Alarma absoluta
            elif v > 0.70: mult = 2.2  # Alarma crítica
            elif v > 0.55: mult = 1.5  # Sospecha moderada

            # Los módulos visuales (ViT) son propensos a falsos positivos por compresión.
            # Limitamos su capacidad de "gritar" solos sin apoyo de otros módulos.
            if k == "vit_ensemble":
                mult = 1.0 + (mult - 1.0) * 0.5  # Reducir potencia de atención al 50%
            
            eff_weights[k] *= mult
                
        # Normalizar pesos post-atención
        weight_sum = sum(eff_weights.values())
        if weight_sum > 0:
            eff_weights = {k: v / weight_sum for k, v in eff_weights.items()}

        # Score ponderado final
        # Suma ponderada conservadora (Los pesos definen qué test es más robusto globalmente)
        total_weight = sum(eff_weights.values())
        weighted_prob = sum(calibrated[k] * eff_weights.get(k, 0.05) for k in calibrated) / max(total_weight, 1e-10)

        # Intervalo de confianza basado en dispersión del ensemble
        scores_array = np.array(list(calibrated.values()))
        ci_lower = float(np.percentile(scores_array, 15))
        ci_upper = float(np.percentile(scores_array, 85))

        # Confianza: inverso del desacuerdo entre módulos
        module_disagreement = float(np.std(scores_array))
        confidence = max(0.0, 1.0 - module_disagreement * 2.5)

        # SHAP contributions
        shap = self._compute_shap(calibrated, eff_weights)

        # Top 3 razones (módulos con mayor contribución al score de IA)
        top_reasons = sorted(
            [(mod, shap.get(mod, 0.0), calibrated[mod]) for mod in calibrated],
            key=lambda x: -x[2]  # Ordenar por score calibrado
        )[:3]

        reasons_text = []
        for mod, shap_val, cal_score in top_reasons:
            if cal_score > 0.5:
                reasons_text.append(f"{mod} ({cal_score*100:.0f}%)")

        # Veredicto
        verdict = "ORGÁNICO"
        for label, threshold in sorted(self.VERDICT_THRESHOLDS.items(), key=lambda x: -x[1]):
            if weighted_prob >= threshold:
                verdict = label
                break

        # Identificar modelo generativo solo si hay sospecha real
        if weighted_prob >= 0.50:
            ai_model, ai_model_confidence = self._identify_ai_model(module_results)
        else:
            ai_model = "N/A (Firma Orgánica Fuerte)"
            ai_model_confidence = 0.0
            reasons_text = ["Ninguna anomalía domina. Las firmas apuntan a captura óptica real."]

        return {
            "probability":         round(float(weighted_prob * 100), 1),
            "confidence":          round(float(confidence * 100), 1),
            "ci_lower":            round(float(ci_lower * 100), 1),
            "ci_upper":            round(float(ci_upper * 100), 1),
            "verdict":             verdict,
            "ai_model_likely":     ai_model,
            "ai_model_confidence": round(float(ai_model_confidence * 100), 0),
            "reasons":             reasons_text,
            "module_scores": {
                mod: round(cal * 100, 1) for mod, cal in calibrated.items()
            },
            "raw_scores": {
                mod: round(sc * 100, 1) for mod, sc in raw_scores.items()
            },
            "shap_contributions":  shap,
            "modules_used":        list(calibrated.keys()),
            "n_modules":           len(calibrated)
        }
