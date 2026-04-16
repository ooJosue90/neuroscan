import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger("Nueroscan.AIImageDetector")

class AIImageDetector(nn.Module):
    """
    Hybrid Deep Learning AI Image Detector - Optimized for Forensics.
    
    Combines four specialized branches:
    1. Semantic (CLIP): High-level visual consistency.
    2. Structural/Texture (EfficientNet): Pixel-level artifact detection.
    3. Frequency (FFT): Detects periodic patterns from AI generators.
    4. Residual Noise: Captures non-natural noise signatures.
    """
    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-large-patch14",
        hidden_dim: int = 256,
        threshold: float = 0.5
    ):
        super().__init__()
        self.threshold = threshold
        
        # Branch 1: CLIP embeddings (Semántico de alto nivel)
        # CLIP ViT-L/14 embedding base dim = 768
        logger.info("Cargando CLIP: %s", clip_model_name)
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        self.clip_head = nn.Sequential(nn.Linear(768, hidden_dim), nn.GELU())
        
        # Branch 2: EfficientNet (Texturas y artefactos de bajo nivel)
        logger.info("Cargando EfficientNet-B4")
        self.efficientnet = None
        eff_out_dim = 1792 # Default para B4 ns
        
        try:
            import timm
            self.efficientnet = timm.create_model(
                "tf_efficientnet_b4_ns",
                pretrained=True,
                features_only=True,
            )
            eff_out_dim = self.efficientnet.feature_info.channels()[-1]
            logger.info("EfficientNet cargado con timm (out_dim=%s)", eff_out_dim)
        except Exception as e:
            logger.warning("timm no disponible o fallo de carga (%s). Fallback TorchHub.", e)
            self.efficientnet = torch.hub.load(
                "NVIDIA/DeepLearningExamples:torchhub",
                "nvidia_efficientnet_b4",
                pretrained=True,
            )
            eff_out_dim = 1792

        self.eff_head = nn.Sequential(nn.Linear(eff_out_dim, hidden_dim), nn.GELU())
        
        # Norm adjustment buffers: CLIP vs ImageNet
        # CLIP: mean=[0.4814, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2757]
        # ImageNet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        self.register_buffer("clip_mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer("clip_std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))
        self.register_buffer("eff_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("eff_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        # Branch 3: FFT features (Dominio frecuencial)
        self.fft_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(8),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, hidden_dim),
            nn.GELU(),
        )

        # Branch 4: Residual noise (x - blur)
        self.residual_conv = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(8),
            nn.Flatten(),
            nn.Linear(48 * 8 * 8, hidden_dim),
            nn.GELU(),
        )
        
        # Fusion Head: retorna LOGITS
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )
        
    def _adjust_norm_for_eff(self, x: torch.Tensor) -> torch.Tensor:
        """Convierte normalización de CLIP (input) a normalización de ImageNet (EfficientNet)."""
        # x esta en rango CLIP normalizado. Denormalizamos a [0, 1] y renornalizamos a ImageNet.
        x = x * self.clip_std + self.clip_mean 
        x = (x - self.eff_mean) / self.eff_std
        return x

    def extract_fft_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracción de features en el dominio de la frecuencia."""
        # Conversión a escala de grises
        gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        # FFT 2D
        fft = torch.fft.fft2(gray)
        fft_shift = torch.fft.fftshift(fft)
        magnitude = torch.log(torch.abs(fft_shift) + 1e-8).unsqueeze(1)
        return self.fft_conv(magnitude)
    
    def _extract_eff_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extrae features de EfficientNet con normalización corregida."""
        x_eff = self._adjust_norm_for_eff(x)
        out = self.efficientnet(x_eff)
        if isinstance(out, (list, tuple)):
            out = out[-1]
        if out.ndim == 4:
            out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return out

    def _extract_clip_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extrae embedding de CLIP robusto (768 para ViT-L/14)."""
        out = self.clip.get_image_features(pixel_values=x)
        if not isinstance(out, torch.Tensor):
            if hasattr(out, "pooler_output"):
                return out.pooler_output
            if hasattr(out, "last_hidden_state"):
                return out.last_hidden_state[:, 0]
        return out

    @staticmethod
    def extract_residual_map(x: torch.Tensor) -> torch.Tensor:
        """Mapa residual para capturar ruido/artefactos de síntesis."""
        blur = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
        return x - blur

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Inferencia completa sobre las cuatro ramas."""
        # RAMA 1: CLIP
        clip_feat = self.clip_head(self._extract_clip_features(pixel_values))
        
        # RAMA 2: EfficientNet
        eff_feat = self.eff_head(self._extract_eff_features(pixel_values))
        
        # RAMA 3: FFT
        fft_feat = self.extract_fft_features(pixel_values)

        # RAMA 4: Residual noise
        residual = self.extract_residual_map(pixel_values)
        residual_feat = self.residual_conv(residual)
        
        # FUSIÓN
        combined = torch.cat([clip_feat, eff_feat, fft_feat, residual_feat], dim=1)
        return self.classifier(combined)

    @torch.no_grad()
    def predict_proba(self, pixel_values: torch.Tensor) -> Dict[str, any]:
        """Salida calibrada para inferencia forense."""
        self.eval() # Asegura comportamiento consistente de BatchNorm/Dropout
        logits = self.forward(pixel_values)
        prob_ai = torch.sigmoid(logits)
        
        # Predicción basada en umbral
        is_ai = (prob_ai > self.threshold).float()
        
        return {
            "logits": logits,
            "prob_ai": prob_ai,
            "is_ai": is_ai,
            "threshold": self.threshold
        }

    def set_threshold(self, value: float):
        """Ajusta la sensibilidad del detector centralmente."""
        self.threshold = value
        logger.info("Umbral forense actualizado a: %.2f", value)

def get_processor():
    """Retorna el procesador estándar para el modelo CLIP seleccionado."""
    return CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
