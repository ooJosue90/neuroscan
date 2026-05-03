import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger("Talos.AIImageDetector")

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
        threshold: float = 0.5,
        freeze_backbones: bool = True,
        use_amp: bool = True,
        compile_model: bool = False
    ):
        super().__init__()
        self.threshold = threshold
        self.use_amp = use_amp
        
        # Branch 1: CLIP embeddings (Semántico de alto nivel)
        # CLIP ViT-L/14 embedding base dim = 768
        logger.info("Cargando CLIP: %s", clip_model_name)
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        self.clip_head = nn.Sequential(nn.Linear(768, hidden_dim), nn.GELU())
        
        # Branch 2: EfficientNet (Texturas y artefactos de bajo nivel)
        self.efficientnet, eff_out_dim = self._load_efficientnet()
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
        
        # Fusion & Classifier: retorna LOGITS (B,)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Linear(512, 1)
        
        if freeze_backbones:
            self.freeze_backbones()

        if compile_model:
            self._try_compile()

    @staticmethod
    def _load_efficientnet():
        """Carga robusta de EfficientNet."""
        try:
            import timm
            logger.info("Cargando EfficientNet-B4 vía timm...")
            model = timm.create_model(
                "tf_efficientnet_b4_ns",
                pretrained=True,
                features_only=True,
            )
            out_dim = model.feature_info.channels()[-1]
            return model, out_dim
        except ImportError:
            raise RuntimeError("timm no instalado. Ejecuta: pip install timm")
        except Exception as e:
            raise RuntimeError(f"Fallo al cargar EfficientNet: {e}")

    def _try_compile(self):
        """Optimiza componentes vía torch.compile (PyTorch 2.0+)."""
        try:
            self.fft_conv = torch.compile(self.fft_conv)
            self.residual_conv = torch.compile(self.residual_conv)
            self.fusion = torch.compile(self.fusion)
            logger.info("torch.compile aplicado exitosamente.")
        except Exception as e:
            logger.warning("torch.compile no disponible o falló: %s", e)

    def freeze_backbones(self):
        """Congela CLIP y EfficientNet para entrenar solo las cabezas de fusión."""
        for param in self.clip.parameters():
            param.requires_grad = False
        if self.efficientnet:
            for param in self.efficientnet.parameters():
                param.requires_grad = False
        logger.info("Backbones (CLIP & EfficientNet) congelados. Solo las cabezas son entrenables.")

    def unfreeze_clip(self, layers_from_end: Optional[int] = None):
        """
        Descongela CLIP para fine-tuning avanzado.
        Args:
            layers_from_end: None -> todo, N -> últimas N capas del transformer.
        """
        if layers_from_end is None:
            for param in self.clip.parameters():
                param.requires_grad = True
            logger.info("CLIP totalmente descongelado.")
        else:
            # HuggingFace CLIPModel structure: vision_model.encoder.layers
            encoder_layers = self.clip.vision_model.encoder.layers
            total = len(encoder_layers)
            for i, layer in enumerate(encoder_layers):
                if i >= total - layers_from_end:
                    for param in layer.parameters():
                        param.requires_grad = True
            logger.info("Últimas %d capas de CLIP descongeladas.", layers_from_end)

    def unfreeze_efficientnet(self, stages_from_end: Optional[int] = None):
        """
        Descongela EfficientNet para fine-tuning.
        Args:
            stages_from_end: None -> todo, N -> últimos N bloques/stages.
        """
        if self.efficientnet is None: return
        
        if stages_from_end is None:
            for param in self.efficientnet.parameters():
                param.requires_grad = True
            logger.info("EfficientNet totalmente descongelado.")
        else:
            # Para EfficientNet de timm, solemos trabajar con 'blocks'
            if hasattr(self.efficientnet, 'blocks'):
                blocks = self.efficientnet.blocks
                total = len(blocks)
                for i, stage in enumerate(blocks):
                    if i >= total - stages_from_end:
                        for param in stage.parameters():
                            param.requires_grad = True
                logger.info("Últimos %d bloques de EfficientNet descongelados.", stages_from_end)
        
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
        fused = self.fusion(combined)
        return self.classifier(fused).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, pixel_values: torch.Tensor) -> Dict[str, any]:
        """Salida calibrada para inferencia forense con Mixed Precision."""
        self.eval()
        device = next(self.parameters()).device
        
        # Autocast adaptativo: fp16 en CUDA, bfloat16 en CPU
        amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
        
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=self.use_amp):
            logits = self(pixel_values) # Llama a __call__ para hooks
        
        # Convertimos a fp32 para sigmoid (estabilidad numérica)
        logits = logits.float()
        prob_ai = torch.sigmoid(logits)
        is_ai = (prob_ai > self.threshold).float()
        
        return {
            "logits": logits,
            "prob_ai": prob_ai,
            "is_ai": is_ai,
            "threshold": self.threshold
        }

    def count_trainable_params(self) -> Dict[str, int]:
        """Diagnóstico de parámetros entrenables."""
        components = {
            "clip": self.clip,
            "clip_head": self.clip_head,
            "efficientnet": self.efficientnet,
            "eff_head": self.eff_head,
            "fft_conv": self.fft_conv,
            "residual_conv": self.residual_conv,
            "fusion": self.fusion,
            "classifier": self.classifier
        }
        return {
            name: sum(p.numel() for p in mod.parameters() if p.requires_grad)
            for name, mod in components.items() if mod is not None
        }

    def estimate_vram_mb(self) -> float:
        """Estimación aproximada de VRAM en fp16."""
        total_params = sum(p.numel() for p in self.parameters())
        return (total_params * 2) / (1024 ** 2)

    def set_threshold(self, value: float):
        """Ajusta la sensibilidad del detector centralmente."""
        self.threshold = value
        logger.info("Umbral forense actualizado a: %.2f", value)

def get_processor():
    """Retorna el procesador estándar para el modelo CLIP seleccionado."""
    return CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
