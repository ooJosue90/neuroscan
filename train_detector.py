
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import logging

# ── Soporte AVIF/HEIC via pillow-avif-plugin si está disponible ──────────────
try:
    import pillow_avif
except ImportError:
    pass

# ── UTF-8 en Windows ──────────────────────────────────────────────────────────
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

from backend.models.modules.ai_image_detector import AIImageDetector, get_processor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Talos.Trainer")

# ─────────────────────────────────────────────────────────────────────────────
# REGLAS DE ETIQUETADO  (prioridad: primero IA, luego REAL)
# ─────────────────────────────────────────────────────────────────────────────
def _classify(filename: str):
    """
    Devuelve 1.0 (IA) / 0.0 (REAL) / None (ignorar).
    Reglas en orden de prioridad descendente.
    """
    name = filename.lower()

    # ── Imágenes IA ──────────────────────────────────────────────────────────
    ia_keywords = [
        "foto ia", "image ia", "img ia",
        "ai_", "_ai_", "_ai.", "deepfake",
        "generated", "synthetic", "fake",
    ]
    for kw in ia_keywords:
        if kw in name:
            return 1.0

    # ── Imágenes REALES ──────────────────────────────────────────────────────
    real_keywords = [
        "foto real", "image real", "img real",
        "real_", "_real_", "_real.",
        "whatsapp image",           # WhatsApp Image 2026-...
        "win_",                     # WIN_20260317_...
        "captura de pantalla",      # Captura de pantalla ...
        "pi7_passport",             # Pi7_Passport_Photo
        "image-removebg",           # Image-removebg-preview
        "photo_real",
    ]
    for kw in real_keywords:
        if kw in name:
            return 0.0

    return None  # sin etiqueta clara → omitir


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────
VALID_EXT = ('.png', '.jpg', '.jpeg', '.webp', '.avif', '.bmp', '.tiff')

class TalosDataset(Dataset):
    def __init__(self, data_dir: str, processor):
        self.processor = processor
        self.samples = []

        skipped = []
        for fname in sorted(os.listdir(data_dir)):
            if not fname.lower().endswith(VALID_EXT):
                continue
            label = _classify(fname)
            if label is None:
                skipped.append(fname)
                continue
            self.samples.append((os.path.join(data_dir, fname), label))

        n_ai   = sum(1 for _, l in self.samples if l == 1.0)
        n_real = sum(1 for _, l in self.samples if l == 0.0)
        logger.info("═" * 60)
        logger.info("DATASET CARGADO:")
        logger.info(f"  Total:      {len(self.samples)} imágenes")
        logger.info(f"  IA (fake):  {n_ai}")
        logger.info(f"  REAL:       {n_real}")
        logger.info(f"  Omitidas:   {len(skipped)}")
        if skipped:
            logger.info(f"  Archivos omitidos: {skipped[:10]}{'...' if len(skipped) > 10 else ''}")
        logger.info("═" * 60)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Error abriendo {img_path}: {e}. Usando imagen negra.")
            img = Image.new("RGB", (224, 224), color=0)

        pixel_values = self.processor(
            images=img, return_tensors="pt"
        )["pixel_values"].squeeze(0)
        return pixel_values, torch.tensor(label, dtype=torch.float32)

    def get_sampler(self) -> WeightedRandomSampler:
        """Balanceador de clases via WeightedRandomSampler."""
        labels = [l for _, l in self.samples]
        n_ai   = labels.count(1.0)
        n_real = labels.count(0.0)
        weight_ai   = 1.0 / n_ai   if n_ai   > 0 else 0
        weight_real = 1.0 / n_real if n_real > 0 else 0
        weights = [weight_ai if l == 1.0 else weight_real for l in labels]
        return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    logger.info(f"Dispositivo: {device.upper()}")

    # ── 1. Modelo ─────────────────────────────────────────────────────────────
    model = AIImageDetector(freeze_backbones=True).to(device)

    # Cargar pesos previos si existen (fine-tuning incremental)
    weights_path = "backend/models/weights/detector_ia_v1.pth"
    if os.path.exists(weights_path):
        try:
            state = torch.load(weights_path, map_location=device)
            model.load_state_dict(state, strict=False)
            logger.info(f"Pesos previos cargados desde: {weights_path}")
        except Exception as e:
            logger.warning(f"No se pudieron cargar pesos previos: {e}. Entrenando desde cero.")
    else:
        logger.info("No se encontraron pesos previos. Entrenando desde cero.")

    # ── 2. Dataset ────────────────────────────────────────────────────────────
    processor = get_processor()
    dataset = TalosDataset("backend/data", processor)

    if len(dataset) == 0:
        logger.error("No se encontraron imágenes válidas en backend/data. Abortando.")
        return

    # Sampler balanceado para evitar sesgo de clase
    sampler = dataset.get_sampler()
    loader  = DataLoader(dataset, batch_size=4, sampler=sampler, num_workers=0, pin_memory=(device == "cuda"))

    # ── 3. Optimización ───────────────────────────────────────────────────────
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4, weight_decay=1e-4
    )
    # Cosine Annealing para convergencia suave
    epochs    = 15
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler    = torch.amp.GradScaler() if device == "cuda" else None

    # ── 4. Bucle de entrenamiento ─────────────────────────────────────────────
    model.train()
    best_loss = float("inf")

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total   = 0

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            ctx = torch.amp.autocast(device_type=device, dtype=torch.float16 if device == "cuda" else torch.bfloat16)
            with ctx:
                logits = model(images)
                loss   = criterion(logits, labels)

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            preds    = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

        scheduler.step()
        avg_loss = running_loss / len(loader)
        accuracy = correct / total * 100 if total > 0 else 0
        lr_now   = scheduler.get_last_lr()[0]

        logger.info(
            f"Epoch [{epoch+1:02d}/{epochs}] | Loss: {avg_loss:.4f} | "
            f"Acc: {accuracy:.1f}% | LR: {lr_now:.2e}"
        )

        # Guardar mejor checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), weights_path)
            logger.info(f"  → Mejor modelo guardado (loss={best_loss:.4f})")

    # ── 5. Guardar modelo final ───────────────────────────────────────────────
    final_path = "backend/models/weights/detector_ia_v1.pth"
    torch.save(model.state_dict(), final_path)
    logger.info("═" * 60)
    logger.info(f"Entrenamiento completado.")
    logger.info(f"Mejor loss logrado: {best_loss:.4f}")
    logger.info(f"Modelo guardado en: {final_path}")
    logger.info("═" * 60)


if __name__ == "__main__":
    train()
