"""
2_entrenar.py
═════════════
Entrena DeepLabV3+ con backbone MobileNetV3-Large sobre el dataset
generado por 1_generar_dataset.py.

Arquitectura:
  torchvision.models.segmentation.deeplabv3_mobilenet_v3_large
  - Preentrenado en COCO/ImageNet  (transfer learning)
  - Se re-entrena el clasificador para 2 clases: vegetacion / suelo
  - Segmentacion binaria pixel a pixel

Salida:
  modelo_surco.pth    ← pesos PyTorch (para continuar entrenando)
  modelo_surco.onnx   ← modelo exportado para inferencia rapida (RPi 5)

Uso:
  python 2_entrenar.py --data dataset
  python 2_entrenar.py --data dataset --epochs 30 --batch 4 --lr 1e-4
  python 2_entrenar.py --data dataset --resume modelo_surco.pth  # continuar
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF
import torchvision.models.segmentation as seg_models
import numpy as np
import cv2
import argparse
import random
import time
from pathlib import Path


# ── Hiperparametros por defecto ────────────────────────────────────
IMG_SIZE   = 320     # resolución de entrada (320 es óptimo para RPi 5)
EPOCHS     = 25
BATCH_SIZE = 4
LR         = 1e-4
VAL_FRAC   = 0.15    # fracción del dataset para validación
NUM_CLASSES = 2      # vegetación (0) y suelo/surco (1)
OUT_MODEL   = "modelo_surco"


# ══════════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════════

class SurcoDataset(Dataset):
    """Carga pares (imagen RGB, máscara binaria)."""

    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(self, data_dir: Path, img_size: int, augment: bool = True):
        self.img_dir  = data_dir / "images"
        self.mask_dir = data_dir / "masks"
        self.size     = img_size
        self.augment  = augment
        self.items    = sorted(self.img_dir.glob("*.png"))
        if not self.items:
            raise FileNotFoundError(f"No se encontraron imágenes en {self.img_dir}")
        print(f"  Dataset: {len(self.items)} pares en {data_dir}")

    def __len__(self):
        return len(self.items)

    def _augment(self, img, mask):
        """Aumentos geométricos + fotométricos ligeros."""
        # Flip horizontal
        if random.random() > 0.5:
            img  = TF.hflip(img)
            mask = TF.hflip(mask)

        # Rotación pequeña
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            img  = TF.rotate(img,  angle)
            mask = TF.rotate(mask, angle)

        # Brillo y contraste (solo imagen, no máscara)
        if random.random() > 0.4:
            img = TF.adjust_brightness(img, random.uniform(0.7, 1.4))
        if random.random() > 0.4:
            img = TF.adjust_contrast(img, random.uniform(0.8, 1.3))
        if random.random() > 0.3:
            img = TF.adjust_saturation(img, random.uniform(0.7, 1.4))

        return img, mask

    def __getitem__(self, idx):
        img_path  = self.items[idx]
        mask_path = self.mask_dir / img_path.name

        # Leer con OpenCV → convertir a tensor PIL-compatible
        bgr  = cv2.imread(str(img_path))
        rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # Redimensionar
        rgb  = cv2.resize(rgb,  (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        # Convertir a tensores PyTorch
        import torchvision.transforms.functional as TF
        from PIL import Image as PILImage
        img_pil  = PILImage.fromarray(rgb)
        mask_pil = PILImage.fromarray(mask)

        if self.augment:
            img_pil, mask_pil = self._augment(img_pil, mask_pil)

        # Normalizar imagen
        img_t  = TF.to_tensor(img_pil)
        img_t  = TF.normalize(img_t, self.MEAN, self.STD)

        # Máscara: 0=vegetación, 1=suelo
        mask_np = np.array(mask_pil)
        mask_t  = torch.from_numpy((mask_np > 127).astype(np.int64))

        return img_t, mask_t


# ══════════════════════════════════════════════════════════════════
#  MODELO
# ══════════════════════════════════════════════════════════════════

def crear_modelo(num_classes: int = 2, pretrained: bool = True):
    """
    DeepLabV3+ con backbone MobileNetV3-Large.
    Sustituye el clasificador final por uno de num_classes clases.
    """
    weights = seg_models.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model   = seg_models.deeplabv3_mobilenet_v3_large(weights=weights)

    # Reemplazar el clasificador (último bloque) para 2 clases
    in_ch  = model.classifier[-1].in_channels
    model.classifier[-1] = nn.Conv2d(in_ch, num_classes, kernel_size=1)

    # Reemplazar también el clasificador auxiliar si existe
    if hasattr(model, "aux_classifier") and model.aux_classifier is not None:
        in_ch_aux = model.aux_classifier[-1].in_channels
        model.aux_classifier[-1] = nn.Conv2d(in_ch_aux, num_classes, kernel_size=1)

    return model


# ══════════════════════════════════════════════════════════════════
#  MÉTRICAS
# ══════════════════════════════════════════════════════════════════

def calc_iou(pred_mask, true_mask, cls=1):
    pred = (pred_mask == cls)
    true = (true_mask == cls)
    inter = (pred & true).sum().item()
    union = (pred | true).sum().item()
    return inter / (union + 1e-6)


# ══════════════════════════════════════════════════════════════════
#  ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════

def entrenar(args):
    data_dir = Path(args.data)
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Dispositivo : {device}")
    print(f"[INFO] Datos       : {data_dir}")
    print(f"[INFO] Img size    : {args.size}x{args.size}")
    print(f"[INFO] Epochs      : {args.epochs}   Batch: {args.batch}   LR: {args.lr}\n")

    # ── Dataset ───────────────────────────────────────────────────
    full_ds  = SurcoDataset(data_dir, args.size, augment=True)
    val_n    = max(1, int(len(full_ds) * VAL_FRAC))
    train_n  = len(full_ds) - val_n
    train_ds, val_ds = random_split(full_ds, [train_n, val_n],
                                    generator=torch.Generator().manual_seed(42))
    # Sin augment en validación
    val_ds.dataset.augment = False

    train_loader = DataLoader(train_ds, batch_size=args.batch,
                               shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch,
                               shuffle=False, num_workers=2, pin_memory=True)

    print(f"  Train: {train_n}  |  Val: {val_n}")

    # ── Modelo ────────────────────────────────────────────────────
    model = crear_modelo(NUM_CLASSES, pretrained=not bool(args.resume))
    model = model.to(device)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        print(f"[INFO] Reanudando desde: {args.resume}")

    # ── Optimizador + scheduler ───────────────────────────────────
    # Diferente LR para backbone (ajuste fino) vs cabezal (reentrenamiento)
    backbone_params = list(model.backbone.parameters())
    head_params     = [p for n, p in model.named_parameters()
                       if not n.startswith("backbone")]
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": head_params,     "lr": args.lr},
    ], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Pérdida: CrossEntropy + peso para clase suelo (puede ser menos frecuente)
    class_weights = torch.tensor([0.4, 0.6], device=device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    best_iou = 0.0

    for epoch in range(1, args.epochs + 1):
        # ── Entrenamiento ─────────────────────────────────────────
        model.train()
        t0 = time.time()
        total_loss = 0.0

        for imgs, masks in train_loader:
            imgs  = imgs.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            out  = model(imgs)["out"]
            loss = criterion(out, masks)
            if hasattr(model, "aux_classifier") and model.training:
                # El modelo también puede dar salida aux durante training
                pass
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        # ── Validación ────────────────────────────────────────────
        model.eval()
        ious = []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs  = imgs.to(device)
                out   = model(imgs)["out"]
                preds = out.argmax(dim=1).cpu()
                for p, m in zip(preds, masks):
                    ious.append(calc_iou(p, m, cls=1))

        mean_iou = float(np.mean(ious))
        elapsed  = time.time() - t0

        print(f"  Epoch {epoch:3d}/{args.epochs}  "
              f"loss={avg_loss:.4f}  IoU_suelo={mean_iou:.4f}  "
              f"({elapsed:.1f}s)")

        # ── Guardar mejor modelo ───────────────────────────────────
        if mean_iou > best_iou:
            best_iou = mean_iou
            torch.save(model.state_dict(), f"{OUT_MODEL}.pth")
            print(f"  [*] Nuevo mejor modelo guardado  (IoU={best_iou:.4f})")

    print(f"\n[OK] Entrenamiento completo. Mejor IoU suelo: {best_iou:.4f}")

    # ── Exportar a ONNX ───────────────────────────────────────────
    print("[INFO] Exportando a ONNX...")
    model.load_state_dict(torch.load(f"{OUT_MODEL}.pth", map_location=device))
    model.eval()
    dummy = torch.randn(1, 3, args.size, args.size, device=device)
    torch.onnx.export(
        model, dummy, f"{OUT_MODEL}.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    print(f"[OK] Exportado: {OUT_MODEL}.onnx")
    print(f"\nSiguiente paso -> python 3_inferir_video.py video.mp4 "
          f"--model {OUT_MODEL}.onnx --size {args.size}")


def main():
    parser = argparse.ArgumentParser(
        description="Entrena DeepLabV3+MobileNetV3 para segmentacion de surco de cana")
    parser.add_argument("--data",   required=True, help="Carpeta del dataset")
    parser.add_argument("--epochs", type=int,   default=EPOCHS)
    parser.add_argument("--batch",  type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",     type=float, default=LR)
    parser.add_argument("--size",   type=int,   default=IMG_SIZE,
                        help="Tamaño de imagen de entrada (default 320)")
    parser.add_argument("--resume", default=None,
                        help="Continuar desde checkpoint .pth")
    args = parser.parse_args()
    entrenar(args)


if __name__ == "__main__":
    main()
