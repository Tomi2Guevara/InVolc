"""Evaluación e inferencia para `modelo_v3.py`.

Permite:
- cargar un checkpoint entrenado
- predecir una imagen individual
- evaluar un split completo desde `metadata.csv`
- reportar loss, IoU y Dice
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple, Mapping
import torchvision.transforms.functional as TF
import numpy as np
import torch
import torch.nn as nn
import random
from PIL import Image
from torch.utils.data import DataLoader
from matplotlib.colors import ListedColormap
from modelo_v3 import InVolcModel
from train_modelo_v3 import compute_iou_and_dice, load_metadata_splits
import helper_functions as hf
from custom_dataset import InVolcDataset, PairedTransform
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METADATA_CSV = PROJECT_ROOT / "data" / "metadata.csv"
DEFAULT_ROOT_DIR = PROJECT_ROOT / "data"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "notebooks" / "invol_v4_model.pth"
DEFAULT_IMAGE_SIZE = (224, 224)
MASK_CMAP = ListedColormap(["black", "orange"])
RESAMPLE_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR
RESAMPLE_NEAREST = getattr(Image, "Resampling", Image).NEAREST


def _to_binary_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim > 2:
        mask = np.squeeze(mask)
    return (mask > 0).astype(np.uint8)


def _format_interferogram_date(value: object) -> str:
    if pd.isna(value):
        return "N/A"

    text = str(value).strip()
    if len(text) == 8 and text.isdigit():
        return pd.to_datetime(text, format="%Y%m%d").strftime("%Y-%m-%d")
    return text


def _resolve_metadata_row(
    metadata_csv: str | Path,
    item_id: Optional[str] = None,
    item_index: Optional[int] = None,
) -> pd.Series:
    if (item_id is None) == (item_index is None):
        raise ValueError("Debes indicar exactamente uno: item_id o item_index.")

    df = pd.read_csv(metadata_csv)

    if item_id is not None:
        matches = df[df["id"].astype(str) == str(item_id)]
        if matches.empty:
            raise ValueError(f"No se encontró ningún ítem con id={item_id}")
        return matches.iloc[0]

    if item_index is None or item_index < 0 or item_index >= len(df):
        raise IndexError(f"item_index fuera de rango: 0..{len(df) - 1}")
    return df.iloc[item_index]


def load_model(
    checkpoint_path: str | Path,
    device: torch.device,
    encoder_name: str = "resnet34",
    encoder_weights: Optional[str] = None,
    in_channels: int = 3,
    classes: int = 1,
) -> InVolcModel:

    model = InVolcModel(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, Mapping):
        raise TypeError("El checkpoint no contiene un state_dict válido.")

    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    if not any(k.startswith("model.") for k in state_dict.keys()):
        state_dict = {f"model.{k}": v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model





def preprocess_image(
    image_path: str | Path,
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    device: Optional[torch.device] = None,
) -> torch.Tensor:

    image = Image.open(image_path).convert("RGB")
    image = image.resize(image_size, resample=RESAMPLE_BILINEAR)

    tensor = TF.to_tensor(image)
    tensor = TF.normalize(
        tensor,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    tensor = tensor.unsqueeze(0)

    if device is not None:
        tensor = tensor.to(device)

    return tensor


@torch.no_grad()
def predict_image(
    model: nn.Module,
    image_path: str | Path,
    threshold: float = 0.5,
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> Dict:
    """Predice la máscara de una sola imagen."""
    device = next(model.parameters()).device
    x = preprocess_image(image_path, image_size=image_size, device=device)

    logits = model(x)
    probs = torch.sigmoid(logits)
    pred_mask = (probs > threshold).float()

    return {
        "image_path": str(image_path),
        "prob_mask": probs.squeeze().cpu().numpy(),
        "pred_mask": pred_mask.squeeze().cpu().numpy(),
    }


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Evalúa un DataLoader completo."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        logits = model(images)
        loss = loss_fn(logits, masks)
        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()

        all_preds.append(preds.cpu())
        all_targets.append(masks.cpu())

    preds_cat = torch.cat(all_preds, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    iou, dice = compute_iou_and_dice(preds_cat, targets_cat)

    return {
        "loss": total_loss / max(1, total_samples),
        "iou": iou,
        "dice": dice,
    }


def build_test_loader(
    metadata_csv: str | Path,
    root_dir: str | Path = DEFAULT_ROOT_DIR,
    batch_size: int = 8,
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    num_workers: int = 0,
    subset: str = "test",
) -> DataLoader:

    eval_transform = PairedTransform(
        size=image_size,
        rotation=0.0,
        hflip_p=0.0,
        vflip_p=0.0,
        translate=(0.0, 0.0),
        scale=(1.0, 1.0),
        shear=0.0,
        use_color_jitter=False,
        use_blur=False,
        normalize=True,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    test_dataset = InVolcDataset(
        metadata_csv=str(metadata_csv),
        root_dir=root_dir,
        transform=eval_transform,
        cache=False,
        subset=subset,
    )

    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def visualize_prediction(
    result: Dict,
    image_path: str | Path,
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> None:
    """Muestra imagen + máscara predicha."""
    import matplotlib.pyplot as plt

    image = Image.open(image_path).convert("L")
    image = image.resize(DEFAULT_IMAGE_SIZE, resample=RESAMPLE_BILINEAR)
    image_np = np.array(image) / 255.0

    prob_mask = np.asarray(result["prob_mask"])
    pred_mask = _to_binary_mask(result["pred_mask"])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(image_np, cmap="gray")
    axes[0].set_title("Imagen")
    axes[0].axis("off")

    axes[1].imshow(prob_mask, cmap="hot", vmin=0.0, vmax=1.0, interpolation="nearest")
    axes[1].set_title(f"Mapa de probabilidad\nmax={prob_mask.max():.3f}")
    axes[1].axis("off")

    axes[2].imshow(pred_mask, cmap=MASK_CMAP, vmin=0, vmax=1, interpolation="nearest")
    axes[2].set_title("Máscara binaria")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Figura guardada en: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def visualize_comparison(
    result: Dict,
    image_path: str | Path,
    mask_path: str | Path,
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> None:
    """Muestra imagen original, máscara real y máscara predicha lado a lado."""
    import matplotlib.pyplot as plt

    image = Image.open(image_path).convert("L")
    image = image.resize(DEFAULT_IMAGE_SIZE, resample=RESAMPLE_BILINEAR)
    image_np = np.array(image) / 255.0

    # cargar máscara real
    real_mask = Image.open(mask_path).convert("L")
    real_mask = real_mask.resize(DEFAULT_IMAGE_SIZE, resample=RESAMPLE_NEAREST)
    real_mask_np = _to_binary_mask(np.array(real_mask, dtype=np.float32))

    prob_mask = np.asarray(result["prob_mask"])
    pred_mask = _to_binary_mask(result["pred_mask"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image_np, cmap="gray")
    axes[0].set_title("Imagen")
    axes[0].axis("off")

    axes[1].imshow(real_mask_np, cmap=MASK_CMAP, vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title("Máscara real")
    axes[1].axis("off")

    axes[2].imshow(pred_mask, cmap=MASK_CMAP, vmin=0, vmax=1, interpolation="nearest")
    axes[2].set_title(f"Máscara predicha\nprob max={prob_mask.max():.3f}")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Figura guardada en: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def show_metadata_item(
    model: nn.Module,
    metadata_csv: str | Path,
    root_dir: str | Path = DEFAULT_ROOT_DIR,
    item_id: Optional[str] = None,
    item_index: Optional[int] = None,
    threshold: float = 0.5,
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> Dict[str, object]:
    """Muestra un ítem del metadata con imagen, máscara real y predicción."""
    import matplotlib.pyplot as plt

    row = _resolve_metadata_row(metadata_csv, item_id=item_id, item_index=item_index)
    root_dir = Path(root_dir)
    img_path = root_dir / str(row["img_path"])

    mask_path = None
    mask_value = row.get("mask_path", None)
    if pd.notna(mask_value):
        mask_rel = str(mask_value).strip()
        if mask_rel and mask_rel.lower() != "nan":
            mask_path = root_dir / mask_rel

    result = predict_image(
        model,
        img_path,
        threshold=threshold,
        image_size=DEFAULT_IMAGE_SIZE,
    )

    image = Image.open(img_path).convert("RGB")
    image = image.resize(DEFAULT_IMAGE_SIZE, resample=RESAMPLE_BILINEAR)
    image_np = np.array(image)

    if mask_path is not None and mask_path.exists():
        real_mask = Image.open(mask_path).convert("L")
        real_mask = real_mask.resize(DEFAULT_IMAGE_SIZE, resample=RESAMPLE_NEAREST)
        real_mask_np = _to_binary_mask(np.array(real_mask, dtype=np.float32))
    else:
        real_mask_np = np.zeros((DEFAULT_IMAGE_SIZE[1], DEFAULT_IMAGE_SIZE[0]), dtype=np.uint8)

    pred_mask = _to_binary_mask(result["pred_mask"])
    primary_date = _format_interferogram_date(row.get("primary_date"))
    secondary_date = _format_interferogram_date(row.get("secondary_date"))

    print(f"ID: {row.get('id')}")
    print(f"FrameID: {row.get('frameID', 'N/A')}")
    print(f"Fechas del interferograma: {primary_date} -> {secondary_date}")
    if mask_path is None:
        print("Máscara real: no disponible")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image_np)
    axes[0].set_title("Imagen a color")
    axes[0].axis("off")

    axes[1].imshow(real_mask_np, cmap=MASK_CMAP, vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title("Máscara real")
    axes[1].axis("off")

    axes[2].imshow(pred_mask, cmap=MASK_CMAP, vmin=0, vmax=1, interpolation="nearest")
    axes[2].set_title(f"Máscara predicha\nprob max={np.asarray(result['prob_mask']).max():.3f}")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Figura guardada en: {save_path}")
    if show:
        plt.show()
    plt.close(fig)

    return {
        "id": row.get("id"),
        "frameID": row.get("frameID"),
        "primary_date": primary_date,
        "secondary_date": secondary_date,
        "image_path": str(img_path),
        "mask_path": str(mask_path) if mask_path is not None else None,
    }


def show_random_positive_samples(
    model: nn.Module,
    metadata_csv: str | Path,
    root_dir: str | Path = DEFAULT_ROOT_DIR,
    n: int = 5,
    threshold: float = 0.5,
    seed: int = 42,
    subset: str = "test",
) -> None:

    hf.set_seed(seed)
    root_dir = Path(root_dir)

    df = pd.read_csv(metadata_csv)

    if "subset" in df.columns:
        df = df[df["subset"] == subset]

    df = df[(df["label_bin"] == 1) & (df["mask_path"].notna())].reset_index(drop=True)

    if len(df) == 0:
        print("No se encontraron muestras positivas con máscara.")
        return

    selected = df.sample(n=min(n, len(df)), random_state=seed)

    for _, row in selected.iterrows():
        img_path = root_dir / row["img_path"]
        mask_path = root_dir / row["mask_path"]

        print(f"Procesando: {img_path}")
        print(f"Máscara: {mask_path}")

        result = predict_image(
            model,
            img_path,
            threshold=threshold,
            image_size=DEFAULT_IMAGE_SIZE,
        )

        visualize_comparison(result, img_path, mask_path)

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluación de modelo_v3.py")
    parser.add_argument("--metadata-csv", type=str, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--root-dir", type=str, default=str(DEFAULT_ROOT_DIR))
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--image", type=str, default=None, help="Ruta a una imagen para predicción individual")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-visualize", action="store_true")
    parser.add_argument("--show-random", type=int, default=10, help="Número de imágenes positivas aleatorias a mostrar (con máscara real). Si >0 se mostrará y se terminará.")
    args = parser.parse_args()

    hf.set_seed(args.seed)
    device = hf.set_device()
    model = load_model(args.checkpoint, device=device)

    loss_fn = nn.BCEWithLogitsLoss()

    if args.image is not None:
        result = predict_image(model, args.image, threshold=args.threshold)
        print(f"Imagen: {result['image_path']}")
        print(f"Prob max: {np.asarray(result['prob_mask']).max():.4f}")
        print(f"Pixels positivos: {(np.asarray(result['pred_mask']) > 0).sum()}")
        if not args.no_visualize:
            visualize_prediction(result, args.image)
        return

    if args.show_random and args.show_random > 0:
        show_random_positive_samples(
            model=model,
            metadata_csv=args.metadata_csv,
            root_dir=args.root_dir,
            n=args.show_random,
            threshold=args.threshold,
            seed=args.seed,
        )
        return

    loader = build_test_loader(
        metadata_csv=args.metadata_csv,
        root_dir=args.root_dir,
        batch_size=args.batch_size,
    )
    metrics = evaluate_loader(model, loader, loss_fn, device=device, threshold=args.threshold)

    print("\nResultados de evaluación")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  IoU:  {metrics['iou']:.4f}")
    print(f"  Dice: {metrics['dice']:.4f}")


if __name__ == "__main__":
    main()

