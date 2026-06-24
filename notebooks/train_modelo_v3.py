"""Entrenamiento base y limpio para segmentación binaria con U-Net.

- Lee imágenes y máscaras desde `metadata.csv`
- Usa una U-Net simple definida en `modelo_v3.py`
- Entrena con `nn.BCEWithLogitsLoss()`
- Calcula IoU y Dice solo como métricas
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from diceBCELoss import BCEDiceLoss
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

import helper_functions as hf
from modelo_v3 import InVolcModel
from custom_dataset import PairedTransform, InVolcDataset
from sklearn.model_selection import train_test_split


# ==========================================================
# CONFIGURACIÓN
# ==========================================================
METADATA_CSV = "metadata_sample_03.csv"
ROOT_DIR = Path("../data")
CHECKPOINT_PATH = Path("invol_v4_model.pth")
HISTORY_PATH = Path("training_history_binary_segmentation_1506.csv")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
PATIENCE = 8
RANDOM_SEED = 42
NUM_WORKERS = 0  # Windows-safe


# ==========================================================
# UTILIDADES
# ==========================================================

def _as_dataframe(metadata_or_path: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(metadata_or_path, pd.DataFrame):
        return metadata_or_path.copy()
    return pd.read_csv(metadata_or_path)


def load_metadata_splits(
    metadata_csv: str | Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Carga metadata y devuelve train/test.

    Si existe la columna `subset`, respeta esa división.
    Si no existe, hace split estratificado por `label_bin`.
    """
    df = pd.read_csv(METADATA_CSV)

    df_valid = df[~((df["label_bin"] == 1) & (df["mask_path"].isna()))].reset_index(drop=True)

    if "frameID" in df_valid.columns:
        groups = df_valid["frameID"].astype(str)
    else:
        groups = df_valid["img_path"].apply(lambda p: Path(str(p)).stem.split("_")[0])

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    train_idx, val_idx = next(sgkf.split(df_valid, df_valid["label_bin"], groups))

    df_valid["subset"] = "unused"
    df_valid.loc[train_idx, "subset"] = "train"
    df_valid.loc[val_idx, "subset"] = "test"

    temp_meta = Path("metadata_with_subset.csv")
    df_valid.to_csv(temp_meta, index=False)

    if "label_bin" not in df.columns:
        raise ValueError("El metadata debe contener la columna 'label_bin' o 'subset'.")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label_bin"],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)



# ==========================================================
# MÉTRICAS
# ==========================================================
@torch.no_grad()
def compute_iou_and_dice(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> Tuple[float, float]:
    """Calcula IoU y Dice sobre predicciones binarias."""
    preds = preds.float()
    targets = targets.float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection

    iou = (intersection + eps) / (union + eps)
    dice = (2 * intersection + eps) / (preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + eps)

    return float(iou.mean().item()), float(dice.mean().item())


# ==========================================================
# TRAIN / TEST
# ==========================================================
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for batch in tqdm(dataloader, desc="Train", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        all_preds.append(preds.detach().cpu())
        all_targets.append(masks.detach().cpu())

    preds_cat = torch.cat(all_preds, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    iou, dice = compute_iou_and_dice(preds_cat, targets_cat)

    return {
        "loss": running_loss / len(dataloader.dataset),
        "iou": iou,
        "dice": dice,
    }


@torch.no_grad()
def test_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for batch in tqdm(dataloader, desc="Test", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        logits = model(images)
        loss = loss_fn(logits, masks)
        running_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        all_preds.append(preds.cpu())
        all_targets.append(masks.cpu())

    preds_cat = torch.cat(all_preds, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    iou, dice = compute_iou_and_dice(preds_cat, targets_cat)

    return {
        "loss": running_loss / len(dataloader.dataset),
        "iou": iou,
        "dice": dice,
    }


# ==========================================================
# FIT COMPLETO
# ==========================================================
def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    num_epochs: int = NUM_EPOCHS,
    patience: int = PATIENCE,
    checkpoint_path: str | Path = CHECKPOINT_PATH,
) -> pd.DataFrame:
    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0
    history: List[Dict[str, float]] = []
    checkpoint_path = Path(checkpoint_path)

    for epoch in range(num_epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        test_metrics = test_one_epoch(model, test_loader, loss_fn, device)

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "train_iou": train_metrics["iou"],
                "train_dice": train_metrics["dice"],
                "test_loss": test_metrics["loss"],
                "test_iou": test_metrics["iou"],
                "test_dice": test_metrics["dice"],
            }
        )

        print(
            f"Epoch {epoch + 1:03d}/{num_epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} IoU: {train_metrics['iou']:.4f} Dice: {train_metrics['dice']:.4f} | "
            f"Test Loss: {test_metrics['loss']:.4f} IoU: {test_metrics['iou']:.4f} Dice: {test_metrics['dice']:.4f}"
        )

        if test_metrics["loss"] < best_val_loss:
            best_val_loss = test_metrics["loss"]
            best_epoch = epoch + 1
            patience_counter = 0
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping en epoch {epoch + 1}. Mejor epoch: {best_epoch}, mejor val loss: {best_val_loss:.4f}")
            break

    history_df = pd.DataFrame(history)
    history_df.to_csv(HISTORY_PATH, index=False)
    return history_df


# ==========================================================
# MAIN
# ==========================================================
def main() -> None:
    hf.set_seed(RANDOM_SEED)
    device = hf.set_device()

    # Cargar metadata y filtrar positivos sin máscara (InVolcDataset requiere mask_path para label_bin==1)
    df = pd.read_csv(METADATA_CSV)
    df_valid = df[~((df['label_bin'] == 1) & (df['mask_path'].isna()))].reset_index(drop=True)

    # Split reproducible estratificado
    train_idx, test_idx = train_test_split(
        df_valid.index,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=df_valid.loc[df_valid.index, 'label_bin']
    )

    df_valid['subset'] = 'unused'
    df_valid.loc[train_idx, 'subset'] = 'train'
    df_valid.loc[test_idx, 'subset'] = 'test'

    temp_meta = Path('metadata_with_subset.csv')
    df_valid.to_csv(temp_meta, index=False)

    print(f"Train samples: {(df_valid['subset']=='train').sum()}")
    print(f"Test samples:  {(df_valid['subset']=='test').sum()}")

    # Transforms: usar PairedTransform (geométricas para máscara, fotométricas para imagen)
    train_transform = PairedTransform(
        size=IMAGE_SIZE,
        rotation=15.0,
        hflip_p=0.5,
        vflip_p=0.5,
        use_color_jitter=True,
        use_blur=False,
        normalize=True,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    test_transform = PairedTransform(
        size=IMAGE_SIZE,
        rotation=0.0,
        hflip_p=0.0,
        vflip_p=0.0,
        use_color_jitter=False,
        use_blur=False,
        normalize=True,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    train_dataset = InVolcDataset(metadata_csv=str(temp_meta),
                                  root_dir=str(ROOT_DIR),
                                  transform=train_transform,
                                  cache=False,
                                  subset='train')
    test_dataset = InVolcDataset(metadata_csv=str(temp_meta),
                                 root_dir=str(ROOT_DIR),
                                 transform=test_transform,
                                 cache=False,
                                 subset='test')

    print(f"Train dataset usable samples: {len(train_dataset)}")
    print(f"Test dataset usable samples:  {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    # Usar 3 canales (RGB) ya que PairedTransform trabaja sobre imágenes RGB
    model = InVolcModel(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    pos_weight = hf.estimate_pos_weight(train_dataset).to(device)
    print(f"pos_weight: {pos_weight.item():.2f}")

    loss_fn = BCEDiceLoss(pos_weight=pos_weight, bce_weight=0.5, dice_weight=0.5)

    history_df = fit_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        num_epochs=NUM_EPOCHS,
        patience=PATIENCE,
        checkpoint_path=CHECKPOINT_PATH,
    )

    print("\nEntrenamiento finalizado")
    print(f"Checkpoint guardado en: {CHECKPOINT_PATH}")
    print(f"Historial guardado en: {HISTORY_PATH}")
    print(history_df.tail())


if __name__ == "__main__":
    main()

