from pathlib import Path
from typing import Optional, Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn


DEFAULT_ROOT_DIR = Path("../data")
DEFAULT_IMAGE_SIZE = (224, 224)
def _format_interferogram_date(value: object) -> str:
    if value is None:
        return "N/A"

    try:
        if pd.isna(value):
            return "N/A"
    except TypeError:
        pass

    text = str(value).strip()

    if text == "" or text.lower() in {"nan", "nat", "none"}:
        return "N/A"

    if len(text) == 8 and text.isdigit():
        date = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
        return "N/A" if pd.isna(date) else date.strftime("%Y-%m-%d")

    date = pd.to_datetime(text, errors="coerce")
    if pd.notna(date):
        return date.strftime("%Y-%m-%d")

    return text


def _resolve_path(root_dir: str | Path, value: object) -> Optional[Path]:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass

    text = str(value).strip()

    if text == "" or text.lower() in {"nan", "nat", "none"}:
        return None

    path = Path(text)
    if path.is_absolute():
        return path

    return Path(root_dir) / path


def _load_rgb_image(path: Path, image_size: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    image = image.resize(image_size, resample=RESAMPLE_BILINEAR)
    return np.asarray(image)


def _load_binary_mask(path: Path, image_size: tuple[int, int]) -> np.ndarray:
    mask = Image.open(path).convert("L")
    mask = mask.resize(image_size, resample=RESAMPLE_NEAREST)
    mask_np = np.asarray(mask, dtype=np.float32)

    return (mask_np > 0).astype(np.uint8)


def _as_2d_array(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value)
    arr = np.squeeze(arr)

    if arr.ndim != 2:
        raise ValueError(
            f"{name} debería tener forma 2D después de squeeze, "
            f"pero tiene forma {arr.shape}."
        )

    return arr


def _compute_binary_metrics(real_mask: np.ndarray, pred_mask: np.ndarray) -> dict[str, float | int]:
    real = real_mask.astype(bool)
    pred = pred_mask.astype(bool)

    tp = int(np.logical_and(real, pred).sum())
    fp = int(np.logical_and(~real, pred).sum())
    fn = int(np.logical_and(real, ~pred).sum())
    tn = int(np.logical_and(~real, ~pred).sum())

    eps = 1e-8

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "iou": float(iou),
        "dice": float(dice),
        "real_area_px": int(real.sum()),
        "pred_area_px": int(pred.sum()),
    }


def show_metadata_item(
    model: nn.Module,
    metadata_csv: str | Path,
    root_dir: str | Path = DEFAULT_ROOT_DIR,
    item_id: Optional[str] = None,
    item_index: Optional[int] = None,
    threshold: float = 0.5,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    show_probability: bool = True,
    show_overlay: bool = True,
    return_data: bool = False,
) -> Optional[dict[str, Any]]:
    import matplotlib.pyplot as plt

    if (item_id is None) == (item_index is None):
        raise ValueError("Debes indicar solo uno: item_id o item_index.")

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold debe estar entre 0 y 1.")

    metadata_csv = Path(metadata_csv)
    root_dir = Path(root_dir)

    if not metadata_csv.exists():
        raise FileNotFoundError(f"No existe metadata_csv: {metadata_csv}")

    df = pd.read_csv(metadata_csv)

    required_columns = {"id", "img_path"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Faltan columnas obligatorias en metadata.csv: {missing_columns}")

    if item_id is not None:
        matches = df[df["id"].astype(str) == str(item_id)]

        if matches.empty:
            raise ValueError(f"No se encontró ningún ítem con id={item_id}")

        if len(matches) > 1:
            print(f"Advertencia: hay {len(matches)} filas con id={item_id}. Se usará la primera.")

        row = matches.iloc[0]

    else:
        if item_index is None:
            raise ValueError("item_index no puede ser None.")

        if item_index < 0 or item_index >= len(df):
            raise IndexError(f"item_index fuera de rango. Rango válido: 0..{len(df) - 1}")

        row = df.iloc[item_index]

    img_path = _resolve_path(root_dir, row.get("img_path"))

    if img_path is None:
        raise ValueError("La fila seleccionada no tiene img_path válido.")

    if not img_path.exists():
        raise FileNotFoundError(f"No existe la imagen: {img_path}")

    mask_path = _resolve_path(root_dir, row.get("mask_path", None))
    has_real_mask = mask_path is not None and mask_path.exists()

    result = predict_image(
        model,
        img_path,
        threshold=threshold,
        image_size=image_size,
    )

    image_np = _load_rgb_image(img_path, image_size)

    if has_real_mask:
        real_mask_np = _load_binary_mask(mask_path, image_size)
    else:
        real_mask_np = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)

    if "prob_mask" in result:
        prob_mask = _as_2d_array(result["prob_mask"], "prob_mask").astype(np.float32)
        pred_mask = (prob_mask >= threshold).astype(np.uint8)
    elif "pred_mask" in result:
        pred_mask = _as_2d_array(result["pred_mask"], "pred_mask")
        pred_mask = (pred_mask > 0).astype(np.uint8)
        prob_mask = pred_mask.astype(np.float32)
    else:
        raise KeyError("predict_image debe devolver al menos 'prob_mask' o 'pred_mask'.")

    if pred_mask.shape != real_mask_np.shape:
        raise ValueError(
            f"La máscara predicha tiene forma {pred_mask.shape}, "
            f"pero la máscara real tiene forma {real_mask_np.shape}."
        )

    primary_date = _format_interferogram_date(row.get("primary_date", None))
    secondary_date = _format_interferogram_date(row.get("secondary_date", None))

    prob_max = float(np.max(prob_mask))
    prob_mean = float(np.mean(prob_mask))
    pred_area = int(pred_mask.sum())
    real_area = int(real_mask_np.sum())

    metrics = None
    if has_real_mask:
        metrics = _compute_binary_metrics(real_mask_np, pred_mask)

    print("=" * 70)
    print(f"ID: {row.get('id', 'N/A')}")
    print(f"Índice en metadata: {row.name}")
    print(f"FrameID: {row.get('frameID', 'N/A')}")
    print(f"Fechas del interferograma: {primary_date} -> {secondary_date}")
    print(f"Imagen: {img_path}")

    if has_real_mask:
        print(f"Máscara real: {mask_path}")
    else:
        print("Máscara real: no disponible")

    print(f"Threshold: {threshold:.3f}")
    print(f"Probabilidad máxima: {prob_max:.4f}")
    print(f"Probabilidad media: {prob_mean:.4f}")
    print(f"Área real: {real_area} px")
    print(f"Área predicha: {pred_area} px")

    if metrics is not None:
        print(f"IoU: {metrics['iou']:.4f}")
        print(f"Dice: {metrics['dice']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")

    n_cols = 4 if show_probability else 3
    if show_overlay:
        n_cols += 1

    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    if n_cols == 1:
        axes = [axes]

    col = 0

    axes[col].imshow(image_np)
    axes[col].set_title("Imagen a color")
    axes[col].axis("off")
    col += 1

    axes[col].imshow(real_mask_np, cmap=MASK_CMAP, vmin=0, vmax=1, interpolation="nearest")
    axes[col].set_title("Máscara real" if has_real_mask else "Máscara real\nno disponible")
    axes[col].axis("off")
    col += 1

    if show_probability:
        im = axes[col].imshow(prob_mask, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
        axes[col].set_title(f"Mapa de probabilidad\nmax={prob_max:.3f}")
        axes[col].axis("off")
        fig.colorbar(im, ax=axes[col], fraction=0.046, pad=0.04)
        col += 1

    axes[col].imshow(pred_mask, cmap=MASK_CMAP, vmin=0, vmax=1, interpolation="nearest")
    axes[col].set_title(f"Máscara predicha\nthreshold={threshold:.2f}")
    axes[col].axis("off")
    col += 1

    if show_overlay:
        axes[col].imshow(image_np)
        axes[col].imshow(pred_mask, cmap=MASK_CMAP, vmin=0, vmax=1, alpha=0.45, interpolation="nearest")
        axes[col].set_title("Predicción sobre imagen")
        axes[col].axis("off")

    title = (
        f"ID: {row.get('id', 'N/A')} | "
        f"FrameID: {row.get('frameID', 'N/A')} | "
        f"{primary_date} -> {secondary_date}"
    )

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()

    output = {
        "row": row,
        "img_path": img_path,
        "mask_path": mask_path if has_real_mask else None,
        "image": image_np,
        "real_mask": real_mask_np,
        "prob_mask": prob_mask,
        "pred_mask": pred_mask,
        "prob_max": prob_max,
        "prob_mean": prob_mean,
        "metrics": metrics,
    }

    if return_data:
        return output

    return None