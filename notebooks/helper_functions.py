import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import random
import torch
import numpy as np
import logging
from typing import Tuple
import requests
from PIL import Image
from torchvision import transforms
# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt





#configurar device para GPU si está disponible
def set_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def split_metadata(
    metadata_path: str = "../metadata_sample_01.csv",
    test_size: float = 0.2,
    random_state: int = 42,
    save_path: str = "../metadata_sample_01.csv",
    subset_column: str = "subset") -> pd.DataFrame:
    """
    Divide el metadata en train/test de forma reproducible y estratificada.

    Args:
        metadata_path: Ruta al CSV de entrada
        test_size: Proporción del conjunto de prueba (0.0-1.0)
        random_state: Semilla para reproducibilidad
        save_path: Ruta donde guardar el CSV modificado
        subset_column: Nombre de la columna a modificar

    Returns:
        DataFrame con la columna 'subset' actualizada

    Raises:
        FileNotFoundError: Si no existe el archivo
        ValueError: Si faltan columnas requeridas o parámetros inválidos
    """
    # Validar parámetros
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size debe estar entre 0.0 y 1.0, recibido: {test_size}")

    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {metadata_path}")

    try:
        df = pd.read_csv(metadata_path)
    except Exception as e:
        raise ValueError(f"Error al leer CSV: {str(e)}")

    # Verificar columnas requeridas
    required_cols = {"label_bin", subset_column}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV debe contener columnas: {required_cols}")

    # División estratificada
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label_bin"],
        random_state=random_state
    )

    # Modificar la columna existente
    df[subset_column] = df[subset_column].astype("object")
    df.loc[train_df.index, subset_column] = "train"
    df.loc[test_df.index, subset_column] = "test"

    # Guardar
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)

    logger.info(f"División completada:")
    logger.info(f"  Train: {len(train_df)} muestras")
    logger.info(f"  Test:  {len(test_df)} muestras")
    logger.info(f"  Archivo guardado en: {save_path}")

    return df


def validate_dataset(
    metadata: pd.DataFrame,
    root_dir: str = "../data",
    verbose: bool = True) -> Tuple[bool, list]:
    """
    Valida que todas las imágenes en el metadata existan en disco.

    Args:
        metadata: DataFrame con columna 'img_path'
        root_dir: Directorio raíz de las imágenes
        verbose: Si True, imprime resultados

    Returns:
        Tupla (todas_existen, lista_faltantes)
    """
    root_dir = Path(root_dir)
    missing = []

    for _, row in metadata.iterrows():
        path = root_dir / row["img_path"]
        if not path.exists():
            missing.append(str(path))

    if verbose:
        if missing:
            logger.warning(f"{len(missing)} archivos no encontrados:")
            for p in missing[:5]:
                logger.warning(f"  {p}")
            if len(missing) > 5:
                logger.warning(f"  ... y {len(missing) - 5} más")
        else:
            logger.info("✓ Todos los archivos existen y coinciden con el metadata.")

    return len(missing) == 0, missing


def set_seed(seed: int = 42) -> None:
    """
    Establece semillas para reproducibilidad en todas las librerías.

    Args:
        seed: Valor de la semilla
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    logger.info(f"Semilla establecida a: {seed}")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataset_stats(metadata: pd.DataFrame) -> dict:
    """
    Retorna estadísticas del dataset (distribución de clases, etc).

    Args:
        metadata: DataFrame con columnas 'label_bin' y 'subset'

    Returns:
        Diccionario con estadísticas
    """
    stats = {
        "total_muestras": len(metadata),
        "distribucion_clases": metadata["label_bin"].value_counts().to_dict(),
        "distribucion_subsets": metadata.get("subset", pd.Series()).value_counts().to_dict(),
    }
    return stats

def prepare_batch(batch, device, normalize_mask=True, require_float_labels=False):
    data = batch["image"]
    masks = batch["mask"]  # puede ser None
    labels = batch["label_bin"]  # o "label_raw" según tu tarea

    data = data.to(device, non_blocking=True)
    masks = masks.to(device, non_blocking=True) if masks is not None else None
    labels = labels.to(device)

    # print("data:", data.shape, data.dtype, data.device)
    # print("masks:", "None" if masks is None else f"{masks.shape} {masks.dtype} {masks.device}")
    # print("labels:", labels.shape, labels.dtype, labels.device)

    if masks is not None:
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)
        if normalize_mask and masks.max() > 1.0:
            masks = masks.float() / 255.0
        if masks.dtype != torch.float32:
            masks = masks.float()

    if require_float_labels and labels.dtype != torch.float32:
        labels = labels.float()


    return data, masks, labels

def preprocess_image(image_path, target_size=(255, 255)):
    """Abre y normaliza una imagen InSAR para la UNet."""
    img = Image.open(image_path).convert("RGB")  # o "RGB", según tu caso

    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),        # convierte a [0,1]
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])  # ajustar según lo entrenado
    ])

    tensor = transform(img)
    tensor = tensor.unsqueeze(0)  # → (1, C, H, W)
    return tensor


def plot_segmentation_results(dataloader, predictions):
    batch = next(iter(dataloader))
    images, masks, _ = prepare_batch(
        batch,
        device='cpu',
        normalize_mask=True,
        require_float_labels=False
    )

    seg_preds = predictions['seg_preds']

    for i in range(len(images)):
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(images[i].permute(1, 2, 0))
        axs[0].set_title('Input Image')
        axs[0].axis('off')

        axs[1].imshow(masks[i].squeeze(), cmap='gray')
        axs[1].set_title('Ground Truth Mask')
        axs[1].axis('off')

        axs[2].imshow(seg_preds[i].squeeze(), cmap='gray')
        axs[2].set_title('Predicted Mask')
        axs[2].axis('off')

        plt.show()
def estimate_pos_weight(dataset, max_weight=50.0):
    pos = 0.0
    total = 0.0

    for i in range(len(dataset)):
        mask = dataset[i]["mask"]
        pos += mask.sum().item()
        total += mask.numel()

    neg = total - pos
    weight = neg / max(pos, 1.0)
    weight = min(weight, max_weight)

    return torch.tensor([weight], dtype=torch.float32)