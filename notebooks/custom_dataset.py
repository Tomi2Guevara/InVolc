from typing import Any

import torch
from PIL.Image import Image
from torch import Tensor
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import warnings

class InVolcDataset(Dataset):
    """
    Dataset personalizado para imágenes de volcanes.

    Args:
        metadata_csv: Ruta al CSV con columnas: img_path, label_bin, label_raw, id
        root_dir: Directorio raíz de las imágenes
        transform: Transformaciones opcionales de torchvision
        cache: Si True, cachea las imágenes en memoria (solo para datasets pequeños)
    """

    def __init__(self, metadata_csv, root_dir="../data", transform=None, cache=False, subset='test'):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.cache = cache
        self.cached_images = {}
        self.subset = subset


        # Validación del CSV
        if not Path(metadata_csv).exists():
            raise FileNotFoundError(f"CSV no encontrado: {metadata_csv}")

        # Cargar el DataFrame COMPLETO
        data_full = pd.read_csv(metadata_csv)

        required_cols = {"img_path", "label_bin", "label_raw", "id", "subset"}
        if not required_cols.issubset(data_full.columns):
            raise ValueError(f"CSV debe contener columnas: {required_cols}")

        # Filtramos el DataFrame según el subset solicitado
        self.data = data_full[data_full["subset"] == self.subset].reset_index(drop=True)

        if len(self.data) == 0:
            print(f"0 muestras encontradas para el subset '{self.subset}' en {metadata_csv}")

        # Validar que los labels sean numéricos
        if not pd.api.types.is_numeric_dtype(self.data["label_bin"]):
            raise ValueError("label_bin debe ser numérico")
        if not pd.api.types.is_numeric_dtype(self.data["label_raw"]):
            raise ValueError("label_raw debe ser numérico")


    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> dict:
        row = self.data.iloc[idx]
        img_path = self.root_dir / row["img_path"]
        label_bin = torch.tensor(int(row["label_bin"]), dtype=torch.long)
        label_raw = torch.tensor(int(row["label_raw"]), dtype=torch.long)

        # La máscara solo existe si label_bin == 1
        mask = None
        if label_bin.item() == 1:
            if "mask_path" in row.index and not pd.isna(row["mask_path"]):
                mask_path = self.root_dir / row["mask_path"]
            else:
                raise ValueError(f"Fila {idx}: label_bin=1 pero mask_path falta o es NaN")
        else:
            #crear una máscara vacía si no existe
            mask = torch.zeros((1, 224, 224), dtype=torch.float32)
            mask_path = None
        # Intenta cargar desde caché o disco
        if idx in self.cached_images:
            image = self.cached_images[idx]["image"]
            mask = self.cached_images[idx]["mask"]
        else:
            try:
                image = Image.open(img_path).convert("RGB")
                if mask_path is not None:
                    mask = Image.open(mask_path).convert("L")
                if self.cache:
                    self.cached_images[idx] = {"image": image, "mask": mask}
            except FileNotFoundError:
                raise FileNotFoundError(f"Imagen no encontrada: {img_path}")
            except Exception as e:
                raise RuntimeError(f"Error al cargar imagen {img_path}: {str(e)}")

        if self.transform:
            image = self.transform(image)
            if mask_path is not None:
                mask = self.transform(mask)

        return {
            "image": image,
            "mask": mask,
            "label_raw": label_raw,
            "label_bin": label_bin,
            "id": row["id"]
        }

    def visualizar_muestra(self, idx):
        """Visualiza imagen y etiquetas de un índice específico."""
        if idx < 0 or idx >= len(self):
            print(f"Índice fuera de rango (0 - {len(self)-1})")
            return

        try:
            sample = self[idx]
            image = sample["image"]
            label_bin = sample["label_bin"].item()
            label_raw = sample["label_raw"].item()

            if isinstance(image, torch.Tensor):
                image_np = image.permute(1, 2, 0).numpy()
            else:
                image_np = np.array(image)

            plt.figure(figsize=(6, 6))
            plt.imshow(image_np)
            plt.axis("off")
            plt.title(f"Índice: {idx}\nLabel bin: {label_bin}\nLabel raw: {label_raw}")
            plt.show()
            print(f"Tamaño de la imagen: {image.shape}")

        except Exception as e:
            print(f"Error al visualizar: {str(e)}")