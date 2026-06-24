from typing import Tuple
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


class PairedTransform:
    """Transformaciones pareadas para imagen/máscara.

    La imagen puede recibir augmentations fotométricas y normalización.
    La máscara solo recibe operaciones geométricas sincronizadas.
    """

    def __init__(
            self,
            size: Tuple[int, int] = (224, 224),
            rotation: float = 0.0,
            hflip_p: float = 0.0,
            vflip_p: float = 0.0,
            translate: Tuple[float, float] = (0.0, 0.0),
            scale: Tuple[float, float] = (1.0, 1.0),
            shear: float = 0.0,
            use_color_jitter: bool = False,
            use_blur: bool = False,
            normalize: bool = True,
            mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
            std: Tuple[float, ...] = (0.229, 0.224, 0.225),
            
    ):
        self.size = list(size)
        self.rotation = rotation
        self.hflip_p = hflip_p
        self.vflip_p = vflip_p
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.use_color_jitter = use_color_jitter
        self.use_blur = use_blur
        self.normalize = normalize
        self.mean = list(mean)
        self.std = list(std)
        self._color_jitter = transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
        ) if use_color_jitter else None

    def _apply_geometric(self, image, mask):
        image = TF.resize(image, self.size, interpolation=InterpolationMode.BILINEAR)  # type: ignore[arg-type]
        if mask is not None:
            mask = TF.resize(mask, self.size, interpolation=InterpolationMode.NEAREST)  # type: ignore[arg-type]

        if random.random() < self.hflip_p:
            image = TF.hflip(image)
            if mask is not None:
                mask = TF.hflip(mask)

        if random.random() < self.vflip_p:
            image = TF.vflip(image)
            if mask is not None:
                mask = TF.vflip(mask)

        angle = random.uniform(-self.rotation, self.rotation) if self.rotation else 0.0
        width, height = image.size
        max_dx = int(self.translate[0] * width) if self.translate else 0
        max_dy = int(self.translate[1] * height) if self.translate else 0
        translations = [
            random.randint(-max_dx, max_dx) if max_dx else 0,
            random.randint(-max_dy, max_dy) if max_dy else 0,
        ]
        scale = random.uniform(self.scale[0], self.scale[1]) if self.scale else 1.0
        shear_x = random.uniform(-self.shear, self.shear) if self.shear else 0.0

        if angle or translations != [0, 0] or scale != 1.0 or shear_x != 0.0:
            image = TF.affine(
                image,
                angle=angle,
                translate=translations,
                scale=scale,
                shear=[shear_x, 0.0],
                interpolation=InterpolationMode.BILINEAR,
                fill=[0.0, 0.0, 0.0],
            )  # type: ignore[arg-type]
            if mask is not None:
                mask = TF.affine(
                    mask,
                    angle=angle,
                    translate=translations,
                    scale=scale,
                    shear=[shear_x, 0.0],
                    interpolation=InterpolationMode.NEAREST,
                    fill=[0.0],
                )  # type: ignore[arg-type]

        return image, mask

    def apply_to_pair(self, image, mask=None):
        image, mask = self._apply_geometric(image, mask)

        if self.use_color_jitter and self._color_jitter is not None:
            image = self._color_jitter(image)

        if self.use_blur:
            image = TF.gaussian_blur(image, kernel_size=[3, 3], sigma=[0.1, 1.0])  # type: ignore[arg-type]

        image = TF.to_tensor(image)  # type: ignore[arg-type]
        if self.normalize:
            image = TF.normalize(image, mean=self.mean, std=self.std)  # type: ignore[arg-type]

        if mask is not None:
            if isinstance(mask, torch.Tensor):
                mask = mask.float()
                if mask.ndim == 2:
                    mask = mask.unsqueeze(0)
                mask = (mask > 0).float()
            else:
                mask_arr = np.array(mask, dtype=np.uint8)
                mask = torch.from_numpy((mask_arr > 0).astype(np.float32)).unsqueeze(0)

        return image, mask


class InVolcDataset(Dataset):
    """Dataset personalizado para imágenes de volcanes."""

    def __init__(self, metadata_csv, root_dir="../data", transform=None, cache=False, subset='test'):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.cache = cache
        self.cached_images = {}
        self.subset = subset

        if not Path(metadata_csv).exists():
            raise FileNotFoundError(f"CSV no encontrado: {metadata_csv}")

        data_full = pd.read_csv(metadata_csv)
        required_cols = {"img_path", "label_bin", "label_raw", "id", "subset"}
        if not required_cols.issubset(data_full.columns):
            raise ValueError(f"CSV debe contener columnas: {required_cols}")

        self.data = data_full[data_full["subset"] == self.subset].reset_index(drop=True)
        if len(self.data) == 0:
            print(f"0 muestras encontradas para el subset '{self.subset}' en {metadata_csv}")

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

        mask = None
        if label_bin.item() == 1:
            if "mask_path" in row.index and not pd.isna(row["mask_path"]):
                mask_path = self.root_dir / row["mask_path"]
            else:
                raise ValueError(f"Fila {idx}: label_bin=1 pero mask_path falta o es NaN")
        else:
            mask = Image.new("L", (224, 224), color=0)
            mask_path = None

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
            if hasattr(self.transform, "apply_to_pair"):
                image, mask = self.transform.apply_to_pair(image, mask)
            else:
                image = self.transform(image)
                if mask_path is not None and mask is not None:
                    mask = self.transform(mask)

        return {
            "image": image,
            "mask": mask,
            "label_raw": label_raw,
            "label_bin": label_bin,
            "id": row["id"]
        }

    def visualizar_muestra(self, idx):
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