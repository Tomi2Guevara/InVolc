"""Modelo base simple para segmentación binaria.

Arquitectura:
- U-Net estándar
- 1 canal de salida
- Sin clasificación, sin atención, sin bloques extra
"""

import torch.nn as nn
import segmentation_models_pytorch as smp


class InVolcModel(nn.Module):
    """U-Net básica para segmentación semántica binaria."""

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        in_channels: int = 1,
        classes: int = 1,
    ):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,  # devuelve logits crudos
        )

    def forward(self, x):
        return self.model(x)

