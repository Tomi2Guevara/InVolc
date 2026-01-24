import torch.nn as nn
import segmentation_models_pytorch as smp
class InVolcModel(nn.Module):
    def __init__(self,
                 encoder_name: str = 'resnet34',
                 encoder_weights: str = 'imagenet',
                 in_channels: int = 3,
                 classes_seg: int = 1,
                 num_classes_clf: int = 1):
        super().__init__()
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes_seg,
        )

        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            nn.Dropout(0.5),
            nn.Linear(self.unet.encoder.out_channels[-1], 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes_clf)
        )

    def forward(self, x):
        """
        x: (B, in_channels, H, W)
        returns:
            clf_logits: (B, num_classes_clf)
            seg_logits: (B, classes_seg, H, W)
        """
        features = self.unet.encoder(x)
        bottleneck = features[-1]

        # Pasar la lista/tupla completa al decoder (sin usar '*')
        decoder_output = self.unet.decoder(features)
        seg_logits = self.unet.segmentation_head(decoder_output)

        clf_logits = self.classification_head(bottleneck)

        return clf_logits, seg_logits



class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode='binary')

    def forward(self, preds, targets):
        bce_loss = self.bce(preds, targets)
        dice_loss = self.dice(preds, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss