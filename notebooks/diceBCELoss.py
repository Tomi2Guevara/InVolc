import torch.nn as nn
import torch
import segmentation_models_pytorch as smp
import torch.nn.functional as F

class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode='binary', from_logits=True)

    def forward(self, preds, targets):
        targets = targets.float().clamp(0.0, 1.0)
        bce_loss = self.bce(preds, targets)
        dice_loss = self.dice(preds, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def soft_dice_loss(logits, targets, eps=1e-7):
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)

    intersection = (probs * targets).sum(dim=dims)
    denominator = probs.sum(dim=dims) + targets.sum(dim=dims)

    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight=None, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.pos_weight = pos_weight
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
        )
        dice = soft_dice_loss(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice

# class FocalLoss(nn.Module):
#     """
#     Focal Loss para segmentación:
#     - Enfatiza píxeles difíciles de clasificar
#     - Mejor para clases desbalanceadas
#     """
#     def __init__(self, alpha=0.25, gamma=2.0):
#         super().__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#
#     def forward(self, preds, targets):
#         bce = F.binary_cross_entropy_with_logits(preds, targets, reduction='none')
#         p_t = torch.sigmoid(preds)
#         p_t = p_t * targets + (1 - p_t) * (1 - targets)
#         focal_weight = self.alpha * (1 - p_t) ** self.gamma
#         loss = focal_weight * bce
#         return loss.mean()
#
#
# class CombinedLoss(nn.Module):
#     """Pérdida combinada: Dice + BCE + Focal"""
#     def __init__(self, dice_weight=0.4, bce_weight=0.4, focal_weight=0.2):
#         super().__init__()
#         self.dice_weight = dice_weight
#         self.bce_weight = bce_weight
#         self.focal_weight = focal_weight
#
#         self.dice = smp.losses.DiceLoss(mode='binary')
#         self.bce = nn.BCEWithLogitsLoss()
#         self.focal = FocalLoss()
#
#     def forward(self, preds, targets):
#         dice_loss = self.dice(preds, targets)
#         bce_loss = self.bce(preds, targets)
#         focal_loss = self.focal(preds, targets)
#
#         return (self.dice_weight * dice_loss +
#                 self.bce_weight * bce_loss +
#                 self.focal_weight * focal_loss)