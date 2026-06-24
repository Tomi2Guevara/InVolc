import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Tuple
import segmentation_models_pytorch as smp
import helper_functions as hf
from tqdm.auto import tqdm


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

        self._classifier_head_trainable = True

    def forward(self, x):
        """
        x: (B, in_channels, H, W)
        returns:
            clf_logits: (B, num_classes_clf)
            seg_logits: (B, classes_seg, H, W)
        """
        features = self.unet.encoder(x)
        bottleneck = features[-1]
        decoder_output = self.unet.decoder(features)
        seg_logits = self.unet.segmentation_head(decoder_output)
        clf_logits = self.classification_head(bottleneck)
        return clf_logits, seg_logits

    def train_step(self,
                   dataloader: DataLoader,
                   loss_fn_clf: torch.nn.Module,
                   loss_fn_seg: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   device: Optional[torch.device] = None) -> Tuple[float, float, float]:
        """Ejecuta un paso de entrenamiento."""
        if device is None:
            device = self._device

        clf_losses, seg_losses = [], []
        self.train()
        loss = 0.0

        for batch in dataloader:
            images, masks, labels = hf.prepare_batch(
                batch, device=device,
                normalize_mask=True,
                require_float_labels=True
            )

            # forward pass
            clf_logits, seg_logits = self(images)

            # compute losses
            loss_clf = loss_fn_clf(clf_logits, labels.unsqueeze(1))
            loss_seg = loss_fn_seg(seg_logits, masks)

            loss = loss_clf + 2 * loss_seg

            # backward pass and optimization
            optimizer.zero_grad()
            loss.backward()

            # opcional pero muy recomendable
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=2.0)
            optimizer.step()

            # logging
            clf_losses.append(loss_clf.item())
            seg_losses.append(loss_seg.item())

        mean_clf = sum(clf_losses) / len(clf_losses)
        mean_seg = sum(seg_losses) / len(seg_losses)
        mean_tot = mean_clf + 2 * mean_seg

        return mean_tot, mean_clf, mean_seg

    def test_step(self,
                  dataloader: DataLoader,
                  loss_fn_clf: torch.nn.Module,
                  loss_fn_seg: torch.nn.Module,
                  metrics_obj=None,
                  device: Optional[torch.device] = None) -> dict:
        """Ejecuta un paso de evaluación."""
        if device is None:
            device = self._device

        self.eval()
        if metrics_obj is not None:
            metrics_obj.reset()

        clf_losses, seg_losses = [], []

        all_clf_preds = []
        all_clf_targets = []

        all_seg_preds = []
        all_seg_targets = []

        with torch.no_grad():
            for batch in dataloader:
                images, masks, labels = hf.prepare_batch(
                    batch, device=device,
                    normalize_mask=True,
                    require_float_labels=True
                )

                # Forward
                clf_logits, seg_logits = self(images)

                # Losses
                loss_clf = loss_fn_clf(clf_logits, labels.unsqueeze(1))
                loss_seg = loss_fn_seg(seg_logits, masks)

                clf_losses.append(loss_clf.item())
                seg_losses.append(loss_seg.item())

                # Predictions
                clf_probs = torch.sigmoid(clf_logits)
                clf_preds = (clf_probs > 0.5).float()

                if metrics_obj is not None:
                    metrics_obj.update(clf_preds.squeeze(1), labels)

                seg_probs = torch.sigmoid(seg_logits)
                seg_preds = (seg_probs > 0.5).float()

                # Save for computing metrics outside
                all_clf_preds.append(clf_preds.cpu())
                all_clf_targets.append(labels.cpu())

                all_seg_preds.append(seg_preds.cpu())
                all_seg_targets.append(masks.cpu())

        # Concatenate everything
        all_clf_preds = torch.cat(all_clf_preds)
        all_clf_targets = torch.cat(all_clf_targets)

        all_seg_preds = torch.cat(all_seg_preds)
        all_seg_targets = torch.cat(all_seg_targets)

        result = {
            "clf_loss": sum(clf_losses) / len(clf_losses),
            "seg_loss": sum(seg_losses) / len(seg_losses),
            "total_loss": sum(clf_losses) / len(clf_losses) + 2 * sum(seg_losses) / len(seg_losses),

            # Raw predictions → for computing metrics outside
            "clf_preds": all_clf_preds,
            "clf_targets": all_clf_targets,
            "seg_preds": all_seg_preds,
            "seg_targets": all_seg_targets,
        }

        return result

    def fit(self,
            train_dataloader: DataLoader,
            test_dataloader: DataLoader,
            loss_fn_clf: torch.nn.Module,
            loss_fn_seg: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            num_epochs: int = 10,
            metrics_obj=None,
            device: Optional[torch.device] = None):
        """Entrena el modelo durante num_epochs épocas."""
        if device is None:
            device = self._device

        test_results = []
        for epoch in tqdm(range(num_epochs)):
            train_total_loss, train_clf_loss, train_seg_loss = self.train_step(
                train_dataloader,
                loss_fn_clf,
                loss_fn_seg,
                optimizer,
                device
            )

            test_results_step = self.test_step(
                test_dataloader,
                loss_fn_clf,
                loss_fn_seg,
                metrics_obj,
                device
            )

            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"  Train Loss: {train_total_loss:.4f} (Clf: {train_clf_loss:.4f}, Seg: {train_seg_loss:.4f})")
            print(
                f"  Test Loss: {test_results_step['total_loss']:.4f} (Clf: {test_results_step['clf_loss']:.4f}, Seg: {test_results_step['seg_loss']:.4f})")
            test_results.append(test_results_step)

        return test_results
