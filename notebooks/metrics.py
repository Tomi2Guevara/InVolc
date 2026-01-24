# metrics.py

import torch
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryAUROC,
    ConfusionMatrix
)
import matplotlib.pyplot as plt


class MetricsManager():
    """
    Clase profesional para manejar métricas de clasificación y segmentación.
    Acumula resultados por batch, permite hacer compute() por epoch, y luego reset().
    """

    def __init__(self, device: torch.device):
        # Clasificación
        self.acc = BinaryAccuracy().to(device)
        self.prec = BinaryPrecision().to(device)
        self.rec = BinaryRecall().to(device)
        self.f1 = BinaryF1Score().to(device)
        self.auc = BinaryAUROC().to(device)

        self.device = device

    # ----------------------------------------------------------------------
    def update(self, clf_preds, clf_targets):
        """
        Actualiza todas las métricas batch a batch.
        Los tensores deben ser binarios (0/1) pero pueden venir en float32.
        """
        self.acc.update(clf_preds, clf_targets)
        self.prec.update(clf_preds, clf_targets)
        self.rec.update(clf_preds, clf_targets)
        self.f1.update(clf_preds, clf_targets)
        self.auc.update(clf_preds, clf_targets)



    # ----------------------------------------------------------------------
    def compute(self):
        """
        Devuelve un diccionario con todas las métricas del epoch.
        No hace reset automáticamente.
        """
        return {
            "accuracy": self.acc.compute().item(),
            "precision": self.prec.compute().item(),
            "recall": self.rec.compute().item(),
            "f1_score": self.f1.compute().item(),
            "auc": self.auc.compute().item(),
        }

    # ----------------------------------------------------------------------
    def reset(self):
        """
        Resetea los estados internos (para un nuevo epoch).
        """
        self.acc.reset()
        self.prec.reset()
        self.rec.reset()
        self.f1.reset()
        self.auc.reset()

    def plot_result(self, result):
        """
        Plotea las métricas de clasificación.
        """

        # Extraer pérdidas de test por epoch
        test_total_loss = [r["total_loss"] for r in result]
        test_clf_loss = [r["clf_loss"] for r in result]
        test_seg_loss = [r["seg_loss"] for r in result]

        epochs = range(1, len(result) + 1)

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, test_total_loss, label="Test total loss")
        plt.plot(epochs, test_clf_loss, label="Test clf loss")
        plt.plot(epochs, test_seg_loss, label="Test seg loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Evolución de la función de pérdida (test)")
        plt.legend()
        plt.grid(True)
        plt.show()


