import os
from random import random
import custom_dataset as dataset
import helper_functions as hf
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch
from model import InVolcModel
from metrics import MetricsManager
from diceBCELoss import DiceBCELoss
import matplotlib.pyplot as plt

def main():
    device = hf.set_device()
    seed = 42
    hf.set_seed(seed)
    metrics = MetricsManager(device)

    train_transforms = transforms.Compose([
        # Resize y Tensor
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),

        transforms.ToTensor(),
    ])

    train_data = dataset.InVolcDataset(
        metadata_csv='metadata_sample_03.csv',
        transform=train_transforms,
        subset='train')
    test_data = dataset.InVolcDataset(
        metadata_csv='metadata_sample_03.csv',
        transform=test_transforms,
        subset='test')

    train_dataloader = DataLoader(
        train_data,
        batch_size=8,
        shuffle=True,
        num_workers=4
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=5,
        shuffle=False,
        num_workers=4
    )

    loss_fn_clf = nn.BCEWithLogitsLoss()
    loss_fn_seg = DiceBCELoss(bce_weight=0.7, dice_weight=0.2)

    hf.set_seed(seed)
    invol_v2 = InVolcModel(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes_seg=1,
            num_classes_clf=1)


    invol_v2.to(device)
    optimizer = torch.optim.Adam(invol_v2.parameters(), lr=1e-4)
    result = invol_v2.fit(
        train_dataloader,
        test_dataloader,
        loss_fn_clf,
        loss_fn_seg,
        optimizer,
        num_epochs=30,
        device=device,
        metrics=metrics,
    )

    metrics.compute()
    # Extraer pérdidas de test por epoch
    test_total_loss = [r["total_loss"] for r in result]
    test_clf_loss   = [r["clf_loss"]   for r in result]
    test_seg_loss   = [r["seg_loss"]   for r in result]

    epochs = range(1, len(result) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, test_total_loss, label="Test total loss")
    plt.plot(epochs, test_clf_loss,   label="Test clf loss")
    plt.plot(epochs, test_seg_loss,   label="Test seg loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Evolución de la función de pérdida (test)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Guardar el modelo entrenado
    model_save_path = "invol_v2_model.pth"
    torch.save(invol_v2.state_dict(), model_save_path)
    print(f"Modelo guardado en: {model_save_path}")


if __name__ == '__main__':
    main()
