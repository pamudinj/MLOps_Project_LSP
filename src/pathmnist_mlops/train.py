from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
from torch.optim import Adam

from pathmnist_mlops.data import get_dataloaders
from pathmnist_mlops.model import Model


class PathMNISTClassifier(pl.LightningModule):
    """PyTorch Lightning module for PathMNIST classification."""

    def __init__(self, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = Model()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.squeeze(1).long()
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.squeeze(1).long()
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr)


def train():
    """
    Train a CNN on the PathMNIST dataset using PyTorch Lightning.

    Loads train and validation dataloaders, trains the model,
    and saves the best checkpoint to the models/ directory.
    """
    train_loader, val_loader, _ = get_dataloaders(batch_size=64)

    model = PathMNISTClassifier(lr=1e-3)

    Path("models").mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath="models",
        filename="pathmnist_cnn",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
    )

    trainer = pl.Trainer(
        max_epochs=20,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
    )

    trainer.fit(model, train_loader, val_loader)
    print("Model saved.")


if __name__ == "__main__":
    train()