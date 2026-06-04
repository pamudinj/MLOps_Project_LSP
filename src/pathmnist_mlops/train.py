from pathlib import Path

import argparse
import wandb

import logging
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.optim import Adam

from pathmnist_mlops.data import get_dataloaders
from pathmnist_mlops.model import Model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        labels = labels.squeeze().long()
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)     
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.squeeze().long()
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr)


def train(lr=1e-3, batch_size = 64, max_epochs = 20, log_every_n_steps = 10):
    """
    Train a CNN on the PathMNIST dataset using PyTorch Lightning.

    Loads train and validation dataloaders, trains the model,
    and saves the best checkpoint to the models/ directory.
    """
    logger.info("Loading data...")
    
    train_loader, val_loader, _ = get_dataloaders(batch_size)

    model = PathMNISTClassifier(lr)

    Path("models").mkdir(parents=True, exist_ok=True)
    
    logger.info("Configuring checkpoint...")
    
    checkpoint_callback = ModelCheckpoint(
        dirpath="/tmp/models",
        filename="pathmnist_cnn",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
    )
    
    logger.info("Starting training...")
    wandb_logger = WandbLogger(project="pathmnist-mlops")
    
    trainer = pl.Trainer(
        max_epochs = max_epochs,
        callbacks=[checkpoint_callback],
        logger = wandb_logger,
        log_every_n_steps = log_every_n_steps,
    )

    trainer.fit(model, train_loader, val_loader)
    
    logger.info("Model saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    args = parser.parse_args()
    
    wandb.init(project="pathmnist-mlops", config=vars(args))
    
    train(
        lr=wandb.config.lr,
        batch_size=wandb.config.batch_size,
        max_epochs=wandb.config.max_epochs,
        log_every_n_steps=wandb.config.log_every_n_steps,
    )