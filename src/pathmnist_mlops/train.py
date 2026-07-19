import logging
import os
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profilers import PyTorchProfiler
from torch import nn
from torch.optim import Adam

from pathmnist_mlops.data import get_dataloaders
from pathmnist_mlops.model import Model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PathMNISTClassifier(pl.LightningModule):
    """PyTorch Lightning module for training and evaluating the PathMNIST CNN.

    The module encapsulates the model architecture, loss function,
    training loop, validation loop, and optimizer configuration.
    """

    def __init__(self, lr: float = 1e-3, weight_decay: float = 1e-4):
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
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_acc",
            acc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.squeeze().long()
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_acc",
            acc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

    def configure_optimizers(self):
        return Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )


CONFIG_PATH = os.getenv(
    "HYDRA_CONFIG_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "configs"),
)


@hydra.main(
    version_base=None,
    config_path=CONFIG_PATH,
    config_name="config",
)
def train(cfg: DictConfig) -> None:
    """
    Train a CNN on the PathMNIST dataset using PyTorch Lightning.

    Loads train and validation dataloaders, trains the model,
    and saves the best checkpoint to the models/ directory.
    """
    pl.seed_everything(cfg.training.seed, workers=True)

    logger.info("Loading data...")

    train_loader, val_loader, _ = get_dataloaders(cfg.training.batch_size)

    model = PathMNISTClassifier(
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    Path("models").mkdir(parents=True, exist_ok=True)

    logger.info("Starting training...")
    wandb_logger = WandbLogger(
        project="pathmnist-mlops",
        entity="pamudinj-ludwig-maximilian-university-of-munich",
    )

    wandb_logger.experiment.config.update(
        OmegaConf.to_container(cfg, resolve=True),
        allow_val_change=True,
    )

    logger.info("Configuring checkpoint...")

    checkpoint_callback = ModelCheckpoint(
        dirpath="models",
        filename="pathmnist_cnn-{epoch:02d}-{val_acc:.4f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
    )

    early_stopping = EarlyStopping(
        monitor="val_acc",
        mode="max",
        patience=cfg.training.early_stopping_patience,
    )

    profiler = PyTorchProfiler(
        dirpath="profiler_logs",
        filename="pathmnist_profile",
        export_to_chrome=True,  # writes a trace.json you can open in chrome://tracing
        profile_memory=True,
        row_limit=20,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        callbacks=[checkpoint_callback, early_stopping],
        logger=wandb_logger,
        log_every_n_steps=cfg.training.log_every_n_steps,
        accelerator="auto",
        devices="auto",
        strategy="ddp",  # set to "auto" when training via CL command "train"
        profiler=profiler,
    )

    trainer.fit(model, train_loader, val_loader)

    if checkpoint_callback.best_model_path:
        artifact = wandb.Artifact(
            name="pathmnist-model",
            type="model",
            metadata={"val_acc": trainer.callback_metrics["val_acc"].item()},
        )

        artifact.add_file(checkpoint_callback.best_model_path)

        wandb_logger.experiment.log_artifact(artifact)

        logger.info("Model saved.")


if __name__ == "__main__":
    train()
