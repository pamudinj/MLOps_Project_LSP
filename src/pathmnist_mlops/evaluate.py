import os
from pathlib import Path

import torch
import typer
import wandb
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
)

from pathmnist_mlops.data import (
    get_dataloaders,
)
from pathmnist_mlops.train import (
    PathMNISTClassifier,
)

load_dotenv()


def load_model_from_wandb(
    device,
):
    api = wandb.Api(
        api_key=os.getenv("WANDB_API_KEY"),
        overrides={
            "entity": os.getenv("WANDB_ENTITY"),
            "project": os.getenv("WANDB_PROJECT"),
        },
    )

    entity = os.getenv("WANDB_ENTITY")
    project = os.getenv("WANDB_PROJECT")

    # Find the run with the highest validation accuracy
    best_run = None
    best_acc = -1.0

    for run in api.runs(f"{entity}/{project}"):
        val_acc = run.summary.get("val_acc")

        if val_acc is None:
            continue

        if val_acc > best_acc:
            best_acc = val_acc
            best_run = run

    if best_run is None:
        raise RuntimeError("No run with 'val_acc' found in W&B.")

    print(f"Loading best model from run '{best_run.name}' (val_acc={best_acc:.4f})")

    # Find the model artifact logged by this run
    model_artifact = None

    for artifact in best_run.logged_artifacts():
        if artifact.type == "model":
            model_artifact = artifact
            break

    if model_artifact is None:
        raise RuntimeError("No model artifact found for the best run.")

    artifact_dir = model_artifact.download(root="artifacts")

    checkpoint_path = next(Path(artifact_dir).glob("*.ckpt"))

    model = PathMNISTClassifier.load_from_checkpoint(checkpoint_path).to(device)
    model.eval()

    print(f"Validation Accuracy: {best_acc:.4f}")
    print(f"Artifact: {model_artifact.name}")
    print(f"Checkpoint: {checkpoint_path}")

    return model


def evaluate(data_modification: str = "raw") -> None:
    """Evaluate the best model retrieved from the Weights & Biases Model Registry
    on the PathMNIST test dataset and report classification metrics."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if data_modification == "raw":
        _, _, test_loader = get_dataloaders(batch_size=64)
    else:
        _, _, test_loader = get_dataloaders(batch_size=64, data_modification="drift")

    model = load_model_from_wandb(device)

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            all_predictions.extend(predictions.cpu().numpy())

            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    print(f"Test Accuracy: {accuracy:.4f}")

    cm = confusion_matrix(
        all_labels,
        all_predictions,
    )

    print("\nConfusion Matrix:")

    print(cm)


if __name__ == "__main__":
    typer.run(evaluate)
