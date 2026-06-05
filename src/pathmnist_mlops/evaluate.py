import os
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
)

import wandb
from pathmnist_mlops.data import (
    get_dataloaders,
)
from pathmnist_mlops.model import (
    Model,
)


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

    artifact = api.artifact(os.getenv("MODEL_NAME"))

    artifact_dir = artifact.download(root="artifacts")

    checkpoint_path = list(Path(artifact_dir).glob("*.ckpt"))[0]

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model = Model().to(device)

    model.load_state_dict(
        checkpoint["state_dict"],
        strict=False,
    )

    model.eval()

    return model


def evaluate() -> None:
    """
    Evaluate model loaded
    from WandB registry.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader = get_dataloaders(batch_size=64)

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
    evaluate()
