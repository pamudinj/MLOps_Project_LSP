from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, confusion_matrix

from pathmnist_mlops.data import get_dataloaders
from pathmnist_mlops.model import Model


def evaluate() -> None:
    """
    Evaluate a trained CNN on the PathMNIST test dataset.

    Loads the saved model checkpoint and computes the
    classification accuracy and confusion matrix on the
    test split.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader = get_dataloaders(batch_size=64)

    model = Model().to(device)

    model_path = Path("models/pathmnist_cnn.pt")

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device,
        )
    )

    model.eval()

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
