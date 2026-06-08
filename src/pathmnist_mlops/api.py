import io
import os

import torch
from dotenv import load_dotenv
from fastapi import (
    FastAPI,
    File,
    UploadFile,
)
from PIL import Image
from torchvision import transforms

import wandb
from pathmnist_mlops.train import (
    PathMNISTClassifier,
)

load_dotenv()

app = FastAPI()

LABELS = [
    "adipose",
    "background",
    "debris",
    "lymphocytes",
    "mucus",
    "smooth muscle",
    "normal colon mucosa",
    "cancer-associated stroma",
    "colorectal adenocarcinoma epithelium",
]

_model = None  # Cache the loaded model


def load_model() -> PathMNISTClassifier:
    """
    Load the trained model checkpoint
    from Weights & Biases artifacts.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    api = wandb.Api()

    model_name = os.getenv("MODEL_NAME")

    if model_name is None:
        raise ValueError("MODEL_NAME environment variable is not set")
    artifact = api.artifact(model_name)

    artifact_dir = artifact.download()

    checkpoint_path = next(
        os.path.join(artifact_dir, file) for file in os.listdir(artifact_dir) if file.endswith(".ckpt")
    )

    model = PathMNISTClassifier.load_from_checkpoint(checkpoint_path)

    model.to(device)

    model.eval()

    return model


def get_model() -> PathMNISTClassifier:
    """Lazy-load model on first use."""
    global _model
    if _model is None:
        _model = load_model()
    return _model


transform = transforms.Compose(
    [
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ]
)


@app.get("/")
def root() -> dict[str, str]:
    """
    Root endpoint for API health check.
    """

    return {"message": "PathMNIST FastAPI inference service"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, str | int | float]:
    """
    Predict the pathology tissue class
    for an uploaded image.
    """

    image_bytes = await file.read()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    width, height = image.size

    if width > 100 or height > 100:
        return {"message": "Input image size is not compatible with PathMNIST images."}

    image_tensor = transform(image).unsqueeze(0)

    model = get_model()  # Load model on first prediction
    device = next(model.parameters()).device

    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1,
        )

        confidence_tensor, prediction_tensor = torch.max(
            probabilities,
            dim=1,
        )

        confidence: float = float(confidence_tensor.item())

        prediction: int = int(prediction_tensor.item())

    if confidence < 0.6:
        return {
            "message": "Input image is not recognized as a PathMNIST sample.",
            "confidence": confidence,
        }

    return {
        "prediction_index": prediction,
        "prediction_label": LABELS[prediction],
        "confidence": confidence,
    }
