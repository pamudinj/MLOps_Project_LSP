import csv
import io
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort  # type: ignore
import torch
from fastapi import (
    FastAPI,
    File,
    UploadFile,
)
from PIL import Image
from torchvision import transforms

session: ort.InferenceSession
transform: transforms.Compose

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

LOG_DIR = Path("monitoring")
LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)
LOG_FILE = LOG_DIR / "inference_log.csv"


def log_prediction(
    filename: str,
    image: Image.Image,
    prediction: str,
    confidence: float,
) -> None:
    """
    Log inference input and output to the monitoring csv.
    """

    image_np = np.asarray(image)
    new_file = not LOG_FILE.exists()

    with open(
        LOG_FILE,
        "a",
        newline="",
    ) as file:
        writer = csv.writer(file)
        if new_file:
            writer.writerow(
                [
                    "timestamp",
                    "filename",
                    "prediction",
                    "confidence",
                    "mean",
                    "std",
                    "red_mean",
                    "green_mean",
                    "blue_mean",
                ]
            )

        writer.writerow(
            [
                datetime.utcnow().isoformat(),
                filename,
                prediction,
                confidence,
                float(image_np.mean()),
                float(image_np.std()),
                float(image_np[:, :, 0].mean()),
                float(image_np[:, :, 1].mean()),
                float(image_np[:, :, 2].mean()),
            ]
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load ONNX model and transforms
    during application startup.
    """

    global session
    global transform

    session_options = ort.SessionOptions()

    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        "models/pathmnist_model.onnx",
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )

    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
        ]
    )

    yield

    del session
    del transform


app = FastAPI(lifespan=lifespan)


@app.get("/")
def root() -> dict[str, str]:
    """
    Root endpoint.
    """

    return {"message": "PathMNIST ONNX inference service"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    """
    Predict pathology class
    from uploaded image.
    """

    image_bytes = await file.read()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image_tensor = transform(image).unsqueeze(0)

    outputs = session.run(
        None,
        {"input": image_tensor.numpy()},
    )[0]

    outputs = torch.tensor(outputs)

    probabilities = torch.softmax(
        outputs,
        dim=1,
    )

    confidence, prediction = torch.max(
        probabilities,
        dim=1,
    )

    confidence_value = float(confidence.item())

    prediction_value = int(prediction.item())

    log_prediction(
        filename=file.filename,
        image=image,
        prediction=LABELS[prediction_value],
        confidence=confidence_value,
    )

    return {
        "prediction_index": prediction_value,
        "prediction_label": LABELS[prediction_value],
        "confidence": confidence_value,
        "warning": "Model was trained only on PathMNIST pathology images.",
    }
