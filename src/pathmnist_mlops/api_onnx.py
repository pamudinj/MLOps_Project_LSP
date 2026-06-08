import io
from contextlib import asynccontextmanager
from typing import Any

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

    return {
        "prediction_index": prediction_value,
        "prediction_label": LABELS[prediction_value],
        "confidence": confidence_value,
        "warning": "Model was trained only on PathMNIST pathology images.",
    }
