import io

import onnxruntime as ort  # type: ignore
import torch
from fastapi import (
    FastAPI,
    File,
    UploadFile,
)
from PIL import Image
from torchvision import transforms

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

transform = transforms.Compose(
    [
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ]
)


session_options = ort.SessionOptions()

session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(
    "models/pathmnist_model.onnx",
    sess_options=session_options,
    providers=["CPUExecutionProvider"],
)


@app.get("/")
def root() -> dict[str, str]:
    """
    Root endpoint for API health check.
    """

    return {"message": "PathMNIST ONNX inference service"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, str | int | float]:
    """
    Predict the pathology tissue class
    using the ONNX runtime model.
    """

    image_bytes = await file.read()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    width, height = image.size

    if width > 100 or height > 100:
        return {"message": "Input image size is not compatible with PathMNIST images."}

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

    if confidence_value < 0.6:
        return {
            "message": "Input image is not recognized as a PathMNIST sample.",
            "confidence": confidence_value,
        }

    return {
        "prediction_index": prediction_value,
        "prediction_label": LABELS[prediction_value],
        "confidence": confidence_value,
    }
