import io

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

import pathmnist_mlops.api as api
from pathmnist_mlops.api import app

client = TestClient(app)


def test_root_endpoint() -> None:
    """Test the root endpoint."""

    response = client.get("/")

    assert response.status_code == 200

    assert response.json() == {"message": "PathMNIST FastAPI inference service"}


def test_docs_endpoint() -> None:
    """Test the Swaggerdocumentation endpoint."""

    response = client.get("/docs")

    assert response.status_code == 200


def test_openapi_schema() -> None:
    """Test the OpenAPI schema endpoint."""

    response = client.get("/openapi.json")

    assert response.status_code == 200

    assert "paths" in response.json()


def test_get_model(monkeypatch):
    """Test lazy loading."""

    dummy = object()

    monkeypatch.setattr(api, "load_model", lambda: dummy)

    api._model = None

    model = api.get_model()

    assert model is dummy


def test_load_model_without_env(monkeypatch):
    """MODEL_NAME should be required."""

    monkeypatch.delenv("MODEL_NAME", raising=False)

    with pytest.raises(ValueError):
        api.load_model()


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)  # one parameter

    def forward(self, x):
        out = torch.zeros((1, 9))
        out[0, 3] = 10
        return out


def test_predict(monkeypatch):
    """
    Test the /predict endpoint using
    a mocked classification model.
    """

    monkeypatch.setattr(api, "get_model", lambda: DummyModel())

    image = Image.new("RGB", (28, 28))

    buffer = io.BytesIO()

    image.save(buffer, format="PNG")

    buffer.seek(0)

    response = client.post(
        "/predict",
        files={"file": ("image.png", buffer, "image/png")},
    )

    assert response.status_code == 200

    assert response.json()["prediction_index"] == 3
