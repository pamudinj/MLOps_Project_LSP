from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

import pathmnist_mlops.api_onnx as api
from pathmnist_mlops.api_onnx import app

client = TestClient(app)


def test_log_prediction(tmp_path, monkeypatch):
    """Test that inference log is created."""

    monkeypatch.setattr(api, "LOG_DIR", tmp_path)
    monkeypatch.setattr(api, "LOG_FILE", tmp_path / "log.csv")

    image = Image.new("RGB", (28, 28), color="red")

    api.log_prediction(
        filename="test.png",
        image=image,
        prediction="adipose",
        confidence=0.99,
    )

    assert api.LOG_FILE.exists()


def test_metrics_endpoint():
    """Test Prometheus metrics endpoint."""

    response = client.get("/metrics")

    assert response.status_code == 200

    assert "prediction_requests_total" in response.text


def test_root_endpoint():
    """Test root endpoint."""

    response = client.get("/")

    assert response.status_code == 200

    assert response.json() == {"message": "PathMNIST ONNX inference service"}
