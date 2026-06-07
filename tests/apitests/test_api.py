from fastapi.testclient import TestClient

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
