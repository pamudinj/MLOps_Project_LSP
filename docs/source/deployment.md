# Deployment

The deployment pipeline consists of:

1. Export the best W&B model to ONNX.
2. Build a Docker container.
3. Push the container image to Google Artifact Registry.
4. Deploy the FastAPI application to Google Cloud Run.

The deployed API accepts PathMNIST images and returns the predicted class.