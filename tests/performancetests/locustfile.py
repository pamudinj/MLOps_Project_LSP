from locust import HttpUser, between, task


class PathMNISTUser(HttpUser):
    """
    Locust user class for load testing the
    PathMNIST FastAPI inference service.
    """

    # Simulate users waiting 1–2 seconds between requests.
    wait_time = between(1, 2)

    @task(1)
    def root_endpoint(self) -> None:
        """
        Test the root health endpoint.
        """

        self.client.get("/")

    @task(3)
    def predict_endpoint(self) -> None:
        """
        Test the prediction endpoint.
        """

        with open(
            "tests/sample_images/test_image.png",
            "rb",
        ) as image_file:
            self.client.post(
                "/predict",
                files={"file": image_file},
            )
