import os
from io import BytesIO

import requests
import streamlit as st
from google.cloud import run_v2  # type: ignore
from PIL import Image


@st.cache_resource
def get_backend_url() -> str:
    """
    Get the backend Cloud Run URL.
    """

    parent = "projects/mlops-project-497719/locations/europe-west1"

    try:
        client = run_v2.ServicesClient()

        services = client.list_services(parent=parent)

        for service in services:
            if service.name.split("/")[-1] == "backend":
                return service.uri

    except Exception:
        pass

    backend = os.environ.get("BACKEND")

    if backend is None:
        raise ValueError("Backend URL not found.")

    return backend


BACKEND_URL = get_backend_url()

API_URL = f"{BACKEND_URL}/predict"

st.set_page_config(
    page_title="PathMNIST Classifier",
    layout="centered",
)

st.title("PathMNIST Classification App")

st.write("Upload a pathology image to predict the tissue class.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=[
        "png",
        "jpg",
        "jpeg",
    ],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded image",
        use_container_width=True,
    )

    if st.button("Predict"):
        with st.spinner("Running inference..."):
            image_bytes = BytesIO()

            image.save(
                image_bytes,
                format="PNG",
            )

            response = requests.post(
                API_URL,
                files={
                    "file": (
                        uploaded_file.name,
                        image_bytes.getvalue(),
                        "image/png",
                    )
                },
            )

        if response.status_code == 200:
            result = response.json()

            if "prediction_label" in result:
                st.success("Prediction completed.")

                st.write(f"### Predicted class: {result['prediction_label']}")

                st.write(f"Confidence: {result['confidence']:.4f}")

            else:
                st.warning(
                    result.get(
                        "message",
                        "Prediction failed.",
                    )
                )

        else:
            st.error("API request failed.")
