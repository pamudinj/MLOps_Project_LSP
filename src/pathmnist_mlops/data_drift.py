"""Generate an Evidently data drift report for PathMNIST."""

import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore
import torch
from evidently.legacy.metric_preset import DataDriftPreset  # type: ignore
from evidently.legacy.report import Report  # type: ignore
from google.cloud import storage  # type: ignore
from medmnist import PathMNIST  # type: ignore
from PIL import Image
from torchvision import transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


ROOT = Path(__file__).parents[2] / "data" / "raw"

REPORT_DIR = Path(__file__).parents[2] / "reports"

REPORT_PATH = REPORT_DIR / "data_drift_report.html"


REFERENCE_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

CURRENT_TRANSFORM = transforms.Compose(
    [
        transforms.ColorJitter(
            brightness=0.10,
            contrast=0.10,
            saturation=0.10,
            hue=0.02,
        ),
        transforms.GaussianBlur(
            kernel_size=3,
            sigma=(0.1, 0.5),
        ),
        transforms.ToTensor(),
    ]
)


def load_dataset(
    split: str,
    transform: transforms.Compose,
) -> PathMNIST:
    """
    Load a PathMNIST split.
    """

    ROOT.mkdir(parents=True, exist_ok=True)

    return PathMNIST(
        split=split,
        root=ROOT,
        download=True,
        transform=transform,
    )


def extract_features(
    dataset: PathMNIST,
) -> pd.DataFrame:
    """
    Convert images into simple numerical features
    for Evidently.
    """

    rows: list[dict[str, float]] = []

    for image, _ in dataset:
        image_np = image.numpy()

        rows.append(
            {
                "mean": float(image_np.mean()),
                "std": float(image_np.std()),
                "min": float(image_np.min()),
                "max": float(image_np.max()),
                "red_mean": float(image_np[0].mean()),
                "green_mean": float(image_np[1].mean()),
                "blue_mean": float(image_np[2].mean()),
            }
        )

    return pd.DataFrame(rows)


def upload_report_to_gcs(
    local_path: Path,
    blob_name: str = "drift_reports/data_drift_report.html",
) -> str | None:
    """
    Upload the generated report to GCS so it survives past the lifetime of the
    (stateless) container that created it. Returns the gs:// URI on success,
    or None if the upload was skipped or failed - callers should treat that as
    non-fatal, since the report still exists locally either way.
    """

    bucket_name = os.getenv("DRIFT_REPORTS_BUCKET")

    if not bucket_name:
        logger.warning("DRIFT_REPORTS_BUCKET not set - skipping GCS upload, report only exists inside the container.")
        return None

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(str(local_path))

        gcs_uri = f"gs://{bucket_name}/{blob_name}"
        logger.info(f"Uploaded drift report to {gcs_uri}")
        return gcs_uri
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to upload drift report to GCS ({e}). Report only exists in the container.")
        return None


def create_reference_dataframe() -> pd.DataFrame:
    """
    Create reference dataframe from
    the training dataset.
    """

    logger.info("Loading reference dataset...")

    dataset = load_dataset(
        split="train",
        transform=REFERENCE_TRANSFORM,
    )

    return extract_features(dataset)


def create_current_dataframe() -> pd.DataFrame:
    """
    Create current dataframe using a
    drifted version of the test set.
    """

    logger.info("Loading current dataset...")

    random.seed(55)
    torch.manual_seed(55)

    dataset = load_dataset(
        split="test",
        transform=CURRENT_TRANSFORM,
    )

    drift_path = Path(__file__).parents[2] / "data" / "drift"
    drift_path.mkdir(parents=True, exist_ok=True)

    images = []
    labels = []

    for image, label in dataset:
        img_np = (image.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
        images.append(img_np)
        labels.append(label)

    np.savez_compressed(
        drift_path / "pathmnist.npz",
        test_images=np.stack(images),
        test_labels=np.array(labels).reshape(-1, 1).astype("uint8"),
    )

    return extract_features(dataset)


def generate_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> str | None:
    """
    Generate and save an Evidently data drift report, then upload it to GCS.

    Returns the gs:// URI if the upload succeeded, else None.
    """

    logger.info("Generating drift report...")

    report = Report(
        metrics=[
            DataDriftPreset(),
        ]
    )

    report.run(
        reference_data=reference_df,
        current_data=current_df,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.save_html(str(REPORT_PATH))

    logger.info(
        "Drift report saved to %s",
        REPORT_PATH,
    )

    return upload_report_to_gcs(REPORT_PATH)


def main() -> None:
    """
    Run the complete data drift pipeline.
    """

    logger.info("Creating reference dataframe...")

    reference_df = create_reference_dataframe()

    logger.info("Creating current dataframe...")

    current_df = create_current_dataframe()

    logger.info(
        "Reference samples: %d",
        len(reference_df),
    )

    logger.info(
        "Current samples: %d",
        len(current_df),
    )

    generate_report(
        reference_df,
        current_df,
    )


if __name__ == "__main__":
    main()
