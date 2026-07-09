"""Generate an Evidently data drift report for PathMNIST."""

import logging
from pathlib import Path

import pandas as pd
from evidently.legacy.metric_preset import DataDriftPreset
from evidently.legacy.report import Report
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

# CURRENT_TRANSFORM = transforms.Compose(
#     [
#         transforms.ColorJitter(
#             brightness=0.35,
#             contrast=0.35,
#             saturation=0.25,
#             hue=0.05,
#         ),
#         transforms.GaussianBlur(kernel_size=3),
#         transforms.ToTensor(),
#     ]
# )

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

    dataset = load_dataset(
        split="test",
        transform=CURRENT_TRANSFORM,
    )

    return extract_features(dataset)


def generate_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> None:
    """
    Generate and save an Evidently
    data drift report.
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
