"""FastAPI service for PathMNIST data drift detection."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from pathmnist_mlops.data_drift import (
    create_current_dataframe,
    create_reference_dataframe,
    generate_report,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown events.
    """

    logger.info("Starting drift detection service...")

    yield

    logger.info("Stopping drift detection service...")


app = FastAPI(
    title="PathMNIST Drift Detection API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict[str, str]:
    """
    Root endpoint.
    """

    return {"message": "PathMNIST drift detection service."}


@app.post("/detect-drift")
def detect_drift() -> dict[str, str]:
    """
    Generate a data drift report using
    the reference and current datasets.
    """

    try:
        logger.info("Loading reference dataset...")

        reference_df = create_reference_dataframe()

        logger.info("Loading current dataset...")

        current_df = create_current_dataframe()

        logger.info("Generating Evidently report...")

        gcs_uri = generate_report(
            reference_df,
            current_df,
        )

        return {
            "status": "success",
            "message": ("Data drift report generated successfully."),
            "report": gcs_uri or "reports/data_drift_report.html",
        }

    except Exception as error:
        logger.exception("Failed to generate drift report.")

        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error
