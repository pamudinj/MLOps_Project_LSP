from google.cloud import aiplatform

PROJECT_ID = "mlops-project-497719"

REGION = "europe-west1"

BUCKET = "gs://mlops_data_bucket-1"


aiplatform.init(
    project=PROJECT_ID,
    location=REGION,
    staging_bucket=BUCKET,
)


job = aiplatform.CustomContainerTrainingJob(
    display_name="pathmnist-training",
    container_uri=("europe-west1-docker.pkg.dev/mlops-project-497719/mlops-container-registry/pathmnist-train:latest"),
)


job.run(
    replica_count=1,
    machine_type="n1-standard-4",
)
