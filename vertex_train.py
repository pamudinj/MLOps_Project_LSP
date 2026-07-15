import itertools
import os
import random

import yaml
from google.cloud import aiplatform

PROJECT_ID = "mlops-project-497719"
REGION = "europe-west1"
BUCKET = "gs://mlops-vertex-europe"
SELECTION_SEED = 50
TRAINING_SEED = 55


def main() -> None:
    """
    Submit multiple Vertex AI training jobs using the hyperparameter
    configurations defined in ``configs/sweep.yaml``.

    The selected hyperparameter combinations are passed to the training
    container as Hydra configuration overrides. Each job trains the model
    independently and logs results to Weights & Biases.
    """

    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        staging_bucket=BUCKET,
    )

    with open("configs/sweep.yaml") as f:
        sweep = yaml.safe_load(f)

    params = sweep["parameters"]

    learning_rates = params["training.learning_rate"]["values"]
    batch_sizes = params["training.batch_size"]["values"]
    epochs = params["training.epochs"]["values"]
    weight_decays = params["training.weight_decay"]["values"]

    all_configs = list(
        itertools.product(
            learning_rates,
            batch_sizes,
            epochs,
            weight_decays,
        )
    )

    random.Random(SELECTION_SEED).shuffle(all_configs)

    run_cap = sweep.get("run_cap", len(all_configs))

    for lr, bs, ep, wd in all_configs[:run_cap]:
        print(f"Submitting job lr={lr}, batch_size={bs}, epochs={ep}, weight_decay={wd}")

        job = aiplatform.CustomContainerTrainingJob(
            display_name=f"pathmnist-lr{lr}-bs{bs}-ep{ep}-wd{wd}",
            container_uri="europe-west1-docker.pkg.dev/mlops-project-497719/mlops-container-registry/pathmnist-train:latest",
        )

        job.run(
            replica_count=1,
            machine_type="n1-standard-4",
            args=[
                f"training.learning_rate={lr}",
                f"training.batch_size={bs}",
                f"training.epochs={ep}",
                f"training.weight_decay={wd}",
                f"training.seed={TRAINING_SEED}",
            ],
            environment_variables={
                "WANDB_API_KEY": os.getenv("WANDB_API_KEY", ""),
                "WANDB_ENTITY": os.getenv("WANDB_ENTITY", ""),
                "WANDB_PROJECT": os.getenv("WANDB_PROJECT", ""),
            },
            sync=False,
        )


if __name__ == "__main__":
    main()
