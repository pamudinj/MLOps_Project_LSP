# pathmnist_mlops

End-to-end MLOps workflow for PathMNIST classification with experiment tracking, hyperparameter optimization, and deployment.

## Project Description

### Overview

This project develops an end-to-end Machine Learning Operations (MLOps) pipeline for automated histopathological image classification using the PathMNIST dataset. The primary objective is to demonstrate the complete machine learning lifecycle, from data handling and model training to deployment, monitoring, and maintenance in a cloud environment. Rather than focusing solely on model accuracy, the project emphasizes reproducibility, automation, continuous deployment, monitoring, and operational best practices.

Histopathological image classification is an important application of deep learning in digital pathology, where tissue images are automatically categorized into different tissue types. Such systems can assist clinicians by providing fast and consistent preliminary classifications while reducing manual workload.

### Project Goals

The main goal of this project is to build a production-ready image classification service following modern MLOps principles. Specifically, the project aims to:

- Train a convolutional neural network (CNN) for multi-class tissue classification.
- Track experiments and manage trained models using Weights & Biases.
- Export the trained PyTorch model to the ONNX format for efficient inference.
- Deploy the inference service as a FastAPI application on Google Cloud Run.
- Develop a Streamlit frontend for interactive image classification.
- Monitor deployed models using Prometheus-compatible metrics and Google Cloud monitoring.
- Detect potential data drift using Evidently to identify changes in incoming data distributions.
Dataset

The project uses the PathMNIST dataset from the MedMNIST benchmark collection. PathMNIST contains 28×28 RGB histopathology image patches extracted from colorectal cancer tissue slides. Each image belongs to one of nine tissue categories, including adipose tissue, background, debris, lymphocytes, mucus, smooth muscle, normal colon mucosa, cancer-associated stroma, and colorectal adenocarcinoma epithelium.

The MedMNIST package provides standardized train, validation, and test splits, making it suitable for reproducible experimentation and benchmarking.

### Model

The project uses a custom Convolutional Neural Network (CNN) implemented in PyTorch and trained using PyTorch Lightning. The model learns to classify tissue images into the nine PathMNIST classes. Hyperparameters and training configuration are managed using Hydra, while experiments and model artifacts are tracked using Weights & Biases.

After training, the best-performing model is exported to the ONNX format to enable faster and more portable inference. The deployed inference API uses ONNX Runtime to perform efficient predictions in production.

### MLOps Workflow

The project follows an end-to-end MLOps workflow consisting of:

- Data loading using MedMNIST.
- Model training with PyTorch Lightning.
- Experiment tracking using Weights & Biases.
- Automated testing, linting, and type checking.
- Continuous integration using GitHub Actions.
- Containerization using Docker.
- Deployment to Google Cloud Run.
- Interactive frontend using Streamlit.
- Model monitoring through Prometheus-compatible metrics.
- Data drift detection using Evidently.
- Cloud-based monitoring and alerting using Google Cloud services.

The overall objective is to demonstrate how a machine learning model can be developed, deployed, monitored, and maintained using modern software engineering and MLOps practices while ensuring reproducibility, scalability, and maintainability.


### System Architecture

The following diagram illustrates the overall MLOps architecture of the project, including model training, deployment, monitoring, and drift detection.

<p align="center">
  <img src="reports/figures/architecture.png" alt="System Architecture" width="1000">
</p>

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
