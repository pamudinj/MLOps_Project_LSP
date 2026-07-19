# Getting Started

## Installation

Clone the repository

```bash
git clone <repository_url>
cd pathmnist_mlops
```

Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

## Train a model

```bash
python -m pathmnist_mlops.train
```

## Evaluate

```bash
python -m pathmnist_mlops.evaluate
```

## Export ONNX

```bash
python -m pathmnist_mlops.export_onnx
```

## Run API

```bash
uvicorn pathmnist_mlops.api_onnx:app
```