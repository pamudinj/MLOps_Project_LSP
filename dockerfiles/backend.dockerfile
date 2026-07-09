FROM python:3.12-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN mkdir /app

WORKDIR /app

COPY requirements_backend.txt /app/requirements_backend.txt

COPY pyproject.toml /app/pyproject.toml
COPY README.md /app/README.md

COPY src /app/src
COPY models /app/models

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements_backend.txt

RUN pip install . --no-deps

EXPOSE $PORT

CMD uvicorn pathmnist_mlops.api_onnx:app --host 0.0.0.0 --port ${PORT:-8080}
