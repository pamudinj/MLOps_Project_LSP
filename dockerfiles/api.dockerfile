FROM python:3.12-slim AS base

WORKDIR /app

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY pyproject.toml .
COPY README.md .

RUN pip install -r requirements.txt --no-cache-dir --verbose

COPY src src/
COPY configs configs/

RUN pip install . --no-deps --no-cache-dir --verbose

EXPOSE $PORT

ENTRYPOINT ["sh", "-c", "uvicorn pathmnist_mlops.api:app --host 0.0.0.0 --port $PORT"]
