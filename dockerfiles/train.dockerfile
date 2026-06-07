FROM python:3.12-slim

WORKDIR /app

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY pyproject.toml .
COPY README.md .

RUN pip install --no-cache-dir -r requirements.txt

COPY configs configs/
ENV HYDRA_CONFIG_PATH=/app/configs
COPY src src/

RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT ["python", "-u", "-m", "pathmnist_mlops.train"]
