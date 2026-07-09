FROM python:3.12-slim

WORKDIR /app

COPY requirements_backend.txt .
COPY requirements.txt .
COPY pyproject.toml .
COPY README.md .

COPY src src/
COPY reports reports/

RUN pip install -r requirements_backend.txt
RUN pip install -r requirements.txt
RUN pip install . --no-deps

CMD ["uvicorn", "pathmnist_mlops.drift_api:app", "--host", "0.0.0.0", "--port", "8080"]
