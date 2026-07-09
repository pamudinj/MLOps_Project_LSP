FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
COPY pyproject.toml .
COPY README.md .

COPY src src/
COPY reports reports/

RUN pip install -r requirements.txt
RUN pip install . --no-deps

EXPOSE $PORT

CMD uvicorn pathmnist_mlops.drift_api:app --host 0.0.0.0 --port ${PORT:-8080}
