FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
COPY requirements_dev.txt .

RUN pip install --no-cache-dir \
    -r requirements.txt

COPY . .

RUN pip install -e .

ENTRYPOINT [
  "python",
  "-u",
  "-m",
  "pathmnist_mlops.train"
]
