# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENV=.venv \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    HUGGINGFACE_HUB_CACHE=/models/huggingface \
    TRANSFORMERS_CACHE=/models/huggingface \
    ULTRALYTICS_CACHE_DIR=/models/ultralytics \
    YOLO_CACHE_DIR=/models/ultralytics \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    ULTRALYTICS_SETTINGS_DIR=/tmp/ultralytics \
    ULTRALYTICS_CONFIG_DIR=/tmp/ultralytics

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv \
    && uv sync --frozen --no-dev --python /usr/local/bin/python

COPY . .
RUN mkdir -p /models/huggingface /models/ultralytics /tmp/ultralytics /storage \
    && chmod -R 777 /tmp/ultralytics /storage

EXPOSE 5000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
