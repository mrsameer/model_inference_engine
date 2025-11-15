# Model Inference Engine

FastAPI powered inference microservice with built-in support for Ultralytics YOLO object detection models and BLIP visual-language models. The service exposes unified APIs that surface the available models and execute inference against either model family.

## Features
- `/models` returns the registered YOLO and vision-language models along with metadata.
- `/inference` accepts either an image URL or base64 encoded payload plus the target model id.
- YOLOv8 nano and small weights for object detection tasks.
- Salesforce BLIP VQA base for visual question answering / captioning workflows.
- Container ready Dockerfile plus a docker-compose stack with a persisted cache volume for model weights.

## Quick start
```bash
uv sync
uv run uvicorn main:app --reload --host 0.0.0.0 --port 5000
```

Then exercise the API:
```bash
curl http://localhost:5000/models

curl -X POST http://localhost:5000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "yolov8n",
    "image_url": "https://ultralytics.com/images/bus.jpg"
  }'
```

## Docker compose deployment
```bash
docker compose up --build
```
The compose file exposes port `5000` and mounts the `models-cache` volume to `/models` so downloaded YOLO weights and Hugging Face checkpoints are persisted between restarts.

### GPU acceleration (optional)
On machines with the NVIDIA Container Toolkit installed you can layer the GPU override file to request all GPUs automatically:
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```
Macs and other hosts without NVIDIA GPUs can simply skip the override file; the stack will run in CPU mode without errors.

## API reference
- `GET /healthz` – readiness probe.
- `GET /models` – list the registered models.
- `POST /inference` – execute inference with payload:
  ```json
  {
    "model_id": "blip-vqa-base",
    "prompt": "What is the driver doing?",
    "image_url": "https://example.com/sample.jpg"
  }
  ```
  Provide either `image_url` or `image_base64`.

## Notes
- The first inference call will download pretrained weights for YOLO and BLIP models. Keep the compose volume attached (or set `HUGGINGFACE_HUB_CACHE` / `ULTRALYTICS_CACHE_DIR`) to avoid repeated downloads.
- To force GPU acceleration, include `docker-compose.gpu.yml` as shown above; omit it (or remove the environment variables) for CPU-only execution.
