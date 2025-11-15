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
The default compose file runs the CPU stack on port `5000` and mounts the `models-cache` volume to `/models` so downloaded YOLO weights and Hugging Face checkpoints are persisted between restarts.
```bash
docker compose up --build
```

### GPU acceleration (optional)
On machines with the NVIDIA Container Toolkit installed you can start the GPU stack (port `5001`) side-by-side with the CPU version. GPU-only workflow:
```bash
docker compose -f docker-compose.gpu.yml up --build
```

To run both variants at once (CPU on `5000`, GPU on `5001`) bring up the combined stack:
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```
Macs and other hosts without NVIDIA GPUs can simply skip the GPU file; the CPU stack will continue to run without errors.

### GPU troubleshooting
If the GPU stack crashes with messages such as `CUDA error: device-side assert triggered` or `compile with TORCH_USE_CUDA_DSA to enable device-side assertions`, follow the checklist below to narrow things down:
1. **Validate the host driver** – run `nvidia-smi` on the host and make sure the driver is recent enough for CUDA 12.x (535+ for Ampere, 550+ for Ada). GPU containers inherit the host driver – an outdated driver is the most common cause of mysterious CUDA crashes.
2. **Confirm the container sees the GPU** – `docker compose -f docker-compose.gpu.yml run --rm inference-api-gpu python - <<'PY'` followed by a short snippet prints the CUDA runtime info:
   ```python
   import torch
   print('torch:', torch.__version__)
   print('cuda:', torch.version.cuda)
   print('available:', torch.cuda.is_available())
   print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')
   ```
3. **Reinstall matching PyTorch wheels** – if the container reports `Torch not compiled with CUDA` install the CUDA build explicitly (inside the container):
   ```bash
   pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 torch==2.3.1 torchvision==0.18.1 --upgrade --force-reinstall
   ```
   Rebuild the image afterwards so the cached layer includes the GPU enabled wheels.
   *Compatibility note:* the CUDA runtime baked into the wheel has to be **newer than or equal to** the version exposed by `nvidia-smi`. For example, the default PyTorch wheel we picked up recently (`2.9.1+cu128`) expects CUDA 12.8, but the host driver `535.261.03` only provides CUDA 12.2, so the kernel immediately fails. Either upgrade the host driver to a CUDA 12.8 capable release (550+) or install a wheel built for CUDA 12.1/12.2 (e.g. `--index-url ... cu121` as shown above).
4. **Use the debugging knobs** – the GPU override compose file exposes `TORCH_USE_CUDA_DSA`, `CUDA_LAUNCH_BLOCKING`, and `PYTORCH_CUDA_ALLOC_CONF`. Export them before running compose to enable extra diagnostics, e.g.:
   ```bash
   export TORCH_USE_CUDA_DSA=1
   export CUDA_LAUNCH_BLOCKING=1
   docker compose -f docker-compose.gpu.yml up --build
   ```

If the crash persists even after the above, temporarily force CPU execution by unsetting `CUDA_VISIBLE_DEVICES` (or removing the GPU override file) while you continue investigating the GPU runtime.

### Ultralytics config path
Ultralytics writes a small `settings.yaml` file to `~/.config/Ultralytics` on startup. Containers sometimes run without a writable home directory, which surfaces as `config directory /root/.config/Ultralytics is not writable`. The image now sets `YOLO_CONFIG_DIR`, `ULTRALYTICS_SETTINGS_DIR`, and `ULTRALYTICS_CONFIG_DIR` to `/tmp/ultralytics`, ensuring the library always has a safe scratch directory. If you prefer to persist those settings, override any of the above environment variables in `docker-compose.yml` to point at a writable folder/volume, for example `/models/ultralytics/config`.

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
