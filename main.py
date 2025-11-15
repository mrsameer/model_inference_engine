from __future__ import annotations

import base64
import io
import logging
import os
import threading
import time
from enum import Enum
from typing import Any, Callable, Dict, List

import httpx
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field, HttpUrl, model_validator


logger = logging.getLogger("model_inference_engine")
logging.basicConfig(level=logging.INFO)


class ModelTask(str, Enum):
    """High-level task supported by the inference engine."""

    object_detection = "object-detection"
    vision_language = "vision-language"


class ModelCard(BaseModel):
    id: str = Field(..., description="Stable identifier for the model")
    name: str = Field(..., description="Friendly name for the model")
    description: str
    task: ModelTask
    framework: str
    tags: List[str] = Field(default_factory=list)
    default_prompt: str | None = Field(
        default=None, description="Optional default prompt for the model"
    )
    capabilities: List[str] = Field(
        default_factory=list, description="Key capabilities supported by the model"
    )


class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class Detection(BaseModel):
    label: str
    confidence: float
    box: BoundingBox


class VisionLanguageAnswer(BaseModel):
    answer: str
    confidence: float


class InferenceRequest(BaseModel):
    model_id: str = Field(..., description="Identifier returned from /models")
    prompt: str | None = Field(
        default=None,
        description="Optional text prompt or question for vision-language models",
    )
    image_url: HttpUrl | None = Field(
        default=None, description="Remote image to fetch before inference"
    )
    image_base64: str | None = Field(
        default=None, description="Base64 encoded image payload"
    )

    @model_validator(mode="after")
    def validate_payload(self):  # type: ignore[override]
        if not (self.image_url or self.image_base64):
            raise ValueError("Provide either image_url or image_base64")
        return self


class InferenceResponse(BaseModel):
    model: ModelCard
    duration_ms: float = Field(..., gt=0)
    detections: List[Detection] | None = None
    answers: List[VisionLanguageAnswer] | None = None


class BaseModelRunner:
    """Abstract runtime for a model."""

    def __init__(self, card: ModelCard):
        self.card = card

    def infer(self, image: Image.Image, prompt: str | None) -> Dict[str, Any]:
        raise NotImplementedError


class YoloRunner(BaseModelRunner):
    """Runs YOLO object detection models via Ultralytics."""

    def __init__(self, card: ModelCard, weights: str, conf_threshold: float = 0.25):
        super().__init__(card)
        self.weights = weights
        self.conf_threshold = conf_threshold
        self._model = None
        self._lock = threading.Lock()

    def _ensure_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from ultralytics import YOLO  # pylint: disable=import-error

                    cache_dir = os.getenv("ULTRALYTICS_CACHE_DIR")
                    if cache_dir:
                        os.environ.setdefault("YOLO_CACHE_DIR", cache_dir)
                    logger.info("Loading YOLO weights: %s", self.weights)
                    self._model = YOLO(self.weights)
        return self._model

    def infer(self, image: Image.Image, prompt: str | None) -> Dict[str, List[Detection]]:  # noqa: ARG002
        model = self._ensure_model()
        results = model.predict(image, conf=self.conf_threshold, verbose=False)
        if not results:
            return {"detections": []}

        names = getattr(model, "names", {})
        detections: List[Detection] = []
        boxes = getattr(results[0], "boxes", None)
        if boxes is None:
            return {"detections": detections}

        xyxys = boxes.xyxy.tolist()
        confs = boxes.conf.tolist()
        classes = boxes.cls.tolist()
        for coords, score, cls_idx in zip(xyxys, confs, classes):
            confidence = float(score)
            if confidence < self.conf_threshold:
                continue
            label = names.get(int(cls_idx), str(int(cls_idx))) if isinstance(names, dict) else names[int(cls_idx)]
            detections.append(
                Detection(
                    label=label,
                    confidence=confidence,
                    box=BoundingBox(
                        x_min=float(coords[0]),
                        y_min=float(coords[1]),
                        x_max=float(coords[2]),
                        y_max=float(coords[3]),
                    ),
                )
            )
        return {"detections": detections}


class VisionLanguageRunner(BaseModelRunner):
    """Runs visual question answering models using Hugging Face pipelines."""

    def __init__(self, card: ModelCard, model_name: str, top_k: int = 3):
        super().__init__(card)
        self.model_name = model_name
        self.top_k = top_k
        self._pipeline = None
        self._lock = threading.Lock()

    def _ensure_pipeline(self):
        if self._pipeline is None:
            with self._lock:
                if self._pipeline is None:
                    from transformers import pipeline  # pylint: disable=import-error

                    logger.info("Loading VLM pipeline: %s", self.model_name)
                    self._pipeline = pipeline(
                        "visual-question-answering",
                        model=self.model_name,
                        tokenizer=self.model_name,
                    )
        return self._pipeline

    def infer(self, image: Image.Image, prompt: str | None) -> Dict[str, List[VisionLanguageAnswer]]:
        qa_pipeline = self._ensure_pipeline()
        question = prompt or "Describe the image"
        outputs = qa_pipeline(image=image, question=question, top_k=self.top_k)
        answers = [
            VisionLanguageAnswer(answer=item["answer"], confidence=float(item.get("score", 0.0)))
            for item in outputs
        ]
        return {"answers": answers}


class ModelRegistry:
    """Registry for supported inference models and their runners."""

    def __init__(self):
        self._models: Dict[str, ModelCard] = {}
        self._factories: Dict[str, Callable[[], BaseModelRunner]] = {}
        self._runners: Dict[str, BaseModelRunner] = {}
        self._lock = threading.Lock()

    def register(self, card: ModelCard, factory: Callable[[], BaseModelRunner]):
        self._models[card.id] = card
        self._factories[card.id] = factory

    def list_models(self) -> List[ModelCard]:
        return list(self._models.values())

    def get_card(self, model_id: str) -> ModelCard:
        try:
            return self._models[model_id]
        except KeyError as exc:  # pragma: no cover - handled by /models
            raise KeyError(f"Unknown model '{model_id}'") from exc

    def get_runner(self, model_id: str) -> BaseModelRunner:
        if model_id not in self._models:
            raise KeyError(f"Unknown model '{model_id}'")
        runner = self._runners.get(model_id)
        if runner:
            return runner
        with self._lock:
            runner = self._runners.get(model_id)
            if runner is None:
                factory = self._factories[model_id]
                runner = factory()
                self._runners[model_id] = runner
        return runner


registry = ModelRegistry()


def register_default_models():
    yolo_n = ModelCard(
        id="yolov8n",
        name="YOLOv8 nano",
        description="Ultralytics YOLOv8 nano object detector optimized for CPU demos",
        task=ModelTask.object_detection,
        framework="ultralytics",
        tags=["yolo", "object-detection", "real-time"],
        capabilities=["object-detection", "bounding-box"],
    )
    registry.register(yolo_n, lambda card=yolo_n: YoloRunner(card=card, weights="yolov8n.pt"))

    yolo_s = ModelCard(
        id="yolov8s",
        name="YOLOv8 small",
        description="Ultralytics YOLOv8 small model for higher accuracy detections",
        task=ModelTask.object_detection,
        framework="ultralytics",
        tags=["yolo", "object-detection"],
        capabilities=["object-detection", "bounding-box"],
    )
    registry.register(yolo_s, lambda card=yolo_s: YoloRunner(card=card, weights="yolov8s.pt"))

    vlm = ModelCard(
        id="blip-vqa-base",
        name="BLIP VQA Base",
        description="Salesforce BLIP model for visual question answering and captions",
        task=ModelTask.vision_language,
        framework="transformers",
        tags=["vlm", "vision-language", "question-answering"],
        default_prompt="Describe the image",
        capabilities=["captioning", "visual-question-answering"],
    )
    registry.register(
        vlm,
        lambda card=vlm: VisionLanguageRunner(
            card=card,
            model_name="Salesforce/blip-vqa-base",
        ),
    )

    # Fall Army Worm Detection Model
    pest_faw = ModelCard(
        id="pest_fall_army_warm_ss",
        name="Fall Army Worm Detector (YOLOv8s)",
        description="Fine-tuned YOLOv8 small model for detecting fall army worms on crop leaves. Optimized for agricultural pest detection with 40% mAP50 on validation data.",
        task=ModelTask.object_detection,
        framework="ultralytics",
        tags=["pest-detection", "yolo", "fall-army-worm", "agriculture", "crop-monitoring"],
        capabilities=["object-detection", "bounding-box", "pest-detection"],
    )
    registry.register(
        pest_faw,
        lambda card=pest_faw: YoloRunner(
            card=card,
            weights="models/pest_fall_army_warm_ss.pt",
            conf_threshold=0.25,
        ),
    )


register_default_models()

app = FastAPI(
    title="Model Inference Engine",
    description=(
        "Unified API for YOLO object detection and vision-language inference "
        "models packaged for containerized deployments."
    ),
    version="0.2.0",
)


def _load_image(payload: InferenceRequest) -> Image.Image:
    if payload.image_base64:
        return _decode_base64_image(payload.image_base64)
    if payload.image_url:
        return _download_image(payload.image_url)
    raise ValueError("No image payload supplied")


def _decode_base64_image(data: str) -> Image.Image:
    try:
        _, _, encoded = data.partition(",")
        encoded = encoded or data
        binary = base64.b64decode(encoded, validate=True)
        image = Image.open(io.BytesIO(binary))
        return image.convert("RGB")
    except Exception as exc:  # pragma: no cover - defensive guard
        raise ValueError("Invalid base64 image payload") from exc


def _download_image(url: HttpUrl) -> Image.Image:
    try:
        with httpx.Client(follow_redirects=True, timeout=15) as client:
            response = client.get(str(url))
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise ValueError(f"Unable to download image: {exc}") from exc

    try:
        image = Image.open(io.BytesIO(response.content))
        return image.convert("RGB")
    except Exception as exc:  # pragma: no cover - defensive guard
        raise ValueError("Downloaded data is not a valid image") from exc


@app.get("/")
async def root():
    return {
        "name": "model-inference-engine",
        "version": "0.2.0",
        "models": [card.id for card in registry.list_models()],
    }


@app.get("/healthz")
async def health():
    return {"status": "ok"}


@app.get("/models", response_model=List[ModelCard])
async def get_models():
    return registry.list_models()


@app.post("/inference", response_model=InferenceResponse)
async def run_inference(payload: InferenceRequest):
    try:
        runner = registry.get_runner(payload.model_id)
        card = registry.get_card(payload.model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        image = _load_image(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    start = time.perf_counter()
    try:
        outputs = runner.infer(image=image, prompt=payload.prompt)
    except Exception as exc:  # pragma: no cover - runtime safeguard
        logger.exception("Inference failed: %s", exc)
        raise HTTPException(status_code=500, detail="Inference failed") from exc

    duration_ms = (time.perf_counter() - start) * 1000
    return InferenceResponse(
        model=card,
        duration_ms=round(duration_ms, 2),
        detections=outputs.get("detections"),
        answers=outputs.get("answers"),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", reload=False)
