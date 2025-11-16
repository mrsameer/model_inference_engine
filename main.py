from __future__ import annotations

import base64
import io
import json
import logging
import os
import sqlite3
import tempfile
import threading
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field, HttpUrl, model_validator

# Load environment variables for API keys
load_dotenv()


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
    user_id: str | None = Field(
        default=None,
        description="Optional user identifier for tracking inference requests",
    )
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


class InferenceHistoryItem(BaseModel):
    id: int
    user_id: str | None
    model_id: str
    image_source: str
    image_url: str | None
    prompt: str | None
    duration_ms: float
    detections_count: int
    detections: List[Detection] | None = None
    answers: List[VisionLanguageAnswer] | None = None
    created_at: str


class InferenceHistoryResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: List[InferenceHistoryItem]


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


# Static descriptions and remedies for each pest/disease
PEST_INFO = {
    "fall_army_worm": {
        "description": "Fall Army Worm is a destructive pest that attacks maize crops. Larvae feed on leaves, creating characteristic ragged holes and windows in the foliage. The caterpillars have an inverted Y-shaped marking on their head capsule.",
        "remedies": "Apply neem-based bio-pesticides or chemical insecticides like Emamectin Benzoate. Use pheromone traps for early detection. Practice crop rotation and intercropping with non-host plants. Remove and destroy infested plants."
    },
    "sheath_blight": {
        "description": "Sheath Blight is a fungal disease affecting paddy crops caused by Rhizoctonia solani. It appears as oval or irregular greenish-gray lesions on leaf sheaths, which later turn brown with a dark brown border. Can cause significant yield loss.",
        "remedies": "Use resistant varieties. Apply fungicides like Validamycin or Hexaconazole at early infection stages. Maintain proper plant spacing for air circulation. Avoid excessive nitrogen fertilization. Practice field sanitation by removing crop debris."
    },
    "brown_plant_hopper": {
        "description": "Brown Plant Hopper (BPH) is a serious insect pest of rice that feeds on plant sap, causing hopperburn - yellowing and drying of plants. They are small brown insects typically found at the base of rice plants and can transmit viral diseases.",
        "remedies": "Use resistant rice varieties. Apply neem oil or insecticides like Imidacloprid or Fipronil. Avoid excessive nitrogen application. Maintain proper water management. Use light traps for monitoring. Encourage natural predators like spiders and mirid bugs."
    },
    "pink_boll_worm": {
        "description": "Pink Boll Worm is a major pest of cotton that attacks cotton bolls. Larvae bore into bolls causing damage to developing seeds and lint. Entry holes are visible on bolls, often with frass (insect waste). Severely affected bolls fail to open properly.",
        "remedies": "Use pheromone traps for mass trapping and monitoring. Plant Bt cotton varieties. Apply chemical insecticides like Cypermethrin during flowering and boll formation. Practice clean cultivation by destroying crop residues. Follow recommended spacing and avoid late season irrigation."
    },
    "white_fly": {
        "description": "White Fly is a tiny sap-sucking insect pest commonly found on cotton and other crops. Adults and nymphs feed on the underside of leaves, causing yellowing, leaf curling, and reduced plant vigor. They also secrete honeydew, leading to sooty mold growth.",
        "remedies": "Use yellow sticky traps for monitoring and control. Apply neem-based products or chemical insecticides like Acetamiprid or Spiromesifen. Encourage natural predators like ladybird beetles and lacewings. Practice crop rotation. Remove heavily infested leaves. Maintain field sanitation."
    }
}


class GeminiVLMRunner(BaseModelRunner):
    """Runs Gemini Vision Language Model for object detection with bounding boxes."""

    # Pest and disease configuration by crop type
    PEST_CONFIGS = {
        "maize": {
            "pests": ["fall_army_worm"],
            "description": "Fall Army Worm on maize/corn crops",
            "detection_details": {
                "fall_army_worm": "Fall Army Worm larvae - look for caterpillars with distinctive inverted Y marking on head"
            },
        },
        "paddy": {
            "pests": ["sheath_blight", "brown_plant_hopper"],
            "description": "Sheath Blight disease and Brown Plant Hopper (BPH) on paddy/rice crops",
            "detection_details": {
                "sheath_blight": "Sheath Blight - fungal disease with oval/irregular lesions on leaf sheaths",
                "brown_plant_hopper": "Brown Plant Hopper (BPH) - small brown insects at base of rice plants"
            },
        },
        "cotton": {
            "pests": ["pink_boll_worm", "white_fly"],
            "description": "Pink Boll Worm and White Fly on cotton crops",
            "detection_details": {
                "pink_boll_worm": "Pink Boll Worm - larvae or damage to cotton bolls with entry holes",
                "white_fly": "White Fly - tiny white insects, usually on underside of leaves"
            },
        },
        "all": {
            "pests": ["fall_army_worm", "sheath_blight", "brown_plant_hopper", "pink_boll_worm", "white_fly"],
            "description": "All supported pests and diseases across maize, paddy, and cotton crops",
            "detection_details": {
                "fall_army_worm": "Fall Army Worm larvae on maize",
                "sheath_blight": "Sheath Blight disease on paddy",
                "brown_plant_hopper": "Brown Plant Hopper on paddy",
                "pink_boll_worm": "Pink Boll Worm on cotton",
                "white_fly": "White Fly on cotton"
            },
        }
    }

    def __init__(self, card: ModelCard, model_name: str = "gemini-2.5-flash", crop_type: str = "all"):
        super().__init__(card)
        self.model_name = model_name
        self.crop_type = crop_type
        self._client = None
        self._lock = threading.Lock()
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not found in environment. Gemini VLM will not work.")

    def _ensure_client(self):
        """Initialize Gemini client lazily."""
        if self._client is None:
            with self._lock:
                if self._client is None:
                    from google import genai  # pylint: disable=import-error

                    if not self.api_key:
                        raise ValueError("GEMINI_API_KEY not found in .env file")

                    logger.info("Initializing Gemini client with model: %s", self.model_name)
                    self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _build_system_instruction(self, crop_type: str) -> str:
        """Build system instruction for Gemini based on crop type."""
        config = self.PEST_CONFIGS[crop_type]
        pests_list = config["pests"]
        detection_details = config["detection_details"]

        instruction = f"""
        You are an expert agricultural entomologist and plant pathologist specialized in detecting pests and diseases.

        CROP TYPE: {crop_type.upper()}
        DESCRIPTION: {config["description"]}

        DETECTION TARGETS:
        """

        for pest in pests_list:
            instruction += f"\n        - {pest}: {detection_details[pest]}"

        instruction += """

        INSTRUCTIONS:
        - Return bounding boxes as an array with labels and confidence scores
        - Never return masks. Limit to 25 objects.
        - Only detect the pests/diseases listed above - do not detect other objects
        - If an object is present multiple times, give each a unique label with position descriptor
        - Use label format: pest_name (e.g., "fall_army_worm", "sheath_blight") followed by position if multiple
        - Confidence should reflect detection certainty (0.0 to 1.0)
        - Only include detections with confidence > 0.5
        - Be especially careful to distinguish between different pest types
        - For diseases, detect visible symptoms like lesions, discoloration, or damage patterns
        """

        return instruction

    def _build_detection_prompt(self, crop_type: str) -> str:
        """Build detection prompt for Gemini based on crop type."""
        config = self.PEST_CONFIGS[crop_type]
        pests_list = ", ".join(config["pests"])
        return f"Detect all instances of {pests_list} in this agricultural image. Provide bounding boxes with labels and confidence scores."

    def _upload_image_to_gemini(self, client, image: Image.Image) -> str:
        """Upload image to Gemini Files API."""
        from google.genai.types import UploadFileConfig  # pylint: disable=import-error

        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)

        # Upload to Gemini Files API
        uploaded_file = client.files.upload(
            file=img_byte_arr,
            config=UploadFileConfig(mime_type='image/jpeg')
        )
        return uploaded_file.uri

    def _get_pest_base_name(self, label: str) -> str:
        """Extract the base pest name from a label with position descriptor."""
        all_pests = set()
        for config in self.PEST_CONFIGS.values():
            all_pests.update(config["pests"])

        for pest in all_pests:
            if label.startswith(pest):
                return pest

        return label

    def infer(self, image: Image.Image, prompt: str | None) -> Dict[str, List[Detection]]:  # noqa: ARG002
        """Run Gemini VLM inference for object detection."""
        from google.genai.types import (  # pylint: disable=import-error
            GenerateContentConfig,
            HarmBlockThreshold,
            HarmCategory,
            Part,
            SafetySetting,
        )
        from pydantic import BaseModel as PydanticBaseModel  # pylint: disable=import-error

        # Determine crop type from prompt if provided
        crop_type = self.crop_type
        if prompt:
            prompt_lower = prompt.lower()
            for crop in self.PEST_CONFIGS.keys():
                if crop in prompt_lower:
                    crop_type = crop
                    logger.info("Detected crop type from prompt: %s", crop_type)
                    break

        if crop_type not in self.PEST_CONFIGS:
            logger.warning("Invalid crop type '%s', defaulting to 'all'", crop_type)
            crop_type = "all"

        # Pydantic model for Gemini's native bounding box format
        class GeminiBoundingBox(PydanticBaseModel):
            box_2d: list[int]
            label: str
            confidence: float

        client = self._ensure_client()

        # Upload image to Gemini Files API
        logger.info("Uploading image to Gemini Files API...")
        file_uri = self._upload_image_to_gemini(client, image)
        width, height = image.size
        logger.info("Image uploaded. Size: %dx%d", width, height)

        # Build system instruction and prompt for the crop type
        system_instruction = self._build_system_instruction(crop_type)
        detection_prompt = self._build_detection_prompt(crop_type)

        # Configure the model with system instructions and structured output
        config = GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.3,
            safety_settings=[
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
                ),
            ],
            response_mime_type="application/json",
            response_schema=list[GeminiBoundingBox],
        )

        # Generate content with structured output
        logger.info("Running Gemini %s detection for crop: %s", self.model_name, crop_type)
        response = client.models.generate_content(
            model=self.model_name,
            contents=[
                Part.from_uri(file_uri=file_uri, mime_type="image/jpeg"),
                detection_prompt,
            ],
            config=config,
        )

        # Get the parsed structured output
        gemini_boxes = response.parsed

        # Convert Gemini's normalized bounding boxes to YOLO-style pixel coordinates
        detections: List[Detection] = []
        for bbox in gemini_boxes:
            # Gemini format: [y_min, x_min, y_max, x_max] normalized to 0-1000
            y_min_norm, x_min_norm, y_max_norm, x_max_norm = bbox.box_2d

            # Convert to pixel coordinates
            x_min = int(x_min_norm / 1000 * width)
            y_min = int(y_min_norm / 1000 * height)
            x_max = int(x_max_norm / 1000 * width)
            y_max = int(y_max_norm / 1000 * height)

            # Create YOLO-style detection
            detections.append(
                Detection(
                    label=bbox.label,
                    confidence=bbox.confidence,
                    box=BoundingBox(
                        x_min=float(x_min),
                        y_min=float(y_min),
                        x_max=float(x_max),
                        y_max=float(y_max),
                    ),
                )
            )

        logger.info("Gemini detected %d pests/diseases", len(detections))
        return {"detections": detections}


# Database configuration and initialization
DB_PATH = os.getenv("INFERENCE_DB_PATH", "./storage/inference.db")


def init_database():
    """Initialize SQLite database with inference logging table."""
    db_path = Path(DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS inference_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            model_id TEXT NOT NULL,
            image_source TEXT NOT NULL,
            image_url TEXT,
            prompt TEXT,
            duration_ms REAL NOT NULL,
            detections_count INTEGER DEFAULT 0,
            detections_json TEXT,
            answers_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create index for faster user-based queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_user_id
        ON inference_logs(user_id)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_created_at
        ON inference_logs(created_at)
    """)

    conn.commit()
    conn.close()
    logger.info("Database initialized at: %s", db_path)


def log_inference_to_db(
    user_id: str | None,
    model_id: str,
    image_source: str,
    image_url: str | None,
    prompt: str | None,
    duration_ms: float,
    detections: List[Detection] | None,
    answers: List[VisionLanguageAnswer] | None,
):
    """Log inference request and results to database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        detections_count = len(detections) if detections else 0
        detections_json = json.dumps([d.model_dump() for d in detections]) if detections else None
        answers_json = json.dumps([a.model_dump() for a in answers]) if answers else None

        cursor.execute("""
            INSERT INTO inference_logs
            (user_id, model_id, image_source, image_url, prompt, duration_ms,
             detections_count, detections_json, answers_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            model_id,
            image_source,
            image_url,
            prompt,
            duration_ms,
            detections_count,
            detections_json,
            answers_json,
        ))

        conn.commit()
        conn.close()
        logger.info("Logged inference to database: user=%s, model=%s, detections=%d",
                   user_id, model_id, detections_count)
    except Exception as exc:
        logger.error("Failed to log inference to database: %s", exc)


def get_user_history(
    user_id: str,
    page: int = 1,
    page_size: int = 10
) -> tuple[int, List[InferenceHistoryItem]]:
    """Retrieve inference history for a user with pagination."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get total count
        cursor.execute("""
            SELECT COUNT(*) as total
            FROM inference_logs
            WHERE user_id = ?
        """, (user_id,))
        total = cursor.fetchone()["total"]

        # Get paginated results
        offset = (page - 1) * page_size
        cursor.execute("""
            SELECT id, user_id, model_id, image_source, image_url, prompt,
                   duration_ms, detections_count, detections_json, answers_json,
                   created_at
            FROM inference_logs
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """, (user_id, page_size, offset))

        rows = cursor.fetchall()
        conn.close()

        items = []
        for row in rows:
            detections = None
            if row["detections_json"]:
                detections_data = json.loads(row["detections_json"])
                detections = [Detection(**d) for d in detections_data]

            answers = None
            if row["answers_json"]:
                answers_data = json.loads(row["answers_json"])
                answers = [VisionLanguageAnswer(**a) for a in answers_data]

            items.append(InferenceHistoryItem(
                id=row["id"],
                user_id=row["user_id"],
                model_id=row["model_id"],
                image_source=row["image_source"],
                image_url=row["image_url"],
                prompt=row["prompt"],
                duration_ms=row["duration_ms"],
                detections_count=row["detections_count"],
                detections=detections,
                answers=answers,
                created_at=row["created_at"],
            ))

        return total, items
    except Exception as exc:
        logger.error("Failed to retrieve user history: %s", exc)
        raise


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

    # Gemini VLM Multi-Pest Detection Model
    vlm_ss = ModelCard(
        id="vlm_ss",
        name="Gemini Multi-Pest Detector (VLM)",
        description=(
            "Gemini 2.5 Vision Language Model with native bounding box detection for agricultural pest identification. "
            "Supports Fall Army Worm (maize), Sheath Blight & Brown Plant Hopper (paddy), "
            "and Pink Boll Worm & White Fly (cotton). Zero-shot detection with flexible prompting. "
            "Specify crop type in prompt (e.g., 'maize', 'paddy', 'cotton', or 'all')."
        ),
        task=ModelTask.object_detection,
        framework="gemini",
        tags=["vlm", "vision-language", "pest-detection", "multi-crop", "zero-shot", "agriculture"],
        default_prompt="all",
        capabilities=[
            "object-detection",
            "bounding-box",
            "pest-detection",
            "disease-detection",
            "multi-crop",
            "zero-shot",
            "flexible-prompting"
        ],
    )
    registry.register(
        vlm_ss,
        lambda card=vlm_ss: GeminiVLMRunner(
            card=card,
            model_name="gemini-2.5-flash",
            crop_type="all",
        ),
    )


register_default_models()

# Initialize database on startup
init_database()

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


@app.get("/history/{user_id}", response_model=InferenceHistoryResponse)
async def get_history(
    user_id: str,
    page: int = 1,
    page_size: int = 10
):
    """
    Get inference history for a specific user with pagination.

    Args:
        user_id: User identifier
        page: Page number (default: 1)
        page_size: Number of items per page (default: 10, max: 100)

    Returns:
        Paginated inference history
    """
    if page < 1:
        raise HTTPException(status_code=400, detail="Page must be >= 1")

    if page_size < 1 or page_size > 100:
        raise HTTPException(status_code=400, detail="Page size must be between 1 and 100")

    try:
        total, items = get_user_history(user_id, page, page_size)
        return InferenceHistoryResponse(
            total=total,
            page=page,
            page_size=page_size,
            items=items
        )
    except Exception as exc:
        logger.exception("Failed to retrieve history for user %s: %s", user_id, exc)
        raise HTTPException(status_code=500, detail="Failed to retrieve history") from exc


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

    # Add pest/disease information to answers for pest detection models
    answers = outputs.get("answers")
    detections = outputs.get("detections")

    # For both YOLO-based and Gemini-based pest detection models
    if detections and (isinstance(runner, (GeminiVLMRunner, YoloRunner))):
        # Extract unique pest types from detections
        unique_pests = set()
        for detection in detections:
            # Get base pest name (strip position descriptors)
            label = detection.label.lower()
            # Check if this label matches any known pest
            for pest_name in PEST_INFO.keys():
                if label.startswith(pest_name) or pest_name in label:
                    unique_pests.add(pest_name)
                    break

        # Build answer with descriptions and remedies for detected pests
        if unique_pests:
            answers = []
            for pest_name in sorted(unique_pests):
                if pest_name in PEST_INFO:
                    info = PEST_INFO[pest_name]
                    answer_text = f"**{pest_name.replace('_', ' ').title()}**\n\n"
                    answer_text += f"Description: {info['description']}\n\n"
                    answer_text += f"Remedies: {info['remedies']}"

                    answers.append(VisionLanguageAnswer(
                        answer=answer_text,
                        confidence=1.0
                    ))

    # Log to database
    image_source = "base64" if payload.image_base64 else "url"
    image_url_str = str(payload.image_url) if payload.image_url else None
    log_inference_to_db(
        user_id=payload.user_id,
        model_id=payload.model_id,
        image_source=image_source,
        image_url=image_url_str,
        prompt=payload.prompt,
        duration_ms=round(duration_ms, 2),
        detections=detections,
        answers=answers,
    )

    return InferenceResponse(
        model=card,
        duration_ms=round(duration_ms, 2),
        detections=detections,
        answers=answers,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", reload=False)
