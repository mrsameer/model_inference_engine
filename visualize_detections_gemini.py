#!/usr/bin/env python3
"""
Script to use Gemini API for object detection and visualize detections with bounding boxes.
Outputs detections in the same format as YOLO models using Gemini's native bounding box detection.
"""

import os
import time
import requests
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from io import BytesIO
from dotenv import load_dotenv
from google import genai
from google.genai.types import (
    GenerateContentConfig,
    HarmBlockThreshold,
    HarmCategory,
    Part,
    SafetySetting,
    UploadFileConfig,
)
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# IMAGE_URL = "https://aspire.ap.gov.in/api/minio/download/stream?objectName=documents/852db639-eb52-4126-b009-c292d0fa8fc7/1000118035.jpg"
IMAGE_URL = "https://aspire.ap.gov.in/api/minio/download/stream?objectName=documents/4c15515a-97df-4eda-8616-3670a704d09e/1000118034.jpg"
MODEL_NAME = "gemini-2.5-flash"  # or "gemini-2.5-pro"

# Pest and disease configuration by crop type
PEST_CONFIGS = {
    "maize": {
        "pests": ["fall_army_worm"],
        "description": "Fall Army Worm on maize/corn crops",
        "detection_details": {
            "fall_army_worm": "Fall Army Worm larvae - look for caterpillars with distinctive inverted Y marking on head"
        },
        "colors": {"fall_army_worm": "red"}
    },
    "paddy": {
        "pests": ["sheath_blight", "brown_plant_hopper"],
        "description": "Sheath Blight disease and Brown Plant Hopper (BPH) on paddy/rice crops",
        "detection_details": {
            "sheath_blight": "Sheath Blight - fungal disease with oval/irregular lesions on leaf sheaths",
            "brown_plant_hopper": "Brown Plant Hopper (BPH) - small brown insects at base of rice plants"
        },
        "colors": {"sheath_blight": "orange", "brown_plant_hopper": "purple"}
    },
    "cotton": {
        "pests": ["pink_boll_worm", "white_fly"],
        "description": "Pink Boll Worm and White Fly on cotton crops",
        "detection_details": {
            "pink_boll_worm": "Pink Boll Worm - larvae or damage to cotton bolls with entry holes",
            "white_fly": "White Fly - tiny white insects, usually on underside of leaves"
        },
        "colors": {"pink_boll_worm": "magenta", "white_fly": "yellow"}
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
        "colors": {
            "fall_army_worm": "red",
            "sheath_blight": "orange",
            "brown_plant_hopper": "purple",
            "pink_boll_worm": "magenta",
            "white_fly": "yellow"
        }
    }
}

# Default crop type - change this or pass as parameter
CROP_TYPE = "maize"  # Options: "maize", "paddy", "cotton", "all"


# Pydantic model for Gemini's native bounding box format
class GeminiBoundingBox(BaseModel):
    """
    Represents a bounding box in Gemini's native format.

    Attributes:
        box_2d: Normalized coordinates [y_min, x_min, y_max, x_max] in range 0-1000
        label: Label for the detected object
        confidence: Confidence score (0-1)
    """
    box_2d: list[int]
    label: str
    confidence: float


def download_image(image_url):
    """Download the image from URL."""
    response = requests.get(image_url, timeout=30)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content))
    return image


def convert_gemini_to_yolo_format(gemini_boxes: list[GeminiBoundingBox], image_width: int, image_height: int):
    """
    Convert Gemini's normalized bounding boxes to YOLO-style pixel coordinates.

    Args:
        gemini_boxes: List of GeminiBoundingBox objects with normalized coords (0-1000)
        image_width: Width of the image in pixels
        image_height: Height of the image in pixels

    Returns:
        List of detections in YOLO format with pixel coordinates
    """
    detections = []
    for bbox in gemini_boxes:
        # Gemini format: [y_min, x_min, y_max, x_max] normalized to 0-1000
        y_min_norm, x_min_norm, y_max_norm, x_max_norm = bbox.box_2d

        # Convert to pixel coordinates
        x_min = int(x_min_norm / 1000 * image_width)
        y_min = int(y_min_norm / 1000 * image_height)
        x_max = int(x_max_norm / 1000 * image_width)
        y_max = int(y_max_norm / 1000 * image_height)

        # Create YOLO-style detection
        detection = {
            "label": bbox.label,
            "confidence": bbox.confidence,
            "box": {
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max
            }
        }
        detections.append(detection)

    return detections


def upload_image_to_gemini(client: genai.Client, image_url: str) -> tuple[str, Image.Image]:
    """
    Download image and upload to Gemini Files API.

    Args:
        client: Gemini client
        image_url: URL of the image to upload

    Returns:
        Tuple of (file_uri, PIL Image)
    """
    # Download the image
    image = download_image(image_url)

    # Convert image to bytes
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)

    # Upload to Gemini Files API
    uploaded_file = client.files.upload(
        file=img_byte_arr,
        config=UploadFileConfig(mime_type='image/jpeg')
    )
    file_uri = uploaded_file.uri

    return file_uri, image


def build_system_instruction(crop_type: str) -> str:
    """
    Build system instruction for Gemini based on crop type.

    Args:
        crop_type: Type of crop (maize, paddy, cotton, all)

    Returns:
        System instruction string
    """
    config = PEST_CONFIGS[crop_type]
    pests_list = config["pests"]
    detection_details = config["detection_details"]

    instruction = f"""
    You are an expert agricultural entomologist and plant pathologist specialized in detecting pests and diseases.

    CROP TYPE: {crop_type.upper()}
    DESCRIPTION: {config["description"]}

    DETECTION TARGETS:
    """

    for pest in pests_list:
        instruction += f"\n    - {pest}: {detection_details[pest]}"

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


def build_detection_prompt(crop_type: str) -> str:
    """
    Build detection prompt for Gemini based on crop type.

    Args:
        crop_type: Type of crop (maize, paddy, cotton, all)

    Returns:
        Detection prompt string
    """
    config = PEST_CONFIGS[crop_type]
    pests_list = ", ".join(config["pests"])
    return f"Detect all instances of {pests_list} in this agricultural image. Provide bounding boxes with labels and confidence scores."


def detect_with_gemini(image_url: str, api_key: str, crop_type: str = "maize"):
    """
    Use Gemini API with native bounding box detection.

    Args:
        image_url: URL of the image to analyze
        api_key: Gemini API key
        crop_type: Type of crop to detect pests for (maize, paddy, cotton, all)

    Returns:
        List of detections in YOLO format
    """
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file")

    if crop_type not in PEST_CONFIGS:
        raise ValueError(f"Invalid crop type: {crop_type}. Options: {list(PEST_CONFIGS.keys())}")

    # Initialize Gemini client
    client = genai.Client(api_key=api_key)

    # Upload image to Gemini Files API
    print("Uploading image to Gemini Files API...")
    file_uri, image = upload_image_to_gemini(client, image_url)
    width, height = image.size
    print(f"Image uploaded. Size: {width}x{height}")

    # Build system instruction and prompt for the crop type
    system_instruction = build_system_instruction(crop_type)
    detection_prompt = build_detection_prompt(crop_type)

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
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            Part.from_uri(file_uri=file_uri, mime_type="image/jpeg"),
            detection_prompt,
        ],
        config=config,
    )

    # Get the parsed structured output
    gemini_boxes = response.parsed

    # Convert to YOLO format
    detections = convert_gemini_to_yolo_format(gemini_boxes, width, height)

    return detections, image


def get_pest_base_name(label: str) -> str:
    """
    Extract the base pest name from a label with position descriptor.

    Args:
        label: Detection label (e.g., "fall_army_worm_center" or "fall_army_worm")

    Returns:
        Base pest name (e.g., "fall_army_worm")
    """
    # List of all possible base pest names
    all_pests = set()
    for config in PEST_CONFIGS.values():
        all_pests.update(config["pests"])

    # Check if the label starts with any known pest name
    for pest in all_pests:
        if label.startswith(pest):
            return pest

    # If no match, return the label as-is
    return label


def draw_detections(image, detections, crop_type: str = "maize"):
    """
    Draw bounding boxes on the image.

    Args:
        image: PIL Image to draw on
        detections: List of detections in YOLO format
        crop_type: Type of crop for color mapping

    Returns:
        Image with drawn bounding boxes
    """
    draw = ImageDraw.Draw(image)

    # Try to use a better font, fall back to default if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except:
        font = ImageFont.load_default()

    # Get colors from crop configuration
    colors = PEST_CONFIGS[crop_type]["colors"]

    for detection in detections:
        label = detection["label"]
        confidence = detection["confidence"]
        box = detection["box"]

        # Extract bounding box coordinates
        x_min = box["x_min"]
        y_min = box["y_min"]
        x_max = box["x_max"]
        y_max = box["y_max"]

        # Get base pest name for color lookup
        pest_name = get_pest_base_name(label)
        color = colors.get(pest_name, "green")

        # Draw bounding box
        draw.rectangle(
            [(x_min, y_min), (x_max, y_max)],
            outline=color,
            width=3
        )

        # Draw label with confidence
        text = f"{label}: {confidence:.2f}"

        # Draw text background
        bbox = draw.textbbox((x_min, y_min - 20), text, font=font)
        draw.rectangle(bbox, fill=color)

        # Draw text
        draw.text(
            (x_min, y_min - 20),
            text,
            fill="white",
            font=font
        )

    return image


def main(crop_type: str = None, image_url: str = None):
    """
    Main function to run Gemini detection and display results.

    Args:
        crop_type: Type of crop to detect pests for (maize, paddy, cotton, all)
        image_url: URL of the image to analyze
    """
    # Use defaults if not provided
    crop_type = crop_type or CROP_TYPE
    image_url = image_url or IMAGE_URL

    # Print configuration
    print(f"Using Gemini model: {MODEL_NAME}")
    print(f"Crop type: {crop_type}")
    print(f"Target pests: {', '.join(PEST_CONFIGS[crop_type]['pests'])}")
    print(f"Image URL: {image_url}")
    print()

    # Run Gemini detection
    print("Running Gemini detection with native bounding box API...")
    start_time = time.time()

    detections, image = detect_with_gemini(image_url, GEMINI_API_KEY, crop_type)

    duration_ms = (time.time() - start_time) * 1000

    # Print results in YOLO format
    print(f"Model: Gemini {MODEL_NAME}")
    print(f"Image size: {image.size}")
    print(f"Duration: {duration_ms:.2f} ms")
    print(f"Detections: {len(detections)}")
    print()

    for i, detection in enumerate(detections, 1):
        print(f"Detection {i}:")
        print(f"  Label: {detection['label']}")
        print(f"  Confidence: {detection['confidence']:.4f}")
        print(f"  Box: ({detection['box']['x_min']:.1f}, {detection['box']['y_min']:.1f}) to "
              f"({detection['box']['x_max']:.1f}, {detection['box']['y_max']:.1f})")
        print()

    if len(detections) == 0:
        print("No detections found!")
        # Display the original image
        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"Gemini {MODEL_NAME} ({crop_type}) - No detections")
        plt.tight_layout()
        plt.show()
        return

    # Draw detections
    print("Drawing detections...")
    image_with_boxes = draw_detections(image.copy(), detections, crop_type)

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(image_with_boxes)
    plt.axis('off')
    plt.title(f"Gemini {MODEL_NAME} ({crop_type}) - {len(detections)} detections")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import sys

    # Support command-line arguments
    # Usage: python visualize_detections_gemini.py [crop_type] [image_url]

    # Show help if requested
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print("=" * 80)
        print("GEMINI MULTI-PEST DETECTION")
        print("=" * 80)
        print("\nUsage:")
        print("  python visualize_detections_gemini.py [crop_type] [image_url]")
        print("\nCrop Types:")
        for crop, config in PEST_CONFIGS.items():
            print(f"\n  {crop}:")
            print(f"    Description: {config['description']}")
            print(f"    Pests: {', '.join(config['pests'])}")
        print("\nExamples:")
        print("  python visualize_detections_gemini.py")
        print("  python visualize_detections_gemini.py paddy")
        print("  python visualize_detections_gemini.py cotton https://example.com/image.jpg")
        print("  python visualize_detections_gemini.py all")
        print("\n" + "=" * 80)
        sys.exit(0)

    crop_arg = sys.argv[1] if len(sys.argv) > 1 else None
    image_arg = sys.argv[2] if len(sys.argv) > 2 else None

    # Validate crop type if provided
    if crop_arg and crop_arg not in PEST_CONFIGS:
        print(f"Error: Invalid crop type '{crop_arg}'")
        print(f"Available crop types: {', '.join(PEST_CONFIGS.keys())}")
        print("\nRun with '--help' for more information.")
        sys.exit(1)

    main(crop_type=crop_arg, image_url=image_arg)
