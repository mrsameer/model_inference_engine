#!/usr/bin/env python3
"""Test script for GeminiVLMRunner class."""

import asyncio
import os
import time
from io import BytesIO
from dotenv import load_dotenv
import requests
from PIL import Image

# Add parent directory to path
import sys
sys.path.insert(0, os.path.dirname(__file__))

# Import the components we need from main.py
from main import GeminiVLMRunner, ModelCard, ModelTask

# Load environment variables
load_dotenv()

# Test image URL
IMAGE_URL = "https://aspire.ap.gov.in/api/minio/download/stream?objectName=documents/4c15515a-97df-4eda-8616-3670a704d09e/1000118034.jpg"

async def test_gemini_vlm_runner():
    """Test the GeminiVLMRunner with a maize pest image."""
    print("=" * 80)
    print("TESTING GEMINI VLM RUNNER")
    print("=" * 80)
    print()

    # Create model card
    card = ModelCard(
        id="vlm_ss_test",
        name="Gemini VLM Test",
        description="Test Gemini VLM for pest detection",
        task=ModelTask.object_detection,
        framework="gemini",
        tags=["test"],
    )

    # Initialize runner
    print("Initializing GeminiVLMRunner...")
    runner = GeminiVLMRunner(
        card=card,
        model_name="gemini-2.5-flash",
        crop_type="all",
    )
    print("✓ Runner initialized")
    print()

    # Download test image
    print(f"Downloading test image from: {IMAGE_URL}")
    response = requests.get(IMAGE_URL, timeout=30)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content))
    print(f"✓ Image downloaded: {image.size}")
    print()

    # Run inference for different crop types
    test_cases = [
        ("maize", None),
        ("paddy", None),
        ("all", "maize"),  # Test with crop parameter
    ]

    for crop_type, crop_param in test_cases:
        print(f"Testing crop_type='{crop_type}', crop_param='{crop_param}'")
        print("-" * 80)

        # Update runner crop type
        runner.crop_type = crop_type

        start = time.perf_counter()
        try:
            result = await runner.infer(image=image, prompt=None, crop=crop_param)
            duration_ms = (time.perf_counter() - start) * 1000

            detections = result.get("detections", [])
            print(f"✓ Inference completed in {duration_ms:.2f}ms")
            print(f"  Detections: {len(detections)}")

            for i, detection in enumerate(detections, 1):
                print(f"  {i}. {detection.label}: {detection.confidence:.4f}")
                print(f"     Box: ({detection.box.x_min:.0f}, {detection.box.y_min:.0f}) -> ({detection.box.x_max:.0f}, {detection.box.y_max:.0f})")

            print()

        except Exception as e:
            print(f"❌ Error: {type(e).__name__}: {e}")
            print()
            import traceback
            traceback.print_exc()
            return False

    print("=" * 80)
    print("✓ ALL TESTS PASSED")
    print("=" * 80)
    return True

if __name__ == "__main__":
    success = asyncio.run(test_gemini_vlm_runner())
    exit(0 if success else 1)
