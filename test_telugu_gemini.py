#!/usr/bin/env python3
"""Test script for Telugu language support in Gemini VLM."""

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
from main import GeminiVLMRunner, ModelCard, ModelTask, PEST_INFO

# Load environment variables
load_dotenv()

# Test image URL - maize with fall army worm
IMAGE_URL = "https://aspire.ap.gov.in/api/minio/download/stream?objectName=documents/4c15515a-97df-4eda-8616-3670a704d09e/1000118034.jpg"


async def test_telugu_support():
    """Test Telugu language support in GeminiVLMRunner."""
    print("=" * 80)
    print("TESTING TELUGU LANGUAGE SUPPORT IN GEMINI VLM")
    print("=" * 80)
    print()

    # Create model card
    card = ModelCard(
        id="vlm_ss_telugu_test",
        name="Gemini VLM Telugu Test",
        description="Test Gemini VLM with Telugu language support",
        task=ModelTask.object_detection,
        framework="gemini",
        tags=["test", "telugu"],
    )

    # Initialize runner
    print("Initializing GeminiVLMRunner...")
    runner = GeminiVLMRunner(
        card=card,
        model_name="gemini-2.5-flash",
        crop_type="maize",
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

    # Test both languages
    for language in ["en", "te"]:
        lang_name = "English" if language == "en" else "Telugu"
        print(f"Testing {lang_name} (language={language})")
        print("-" * 80)

        start = time.perf_counter()
        try:
            result = await runner.infer(
                image=image, prompt=None, crop="maize", language=language
            )
            duration_ms = (time.perf_counter() - start) * 1000

            detections = result.get("detections", [])
            print(f"✓ Inference completed in {duration_ms:.2f}ms")
            print(f"  Detections: {len(detections)}")

            for i, detection in enumerate(detections, 1):
                print(f"  {i}. {detection.label}: {detection.confidence:.4f}")

            # Build answers using PEST_INFO (simulating the API behavior)
            if detections:
                unique_pests = set()
                for detection in detections:
                    label = detection.label.lower()
                    for pest_name in PEST_INFO[language].keys():
                        if label.startswith(pest_name) or pest_name in label:
                            unique_pests.add(pest_name)
                            break

                if unique_pests:
                    print()
                    print(f"  {lang_name} Pest Information:")
                    for pest_name in sorted(unique_pests):
                        if pest_name in PEST_INFO[language]:
                            info = PEST_INFO[language][pest_name]
                            print(f"  Name: {info['name']}")
                            print(f"  Description: {info['description'][:100]}...")
                            print(f"  Remedies: {info['remedies'][:100]}...")
                            print()

        except Exception as e:
            print(f"❌ Error: {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()
            return False

        print()

    print("=" * 80)
    print("✓ TELUGU LANGUAGE SUPPORT TEST PASSED")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = asyncio.run(test_telugu_support())
    exit(0 if success else 1)
