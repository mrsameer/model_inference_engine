#!/usr/bin/env python3
"""Test script to verify language tracking in inference history."""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from main import init_database, log_inference_to_db, get_user_history, Detection, VisionLanguageAnswer, BoundingBox

async def test_history_language_tracking():
    """Test that language is properly stored and retrieved in history."""
    print("=" * 80)
    print("TESTING LANGUAGE TRACKING IN INFERENCE HISTORY")
    print("=" * 80)
    print()

    # Initialize database
    print("Initializing database...")
    await init_database()
    print("✓ Database initialized")
    print()

    # Test data
    test_user_id = "test_farmer_lang_123"

    # Log English inference
    print("Logging English inference...")
    await log_inference_to_db(
        user_id=test_user_id,
        model_id="vlm_ss",
        crop="maize",
        language="en",
        image_source="url",
        image_url="https://example.com/test1.jpg",
        prompt=None,
        duration_ms=3500.0,
        detections=[
            Detection(
                label="fall_army_worm",
                confidence=0.98,
                box=BoundingBox(x_min=146.0, y_min=194.0, x_max=372.0, y_max=337.0)
            )
        ],
        answers=[
            VisionLanguageAnswer(
                answer="**Fall Army Worm**\n\nFall Army Worm is a destructive pest...",
                confidence=1.0
            )
        ],
    )
    print("✓ English inference logged")
    print()

    # Log Telugu inference
    print("Logging Telugu inference...")
    await log_inference_to_db(
        user_id=test_user_id,
        model_id="vlm_ss",
        crop="paddy",
        language="te",
        image_source="url",
        image_url="https://example.com/test2.jpg",
        prompt=None,
        duration_ms=2100.0,
        detections=[
            Detection(
                label="sheath_blight",
                confidence=0.92,
                box=BoundingBox(x_min=100.0, y_min=150.0, x_max=300.0, y_max=350.0)
            )
        ],
        answers=[
            VisionLanguageAnswer(
                answer="**షీత్ బ్లైట్**\n\nషీత్ బ్లైట్ అనేది రైజోక్టోనియా సొలానీ...",
                confidence=1.0
            )
        ],
    )
    print("✓ Telugu inference logged")
    print()

    # Retrieve history
    print("Retrieving user history...")
    total, items = await get_user_history(test_user_id, page=1, page_size=10)
    print(f"✓ Retrieved {len(items)} items (total: {total})")
    print()

    # Verify language tracking
    print("Verifying language tracking:")
    print("-" * 80)

    success = True
    for i, item in enumerate(items, 1):
        print(f"{i}. Inference {item.id}:")
        print(f"   Crop: {item.crop}")
        print(f"   Language: {item.language}")
        print(f"   Model: {item.model_id}")
        print(f"   Detections: {item.detections_count}")
        print(f"   Duration: {item.duration_ms}ms")

        # Verify language field exists and is correct
        if item.language not in ["en", "te"]:
            print(f"   ❌ ERROR: Invalid language '{item.language}'")
            success = False
        else:
            print(f"   ✓ Language tracking working")

        # Check if answers reflect the correct language
        if item.answers:
            answer_preview = item.answers[0].answer[:50]
            print(f"   Answer preview: {answer_preview}...")

            # Simple check: Telugu text should contain Telugu characters
            has_telugu = any('\u0C00' <= char <= '\u0C7F' for char in item.answers[0].answer)
            if item.language == "te" and not has_telugu:
                print(f"   ⚠ WARNING: Telugu language but no Telugu characters found")
            elif item.language == "en" and has_telugu:
                print(f"   ⚠ WARNING: English language but Telugu characters found")
            else:
                print(f"   ✓ Answer language matches request")

        print()

    if success:
        print("=" * 80)
        print("✓ LANGUAGE TRACKING IN HISTORY TEST PASSED")
        print("=" * 80)
        return True
    else:
        print("=" * 80)
        print("❌ LANGUAGE TRACKING TEST FAILED")
        print("=" * 80)
        return False

if __name__ == "__main__":
    success = asyncio.run(test_history_language_tracking())
    exit(0 if success else 1)
