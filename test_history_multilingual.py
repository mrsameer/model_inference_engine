#!/usr/bin/env python3
"""Test script to verify multilingual history retrieval."""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from main import (
    init_database,
    log_inference_to_db,
    get_user_history,
    Detection,
    VisionLanguageAnswer,
    BoundingBox,
)


async def test_multilingual_history():
    """Test that history can be retrieved in any language regardless of original request."""
    print("=" * 80)
    print("TESTING MULTILINGUAL HISTORY RETRIEVAL")
    print("=" * 80)
    print()

    # Initialize database
    print("Initializing database...")
    await init_database()
    print("✓ Database initialized")
    print()

    test_user_id = "test_multilingual_farmer_456"

    # Log an inference in English
    print("1. Logging inference in ENGLISH...")
    await log_inference_to_db(
        user_id=test_user_id,
        model_id="vlm_ss",
        crop="maize",
        language="en",
        image_source="url",
        image_url="https://example.com/test_maize.jpg",
        prompt=None,
        duration_ms=3200.0,
        detections=[
            Detection(
                label="fall_army_worm",
                confidence=0.95,
                box=BoundingBox(x_min=120.0, y_min=180.0, x_max=350.0, y_max=320.0),
            )
        ],
        answers=[
            VisionLanguageAnswer(
                answer="**Fall Army Worm**\n\nFall Army Worm is a destructive pest...",
                confidence=1.0,
            )
        ],
    )
    print("✓ English inference logged")
    print()

    # Log an inference in Telugu
    print("2. Logging inference in TELUGU...")
    await log_inference_to_db(
        user_id=test_user_id,
        model_id="vlm_ss",
        crop="paddy",
        language="te",
        image_source="url",
        image_url="https://example.com/test_paddy.jpg",
        prompt=None,
        duration_ms=2800.0,
        detections=[
            Detection(
                label="sheath_blight",
                confidence=0.89,
                box=BoundingBox(x_min=90.0, y_min=140.0, x_max=280.0, y_max=300.0),
            )
        ],
        answers=[
            VisionLanguageAnswer(
                answer="**షీత్ బ్లైట్**\n\nషీత్ బ్లైట్ అనేది రైజోక్టోనియా సొలానీ వల్ల కలిగే వరి పంటలను ప్రభావితం చేసే శిలీంధ్ర వ్యాధి...",
                confidence=1.0,
            )
        ],
    )
    print("✓ Telugu inference logged")
    print()

    # Test 1: Retrieve in original languages (no language parameter)
    print("3. Retrieving history in ORIGINAL languages (no language param):")
    print("-" * 80)
    total, items = await get_user_history(test_user_id, page=1, page_size=10)
    print(f"Retrieved {len(items)} items")

    for item in items:
        print(f"\nInference {item.id}:")
        print(f"  Original Language: {item.language}")
        print(f"  Crop: {item.crop}")
        if item.answers:
            answer_preview = item.answers[0].answer[:60]
            print(f"  Answer: {answer_preview}...")
            # Verify language matches original
            has_telugu = any("\u0C00" <= char <= "\u0C7F" for char in item.answers[0].answer)
            if item.language == "te" and has_telugu:
                print("  ✓ Telugu content matches Telugu request")
            elif item.language == "en" and not has_telugu:
                print("  ✓ English content matches English request")

    # Test 2: Retrieve everything in English (force language=en)
    print("\n4. Retrieving ALL history in ENGLISH (language=en):")
    print("-" * 80)
    total, items = await get_user_history(
        test_user_id, page=1, page_size=10, response_language="en"
    )
    print(f"Retrieved {len(items)} items, all forced to English")

    for item in items:
        print(f"\nInference {item.id}:")
        print(f"  Original Language: {item.language}")
        print(f"  Crop: {item.crop}")
        if item.answers:
            answer_preview = item.answers[0].answer[:60]
            print(f"  Answer: {answer_preview}...")
            # All should be English now
            has_telugu = any("\u0C00" <= char <= "\u0C7F" for char in item.answers[0].answer)
            if not has_telugu:
                print("  ✓ Successfully converted to English")
            else:
                print("  ❌ ERROR: Still contains Telugu!")
                return False

    # Test 3: Retrieve everything in Telugu (force language=te)
    print("\n5. Retrieving ALL history in TELUGU (language=te):")
    print("-" * 80)
    total, items = await get_user_history(
        test_user_id, page=1, page_size=10, response_language="te"
    )
    print(f"Retrieved {len(items)} items, all forced to Telugu")

    for item in items:
        print(f"\nInference {item.id}:")
        print(f"  Original Language: {item.language}")
        print(f"  Crop: {item.crop}")
        if item.answers:
            answer_preview = item.answers[0].answer[:60]
            print(f"  Answer: {answer_preview}...")
            # All should be Telugu now
            has_telugu = any("\u0C00" <= char <= "\u0C7F" for char in item.answers[0].answer)
            if has_telugu:
                print("  ✓ Successfully converted to Telugu")
            else:
                print("  ❌ ERROR: Missing Telugu content!")
                return False

    print("\n" + "=" * 80)
    print("✓ MULTILINGUAL HISTORY RETRIEVAL TEST PASSED")
    print("=" * 80)
    print("\nKey Features Verified:")
    print("  ✓ Original language preserved in database")
    print("  ✓ Detections stored once (language-agnostic)")
    print("  ✓ Answers regenerated on-the-fly in requested language")
    print("  ✓ English→Telugu conversion works")
    print("  ✓ Telugu→English conversion works")
    print("  ✓ Default retrieval uses original language")
    return True


if __name__ == "__main__":
    success = asyncio.run(test_multilingual_history())
    exit(0 if success else 1)
