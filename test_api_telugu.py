#!/usr/bin/env python3
"""Example API request showing Telugu language support."""

import json

# Example API request for English
english_request = {
    "model_id": "vlm_ss",
    "user_id": "test_user_123",
    "crop": "maize",
    "language": "en",  # English
    "image_url": "https://aspire.ap.gov.in/api/minio/download/stream?objectName=documents/4c15515a-97df-4eda-8616-3670a704d09e/1000118034.jpg"
}

# Example API request for Telugu
telugu_request = {
    "model_id": "vlm_ss",
    "user_id": "test_user_123",
    "crop": "maize",
    "language": "te",  # Telugu
    "image_url": "https://aspire.ap.gov.in/api/minio/download/stream?objectName=documents/4c15515a-97df-4eda-8616-3670a704d09e/1000118034.jpg"
}

print("=" * 80)
print("EXAMPLE API REQUESTS WITH LANGUAGE SUPPORT")
print("=" * 80)
print()

print("1. English Request:")
print("-" * 80)
print("POST http://localhost:8000/inference")
print("Content-Type: application/json")
print()
print(json.dumps(english_request, indent=2))
print()

print("Expected Response (English):")
print("-" * 80)
print("""
{
  "model": {
    "id": "vlm_ss",
    "name": "Multi-Pest-Disease Detector (VLM) SS",
    ...
  },
  "duration_ms": 3827.65,
  "detections": [
    {
      "label": "fall_army_worm",
      "confidence": 0.98,
      "box": {
        "x_min": 146.0,
        "y_min": 194.0,
        "x_max": 372.0,
        "y_max": 337.0
      }
    }
  ],
  "answers": [
    {
      "answer": "**Fall Army Worm**\\n\\nFall Army Worm is a destructive pest that attacks maize crops...\\n\\n**Remedies:**\\nApply neem-based bio-pesticides...",
      "confidence": 1.0
    }
  ]
}
""")

print()
print("2. Telugu Request:")
print("-" * 80)
print("POST http://localhost:8000/inference")
print("Content-Type: application/json")
print()
print(json.dumps(telugu_request, indent=2))
print()

print("Expected Response (Telugu):")
print("-" * 80)
print("""
{
  "model": {
    "id": "vlm_ss",
    "name": "Multi-Pest-Disease Detector (VLM) SS",
    ...
  },
  "duration_ms": 2055.21,
  "detections": [
    {
      "label": "fall_army_worm",
      "confidence": 0.98,
      "box": {
        "x_min": 146.0,
        "y_min": 194.0,
        "x_max": 372.0,
        "y_max": 337.0
      }
    }
  ],
  "answers": [
    {
      "answer": "**ఫాల్ ఆర్మీ వార్మ్**\\n\\nఫాల్ ఆర్మీ వార్మ్ మొక్కజొన్న పంటలపై దాడి చేసే విధ్వంసక తెగులు...\\n\\n**నివారణలు:**\\nవేప ఆధారిత జీవ-పురుగుమందులు...",
      "confidence": 1.0
    }
  ]
}
""")

print()
print("=" * 80)
print("KEY CHANGES:")
print("=" * 80)
print("1. Added 'language' parameter to InferenceRequest (default: 'en')")
print("2. Supported languages: 'en' (English), 'te' (Telugu)")
print("3. Pest names, descriptions, and remedies returned in requested language")
print("4. Detection labels remain in English (fall_army_worm, etc.)")
print("5. Language parameter validated - falls back to 'en' if invalid")
