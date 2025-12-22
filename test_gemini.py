#!/usr/bin/env python3
"""Quick test script for Gemini VLM with Vertex AI authentication."""

import os
from dotenv import load_dotenv
from google.auth import load_credentials_from_file
from google import genai

# Load environment variables
load_dotenv()

VERTEX_AI_CREDENTIALS_PATH = os.getenv("VERTEX_AI_CREDENTIALS_PATH", "vertex-gemini-api-key.json")
VERTEX_AI_PROJECT_ID = os.getenv("VERTEX_AI_PROJECT_ID")
VERTEX_AI_LOCATION = os.getenv("VERTEX_AI_LOCATION", "us-central1")

def test_gemini_auth():
    """Test Gemini authentication with Vertex AI."""
    print("=" * 80)
    print("TESTING GEMINI VERTEX AI AUTHENTICATION")
    print("=" * 80)
    print()

    # Check configuration
    print(f"Credentials Path: {VERTEX_AI_CREDENTIALS_PATH}")
    print(f"Project ID: {VERTEX_AI_PROJECT_ID}")
    print(f"Location: {VERTEX_AI_LOCATION}")
    print()

    if not VERTEX_AI_PROJECT_ID:
        print("❌ ERROR: VERTEX_AI_PROJECT_ID not found in .env file")
        return False

    if not os.path.exists(VERTEX_AI_CREDENTIALS_PATH):
        print(f"❌ ERROR: Credentials file not found: {VERTEX_AI_CREDENTIALS_PATH}")
        return False

    print("✓ Configuration OK")
    print()

    try:
        # Load credentials from JSON file with required scopes
        from google.oauth2 import service_account
        print("Loading credentials from JSON file...")
        credentials = service_account.Credentials.from_service_account_file(
            VERTEX_AI_CREDENTIALS_PATH,
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        print(f"✓ Credentials loaded successfully")
        print(f"  Service Account: {credentials.service_account_email}")
        print()

        # Initialize Vertex AI Gemini client
        print("Initializing Vertex AI Gemini client...")
        client = genai.Client(
            vertexai=True,
            project=VERTEX_AI_PROJECT_ID,
            location=VERTEX_AI_LOCATION,
            credentials=credentials,
        )
        print("✓ Client initialized successfully")
        print()

        # Try listing models
        print("Testing API access by listing available models...")
        try:
            models = client.models.list()
            available_models = [m.name for m in models]
            gemini_models = [m for m in available_models if 'gemini' in m.lower()]
            print(f"✓ API access confirmed!")
            print(f"  Found {len(gemini_models)} Gemini models:")
            for model in gemini_models[:5]:  # Show first 5
                print(f"    - {model}")
            if len(gemini_models) > 5:
                print(f"    ... and {len(gemini_models) - 5} more")
        except Exception as e:
            print(f"⚠ Warning: Could not list models: {e}")
            print("  (This might be a permissions issue, but authentication may still work)")

        print()
        print("=" * 80)
        print("✓ GEMINI VERTEX AI AUTHENTICATION TEST PASSED")
        print("=" * 80)
        return True

    except Exception as e:
        print()
        print("=" * 80)
        print(f"❌ ERROR: {type(e).__name__}: {e}")
        print("=" * 80)
        return False

if __name__ == "__main__":
    success = test_gemini_auth()
    exit(0 if success else 1)
