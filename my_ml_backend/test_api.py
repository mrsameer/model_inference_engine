"""
Test script for Label Studio ML Backend
"""

import requests
import json

# Configuration
BACKEND_URL = "http://localhost:5000"
TEST_IMAGE_URL = "https://apaims2.0.vassarlabs.com/staging/api/minio/download/stream?objectName=documents/ca9fa0f0-c6bb-41b6-9eee-c2763b0300ef/img108.jpg"

def test_health():
    """Test health endpoint"""
    print("Testing /health endpoint...")
    response = requests.get(f"{BACKEND_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_predict():
    """Test prediction endpoint"""
    print("Testing /predict endpoint...")

    # Create a Label Studio task
    task = {
        "data": {
            "image": TEST_IMAGE_URL,
            "crop": "maize"
        },
        "id": 1
    }

    # Prepare request payload (Label Studio format)
    payload = {
        "tasks": [task],
        "model_version": "vlm_ss"
    }

    print(f"Request payload:\n{json.dumps(payload, indent=2)}\n")

    # Make prediction request
    response = requests.post(
        f"{BACKEND_URL}/predict",
        json=payload,
        headers={"Content-Type": "application/json"}
    )

    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"Response:\n{json.dumps(result, indent=2)}")

        predictions = result.get("results", [])
        if predictions:
            num_detections = len(predictions[0].get("result", []))
            print(f"\n✅ Successfully received {num_detections} detections")
        else:
            print("\n⚠️  No predictions returned")
    else:
        print(f"Error: {response.text}")
    print()

def test_setup():
    """Test setup endpoint"""
    print("Testing /setup endpoint...")

    payload = {
        "project": "1",
        "schema": """
        <View>
          <Image name="image" value="$image"/>
          <RectangleLabels name="label" toName="image">
            <Label value="Fall Army Worm" background="red"/>
          </RectangleLabels>
        </View>
        """
    }

    response = requests.post(
        f"{BACKEND_URL}/setup",
        json=payload
    )

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {response.json()}")
    else:
        print(f"Error: {response.text}")
    print()

if __name__ == "__main__":
    print("=" * 60)
    print("Label Studio ML Backend Test Suite")
    print("=" * 60)
    print()

    try:
        test_health()
        test_setup()
        test_predict()

        print("=" * 60)
        print("✅ All tests completed!")
        print("=" * 60)

    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to ML backend")
        print(f"Make sure the backend is running at {BACKEND_URL}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
