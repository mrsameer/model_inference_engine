#!/usr/bin/env python3
"""
Simple script to call the inference server and visualize detections with bounding boxes.
"""

import requests
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from io import BytesIO

# Configuration
INFERENCE_URL = "https://aspire.ap.gov.in/inference/inference"
MODEL_ID = ("qwen25_vllm")
IMAGE_URL = "http://minio.mahanidan.vassarlabs.com/minio/ui/api/v1/download-shared-object/aHR0cDovLzEyNy4wLjAuMTo5MDAwL21haXplLWZhbGxhcm15d29ybS9pbWFnZS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD03OVBKRExDNjM4TTRVWkw2UldITiUyRjIwMjUxMjI5JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MTIyOVQxMTM1NDhaJlgtQW16LUV4cGlyZXM9NDMxOTkmWC1BbXotU2VjdXJpdHktVG9rZW49ZXlKaGJHY2lPaUpJVXpVeE1pSXNJblI1Y0NJNklrcFhWQ0o5LmV5SmhZMk5sYzNOTFpYa2lPaUkzT1ZCS1JFeEROak00VFRSVldrdzJVbGRJVGlJc0ltVjRjQ0k2TVRjMk56QTFNRGswTml3aWNHRnlaVzUwSWpvaWJXbHVhVzloWkcxcGJpSjkuZGVNMFBRVVhmQUE2MVhlU19mRkRRVWhXU2xDOFN1bFc0Y1dlSFh6dGd1UVI0U2N4RFllS1BYSkhlT1pzVENTc0dmXzdBS1VGR2RQdXc2Q2d1VnowZGcmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JnZlcnNpb25JZD1udWxsJlgtQW16LVNpZ25hdHVyZT0xOWQyMTBjNDUyOTk0OGYxMGJjYjQ4ZGU2ZTc0Njk3Y2E5ODJjZTJlYTI3MDgyNzk1MzdhYTNiNzdiNjc1NDMw"
# IMAGE_URL = "http://minio.mahanidan.vassarlabs.com/minio/ui/api/v1/download-shared-object/aHR0cDovLzEyNy4wLjAuMTo5MDAwL2NvdHRvbi13aGl0ZWZseS8xMDkzMzYuanBnP1gtQW16LUFsZ29yaXRobT1BV1M0LUhNQUMtU0hBMjU2JlgtQW16LUNyZWRlbnRpYWw9NzlQSkRMQzYzOE00VVpMNlJXSE4lMkYyMDI1MTIyOSUyRnVzLWVhc3QtMSUyRnMzJTJGYXdzNF9yZXF1ZXN0JlgtQW16LURhdGU9MjAyNTEyMjlUMTEzMzM3WiZYLUFtei1FeHBpcmVzPTQzMTk5JlgtQW16LVNlY3VyaXR5LVRva2VuPWV5SmhiR2NpT2lKSVV6VXhNaUlzSW5SNWNDSTZJa3BYVkNKOS5leUpoWTJObGMzTkxaWGtpT2lJM09WQktSRXhETmpNNFRUUlZXa3cyVWxkSVRpSXNJbVY0Y0NJNk1UYzJOekExTURrME5pd2ljR0Z5Wlc1MElqb2liV2x1YVc5aFpHMXBiaUo5LmRlTTBQUVVYZkFBNjFYZVNfZkZEUVVoV1NsQzhTdWxXNGNXZUhYenRndVFSNFNjeERZZUtQWEpIZU9ac1RDU3NHZl83QUtVRkdkUHV3NkNndVZ6MGRnJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZ2ZXJzaW9uSWQ9bnVsbCZYLUFtei1TaWduYXR1cmU9ZmU2NWVkNjRhYTE3OTI1N2E5MzMwNDE1ZGNiZTFiNzA0MTllNDk4Mzc4MjEyYTY4YjhhODFjMGQ5MGI0MTcwYw"
# IMAGE_URL = "http://minio.mahanidan.vassarlabs.com/minio/ui/api/v1/download-shared-object/aHR0cDovLzEyNy4wLjAuMTo5MDAwL3BhZGR5LXJpY2VsZWFmLWFwL0VBMDMyNX4xLkpQRz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPTc5UEpETEM2MzhNNFVaTDZSV0hOJTJGMjAyNTEyMjklMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMjI5VDExMzA1OVomWC1BbXotRXhwaXJlcz00MzE5OSZYLUFtei1TZWN1cml0eS1Ub2tlbj1leUpoYkdjaU9pSklVelV4TWlJc0luUjVjQ0k2SWtwWFZDSjkuZXlKaFkyTmxjM05MWlhraU9pSTNPVkJLUkV4RE5qTTRUVFJWV2t3MlVsZElUaUlzSW1WNGNDSTZNVGMyTnpBMU1EazBOaXdpY0dGeVpXNTBJam9pYldsdWFXOWhaRzFwYmlKOS5kZU0wUFFVWGZBQTYxWGVTX2ZGRFFVaFdTbEM4U3VsVzRjV2VIWHp0Z3VRUjRTY3hEWWVLUFhKSGVPWnNUQ1NzR2ZfN0FLVUZHZFB1dzZDZ3VWejBkZyZYLUFtei1TaWduZWRIZWFkZXJzPWhvc3QmdmVyc2lvbklkPW51bGwmWC1BbXotU2lnbmF0dXJlPTdmNjNhZjJkOGQxYjA5YjkzNDgwM2VmOTUwOThjMTg4YWMwOTI0N2Y0N2M5MjU5NjBiYjc1ZmJlNWEwMzk2MDQ"
# IMAGE_URL = "https://apaims2.0.vassarlabs.com/staging/api/minio/download/stream?objectName=documents/b3b90e37-b510-4a37-aa00-d217842e3875/test.jpg"
# IMAGE_URL = "https://minio.apaims2.0.vassarlabs.com/minio/ui/api/v1/download-shared-object/aHR0cDovLzEyNy4wLjAuMTo5MDAwL2FwYWltcy10ZXN0L2ltZzIuanBnP1gtQW16LUFsZ29yaXRobT1BV1M0LUhNQUMtU0hBMjU2JlgtQW16LUNyZWRlbnRpYWw9TlVFN085UkQ2REMzV1pUWEY5UEglMkYyMDI1MTExNiUyRnVzLWVhc3QtMSUyRnMzJTJGYXdzNF9yZXF1ZXN0JlgtQW16LURhdGU9MjAyNTExMTZUMDUxNzU3WiZYLUFtei1FeHBpcmVzPTQzMTg5JlgtQW16LVNlY3VyaXR5LVRva2VuPWV5SmhiR2NpT2lKSVV6VXhNaUlzSW5SNWNDSTZJa3BYVkNKOS5leUpoWTJObGMzTkxaWGtpT2lKT1ZVVTNUemxTUkRaRVF6TlhXbFJZUmpsUVNDSXNJbVY0Y0NJNk1UYzJNek14TXpNNU1Dd2ljR0Z5Wlc1MElqb2liV2x1YVc5aFpHMXBiaUo5LkhnMUh0aDZ0anc3MDRhMXVycFNpM2hMRVZaZUtvOUFwYm15SmhpLWR6NThLVjhXekJaZWtyd2JycDJBV3dhWEV2SXRtdG8tOEd1S3RjdUNQRHhkWDVRJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZ2ZXJzaW9uSWQ9bnVsbCZYLUFtei1TaWduYXR1cmU9ODU0ODBkN2IxNGM0MTQ1YTQxZjczZjFkNjllNmVhNzQwYTFjYWEwZjkzY2I3NzFlZDJhNDVlNjNkN2U3MWMzOA"
# IMAGE_URL = "https://minio.apaims2.0.vassarlabs.com/minio/ui/api/v1/download-shared-object/aHR0cDovLzEyNy4wLjAuMTo5MDAwL2FwYWltcy10ZXN0L2NkZTk2ZTMwLTQ1ZTYtNGI1OS1iOGQ4LTQzNmI2NjFkM2EwOV8wYjcyNWQwYi0xZjc3LTRlYmItOGQ2ZC00NTZiYmEyNjRhYjNfaW1nMjEuanBnP1gtQW16LUFsZ29yaXRobT1BV1M0LUhNQUMtU0hBMjU2JlgtQW16LUNyZWRlbnRpYWw9TlVFN085UkQ2REMzV1pUWEY5UEglMkYyMDI1MTExNiUyRnVzLWVhc3QtMSUyRnMzJTJGYXdzNF9yZXF1ZXN0JlgtQW16LURhdGU9MjAyNTExMTZUMDUyMTUwWiZYLUFtei1FeHBpcmVzPTQzMjAwJlgtQW16LVNlY3VyaXR5LVRva2VuPWV5SmhiR2NpT2lKSVV6VXhNaUlzSW5SNWNDSTZJa3BYVkNKOS5leUpoWTJObGMzTkxaWGtpT2lKT1ZVVTNUemxTUkRaRVF6TlhXbFJZUmpsUVNDSXNJbVY0Y0NJNk1UYzJNek14TXpNNU1Dd2ljR0Z5Wlc1MElqb2liV2x1YVc5aFpHMXBiaUo5LkhnMUh0aDZ0anc3MDRhMXVycFNpM2hMRVZaZUtvOUFwYm15SmhpLWR6NThLVjhXekJaZWtyd2JycDJBV3dhWEV2SXRtdG8tOEd1S3RjdUNQRHhkWDVRJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZ2ZXJzaW9uSWQ9bnVsbCZYLUFtei1TaWduYXR1cmU9ZDA4M2QzZDliZTE3MjVjZDBlODZhODJiZTIxNjZiZTQ5ZTczYzQ1MWYwZThjYWQxMmI3Nzc4ZDhjNDA5ZmYxMA"
# IMAGE_URL = "https://minio.apaims2.0.vassarlabs.com/minio/ui/api/v1/download-shared-object/aHR0cDovLzEyNy4wLjAuMTo5MDAwL2FwYWltcy10ZXN0L2NkZTk2ZTMwLTQ1ZTYtNGI1OS1iOGQ4LTQzNmI2NjFkM2EwOV8yZTk4MTBiMC02YWMwLTQyZDctYmMxYi01ZTFiNjI1ODliN2NfaW1nNzguanBnP1gtQW16LUFsZ29yaXRobT1BV1M0LUhNQUMtU0hBMjU2JlgtQW16LUNyZWRlbnRpYWw9TlVFN085UkQ2REMzV1pUWEY5UEglMkYyMDI1MTExNiUyRnVzLWVhc3QtMSUyRnMzJTJGYXdzNF9yZXF1ZXN0JlgtQW16LURhdGU9MjAyNTExMTZUMDUyNTQ3WiZYLUFtei1FeHBpcmVzPTQzMjAwJlgtQW16LVNlY3VyaXR5LVRva2VuPWV5SmhiR2NpT2lKSVV6VXhNaUlzSW5SNWNDSTZJa3BYVkNKOS5leUpoWTJObGMzTkxaWGtpT2lKT1ZVVTNUemxTUkRaRVF6TlhXbFJZUmpsUVNDSXNJbVY0Y0NJNk1UYzJNek14TXpNNU1Dd2ljR0Z5Wlc1MElqb2liV2x1YVc5aFpHMXBiaUo5LkhnMUh0aDZ0anc3MDRhMXVycFNpM2hMRVZaZUtvOUFwYm15SmhpLWR6NThLVjhXekJaZWtyd2JycDJBV3dhWEV2SXRtdG8tOEd1S3RjdUNQRHhkWDVRJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZ2ZXJzaW9uSWQ9bnVsbCZYLUFtei1TaWduYXR1cmU9NWE5Zjg4MzMwNGEyMzZiZjdjZDAzNjIxM2NlOGEwZDAyY2U1YmIyMjY4ZDVlZTAyODQ4NjU1ZTU1MjkxOTBiOA"
# IMAGE_URL = "https://apaims2.0.vassarlabs.com/staging/api/minio/download/stream?objectName=documents/9a4f1719-f022-45f0-9bce-bf0d4ae5e20a/img66.jpg"
# IMAGE_URL = "https://apaims2.0.vassarlabs.com/staging/api/minio/download/stream?objectName=documents/ca9fa0f0-c6bb-41b6-9eee-c2763b0300ef/img108.jpg"

def call_inference_api(model_id, image_url):
    """Call the inference API and return the response."""
    payload = {
        "model_id": model_id,
        "image_url": image_url,
        "crop": "maize",
        "task": "pest",
        "user_id": "shahen"
    }
    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(INFERENCE_URL, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()

def download_image(image_url):
    """Download the image from URL."""
    response = requests.get(image_url)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content))
    return image

def draw_detections(image, detections):
    """Draw bounding boxes on the image."""
    draw = ImageDraw.Draw(image)

    # Try to use a better font, fall back to default if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except Exception:
        font = ImageFont.load_default()

    # Define colors for different labels
    colors = {
        "fall_army_worm": "red"
    }

    for detection in detections:
        label = detection["label"]
        confidence = detection["confidence"]
        box = detection["box"]

        # Extract bounding box coordinates (already in pixel coordinates from API)
        x_min = box["x_min"]
        y_min = box["y_min"]
        x_max = box["x_max"]
        y_max = box["y_max"]

        # Get color for this label
        color = colors.get(label, "green")

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

def main():
    """Main function to run the inference and display results."""
    print(f"Calling inference API for model: {MODEL_ID}")
    print(f"Image URL: {IMAGE_URL}")
    print()

    # Call the inference API
    result = call_inference_api(MODEL_ID, IMAGE_URL)

    # Print results
    print(f"Model: {result['model']['name']}")
    print(f"Duration: {result['duration_ms']:.2f} ms")
    print(f"Detections: {len(result['detections'])}")
    print()

    for i, detection in enumerate(result['detections'], 1):
        print(f"Detection {i}:")
        print(f"  Label: {detection['label']}")
        print(f"  Confidence: {detection['confidence']:.4f}")
        print(f"  Box: ({detection['box']['x_min']:.1f}, {detection['box']['y_min']:.1f}) to "
              f"({detection['box']['x_max']:.1f}, {detection['box']['y_max']:.1f})")
        print()

    # Download and display the image
    print("Downloading image...")
    image = download_image(IMAGE_URL)

    # Draw detections
    print("Drawing detections...")
    image_with_boxes = draw_detections(image, result['detections'])

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(image_with_boxes)
    plt.axis('off')
    plt.title(f"{result['model']['name']} - {len(result['detections'])} detections")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
