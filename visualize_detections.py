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
MODEL_ID = "pest_fall_army_warm_ss"
IMAGE_URL = "https://aspire.ap.gov.in/api/minio/download/stream?objectName=documents/852db639-eb52-4126-b009-c292d0fa8fc7/1000118035.jpg"

def call_inference_api(model_id, image_url):
    """Call the inference API and return the response."""
    payload = {
        "model_id": model_id,
        "image_url": image_url
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
    except:
        font = ImageFont.load_default()

    # Define colors for different labels
    colors = {
        "fall_army_worm": "red"
    }

    for detection in detections:
        label = detection["label"]
        confidence = detection["confidence"]
        box = detection["box"]

        # Extract bounding box coordinates
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
