"""Test script to run inference on a local directory of images."""

import base64
import time
from pathlib import Path

import httpx

BASE_URL = "https://aspire.ap.gov.in/inference"
DIRECTORY_PATH = "/Users/shaiksameer/Downloads/fall_army_worm_folder"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"}


def test_directory_inference():
    dir_path = Path(DIRECTORY_PATH)
    if not dir_path.exists():
        print(f"Directory not found: {DIRECTORY_PATH}")
        return

    image_files = [
        f for f in dir_path.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not image_files:
        print(f"No image files found in: {DIRECTORY_PATH}")
        return

    print(f"Testing inference on {len(image_files)} images from: {DIRECTORY_PATH}")
    print("-" * 60)

    successful = 0
    failed = 0
    total_start = time.perf_counter()

    with httpx.Client(timeout=600) as client:
        for image_file in sorted(image_files):
            start = time.perf_counter()
            try:
                # Read and encode image as base64
                with open(image_file, "rb") as f:
                    img_base64 = base64.b64encode(f.read()).decode("utf-8")

                payload = {
                    "model_id": "qwen25_vllm",
                    "image_base64": img_base64,
                    "crop": "maize",
                    "prompt": "Detect any pests in this image",
                }

                response = client.post(f"{BASE_URL}/inference", json=payload)
                duration_ms = (time.perf_counter() - start) * 1000

                if response.status_code != 200:
                    print(f"\n✗ {image_file.name} ({duration_ms:.2f}ms)")
                    print(f"  Error: {response.status_code} - {response.text[:200]}")
                    failed += 1
                    continue

                data = response.json()
                successful += 1
                print(f"\n✓ {image_file.name} ({duration_ms:.2f}ms)")

                detections = data.get("detections") or []
                if detections:
                    print(f"  Detections: {len(detections)}")
                    for det in detections:
                        print(f"    - {det['label']} ({det['confidence']:.2f})")

                answers = data.get("answers") or []
                if answers:
                    for ans in answers:
                        preview = (
                            ans["answer"][:100] + "..."
                            if len(ans["answer"]) > 100
                            else ans["answer"]
                        )
                        print(f"  Answer: {preview}")

            except Exception as exc:
                duration_ms = (time.perf_counter() - start) * 1000
                print(f"\n✗ {image_file.name} ({duration_ms:.2f}ms)")
                print(f"  Error: {exc}")
                failed += 1

    total_duration_ms = (time.perf_counter() - total_start) * 1000
    print("\n" + "-" * 60)
    print(f"Total: {len(image_files)} | Successful: {successful} | Failed: {failed}")
    print(f"Total duration: {total_duration_ms:.2f}ms")


if __name__ == "__main__":
    test_directory_inference()
