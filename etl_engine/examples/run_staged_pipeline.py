"""
Demo: 3-stage ML inference pipeline with per-stage parallelism

  Stage 1 – Download     (CPU / network I/O)  parallelism = 2
  Stage 2 – Prediction   (GPU / compute)       parallelism = 1
  Stage 3 – PostProcess  (CPU)                 parallelism = 1

Run:
    cd /home/user/etl_engine
    python -m examples.run_staged_pipeline
"""
import sys
import os
import time
import random
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from etl.staged_pipeline import Stage, StagedPipeline

# ── Helpers ───────────────────────────────────────────────────────────────────

def _log(stage: str, item_id: int, msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    tid = threading.current_thread().name
    print(f"  {ts}  [{stage:<14}] [{tid:<24}]  item={item_id:02d}  {msg}")


# ── Stage functions ───────────────────────────────────────────────────────────

def download(item: dict) -> dict:
    """Simulate downloading raw bytes from a remote source (network I/O)."""
    delay = random.uniform(0.3, 0.7)
    _log("DOWNLOAD", item["id"], f"fetching url={item['url']}  …  ({delay:.2f}s)")
    time.sleep(delay)
    payload = f"<raw_bytes_for_{item['id']}>"
    _log("DOWNLOAD", item["id"], f"done  →  {len(payload)} bytes")
    return {**item, "raw": payload}


def predict(item: dict) -> dict:
    """Simulate GPU inference on downloaded data (compute-bound)."""
    delay = random.uniform(0.5, 0.9)   # GPU is slower but deterministic
    _log("PREDICT", item["id"], f"running model on raw data  …  ({delay:.2f}s)")
    time.sleep(delay)
    score = round(random.uniform(0.5, 0.99), 4)
    label = "cat" if score > 0.75 else "dog"
    _log("PREDICT", item["id"], f"done  →  label={label}  score={score}")
    return {**item, "label": label, "score": score}


def postprocess(item: dict) -> dict:
    """Simulate CPU post-processing: format results, write to store, etc."""
    delay = random.uniform(0.1, 0.3)
    _log("POSTPROCESS", item["id"], f"formatting result  …  ({delay:.2f}s)")
    time.sleep(delay)
    output = {
        "id":    item["id"],
        "label": item["label"],
        "score": item["score"],
        "url":   item["url"],
    }
    _log("POSTPROCESS", item["id"], f"done  →  {output}")
    return output


# ── Build and run the pipeline ────────────────────────────────────────────────

def main() -> int:
    items = [
        {"id": i, "url": f"https://images.example.com/img_{i:04d}.jpg"}
        for i in range(1, 7)          # 6 images to process
    ]

    print("\n" + "=" * 75)
    print("  3-Stage Pipeline Demo")
    print("    Stage 1  Download     parallelism = 2   (CPU / network I/O)")
    print("    Stage 2  Prediction   parallelism = 1   (GPU / compute)")
    print("    Stage 3  PostProcess  parallelism = 1   (CPU)")
    print(f"\n  Processing {len(items)} items\n")
    print("-" * 75)

    pipeline = (
        StagedPipeline("ml_inference", queue_maxsize=4)
        .add_stage(Stage("download",    download,    parallelism=2))
        .add_stage(Stage("predict",     predict,     parallelism=1))
        .add_stage(Stage("postprocess", postprocess, parallelism=1))
    )

    result = pipeline.run(items)

    print("-" * 75)
    print(f"\n  {result.summary()}")

    if result.errors:
        print("\n  Errors:")
        for e in result.errors:
            print(f"    - {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
