import time
import random
from etl.workers.base_worker import BaseWorker
from etl.utils.logger import get_logger

logger = get_logger(__name__)


class PredictWorker(BaseWorker):
    """Stage 2 – Run model inference on downloaded data (GPU-bound).

    Parallelism = 1: the GPU processes one item at a time.
    Simulates a slower, compute-heavy step; replace process() with a
    real model call (torch, onnxruntime, TensorRT, etc.).
    """
    stage = "predict"
    parallelism = 1
    poll_interval = 0.3

    def process(self, payload: dict) -> dict:
        delay = random.uniform(0.5, 0.9)
        logger.info("[predict]  running inference on item=%s  (%.2fs)", payload["id"], delay)
        time.sleep(delay)
        score = round(random.uniform(0.50, 0.99), 4)
        label = "cat" if score > 0.75 else "dog"
        return {
            **payload,
            "label": label,
            "score": score,
        }
