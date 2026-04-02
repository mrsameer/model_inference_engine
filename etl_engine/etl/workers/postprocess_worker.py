import time
import random
from etl.workers.base_worker import BaseWorker
from etl.utils.logger import get_logger

logger = get_logger(__name__)


class PostProcessWorker(BaseWorker):
    """Stage 3 – CPU post-processing: format, enrich, and persist results.

    Parallelism = 1: lightweight but serialized (e.g. writing to a
    downstream database or message queue).
    Replace process() with real persistence / notification logic.
    """
    stage = "postprocess"
    parallelism = 1
    poll_interval = 0.3

    def process(self, payload: dict) -> dict:
        delay = random.uniform(0.1, 0.3)
        logger.info("[postprocess] formatting item=%s  label=%s  score=%s  (%.2fs)",
                    payload["id"], payload.get("label"), payload.get("score"), delay)
        time.sleep(delay)
        return {
            "id":     payload["id"],
            "label":  payload.get("label"),
            "score":  payload.get("score"),
            "url":    payload.get("source_url", payload.get("url")),
            "status": "complete",
        }
