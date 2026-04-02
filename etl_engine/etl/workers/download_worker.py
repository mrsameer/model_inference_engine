import time
import random
from etl.workers.base_worker import BaseWorker
from etl.utils.logger import get_logger

logger = get_logger(__name__)


class DownloadWorker(BaseWorker):
    """Stage 1 – Download raw data (CPU / network I/O).

    Parallelism = 2: two items can be fetched simultaneously.
    Simulates a network request; in production replace the body of
    process() with real HTTP / S3 / FTP download logic.
    """
    stage = "download"
    parallelism = 2
    poll_interval = 0.3

    def process(self, payload: dict) -> dict:
        url = payload.get("url", f"https://data.example.com/{payload['id']}")
        delay = random.uniform(0.3, 0.7)
        logger.info("[download] fetching %s  (%.2fs)", url, delay)
        time.sleep(delay)
        return {
            **payload,
            "raw_bytes": len(url) * 512,   # pretend we downloaded this many bytes
            "source_url": url,
        }
