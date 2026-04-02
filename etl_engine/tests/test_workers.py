import time
import threading
import pytest
from etl.db.task_store import InMemoryTaskStore
from etl.workers.base_worker import BaseWorker
from etl.workers.download_worker import DownloadWorker
from etl.workers.predict_worker import PredictWorker
from etl.workers.postprocess_worker import PostProcessWorker
from etl.workers.pool import WorkerPool


# ── Fast worker stubs (no sleep) ──────────────────────────────────────────────

class FastDownload(DownloadWorker):
    poll_interval = 0.05
    def process(self, payload):
        return {**payload, "raw_bytes": 100, "source_url": payload.get("url", "")}

class FastPredict(PredictWorker):
    poll_interval = 0.05
    def process(self, payload):
        return {**payload, "label": "cat", "score": 0.9}

class FastPostProcess(PostProcessWorker):
    poll_interval = 0.05
    def process(self, payload):
        return {"id": payload["id"], "label": payload.get("label"),
                "score": payload.get("score"), "status": "complete"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def run_worker(worker: BaseWorker, duration: float = 1.0):
    """Run a worker in a background thread and stop it after `duration` seconds."""
    t = threading.Thread(target=worker.run, daemon=True)
    t.start()
    time.sleep(duration)
    worker.stop()
    t.join(timeout=3)


def seeded_store(n: int = 3) -> InMemoryTaskStore:
    store = InMemoryTaskStore()
    items = [{"id": f"item_{i}", "url": f"http://x/{i}"} for i in range(n)]
    store.seed(items)
    return store


# ── BaseWorker behaviour ──────────────────────────────────────────────────────

class TestBaseWorker:
    def test_processes_tasks_and_marks_done(self):
        store = seeded_store(3)
        worker = FastDownload(store)
        run_worker(worker, duration=0.5)
        counts = store.counts()
        assert counts["download"].get("done", 0) == 3

    def test_creates_next_stage_tasks(self):
        store = seeded_store(2)
        worker = FastDownload(store)
        run_worker(worker, duration=0.5)
        counts = store.counts()
        assert counts["predict"].get("pending", 0) == 2

    def test_worker_stops_on_request(self):
        store = InMemoryTaskStore()  # empty — worker just polls
        worker = FastDownload(store)
        t = threading.Thread(target=worker.run, daemon=True)
        t.start()
        time.sleep(0.1)
        worker.stop()
        t.join(timeout=2)
        assert not t.is_alive()

    def test_failed_task_recorded(self):
        class BoomDownload(FastDownload):
            def process(self, payload):
                raise RuntimeError("disk full")

        store = seeded_store(1)
        worker = BoomDownload(store)
        run_worker(worker, duration=0.5)
        counts = store.counts()
        assert counts["download"].get("failed", 0) == 1
        # Failed task should NOT propagate to predict
        assert counts["predict"].get("pending", 0) == 0


# ── Parallelism ───────────────────────────────────────────────────────────────

class TestWorkerParallelism:
    def test_download_runs_two_items_concurrently(self):
        """DownloadWorker.parallelism=2 — verify two items overlap in time."""
        active = [0]
        max_active = [0]
        lock = threading.Lock()

        class TimedDownload(FastDownload):
            def process(self, payload):
                with lock:
                    active[0] += 1
                    if active[0] > max_active[0]:
                        max_active[0] = active[0]
                time.sleep(0.15)
                with lock:
                    active[0] -= 1
                return {**payload, "raw_bytes": 1, "source_url": ""}

        store = seeded_store(6)
        worker = TimedDownload(store)
        run_worker(worker, duration=1.5)
        assert max_active[0] == 2

    def test_predict_processes_one_at_a_time(self):
        max_active = [0]
        active = [0]
        lock = threading.Lock()

        class TimedPredict(FastPredict):
            def process(self, payload):
                with lock:
                    active[0] += 1
                    if active[0] > max_active[0]:
                        max_active[0] = active[0]
                time.sleep(0.1)
                with lock:
                    active[0] -= 1
                return {**payload, "label": "cat", "score": 0.9}

        store = InMemoryTaskStore()
        for i in range(4):
            store.create_task(f"item_{i}", "predict", {"id": f"item_{i}"})

        worker = TimedPredict(store)
        run_worker(worker, duration=1.5)
        assert max_active[0] == 1


# ── End-to-end pool ───────────────────────────────────────────────────────────

class TestWorkerPool:
    def test_full_pipeline_all_items_complete(self):
        store = seeded_store(4)
        pool = (
            WorkerPool(store)
            .add(FastDownload(store))
            .add(FastPredict(store))
            .add(FastPostProcess(store))
        )
        pool.run_until_done(timeout=10, idle_grace=0.5)

        counts = store.counts()
        assert counts["download"]["done"] == 4
        assert counts["predict"]["done"] == 4
        assert counts["postprocess"]["done"] == 4

    def test_no_tasks_pool_exits_cleanly(self):
        store = InMemoryTaskStore()   # nothing seeded
        pool = (
            WorkerPool(store)
            .add(FastDownload(store))
            .add(FastPredict(store))
            .add(FastPostProcess(store))
        )
        pool.run_until_done(timeout=5, idle_grace=0.3)
        # Should reach here without hanging
