import threading
import pytest
from etl.db.task_store import InMemoryTaskStore, Task


@pytest.fixture
def store():
    return InMemoryTaskStore()


class TestCreateAndClaim:
    def test_create_returns_id(self, store):
        task_id = store.create_task("item1", "download", {"url": "http://x"})
        assert isinstance(task_id, int)

    def test_claim_returns_pending_task(self, store):
        store.create_task("item1", "download", {"url": "http://x"})
        task = store.claim("download", "worker-1")
        assert task is not None
        assert task.stage == "download"
        assert task.status == "in_progress"
        assert task.worker_id == "worker-1"

    def test_claim_empty_returns_none(self, store):
        assert store.claim("download", "w") is None

    def test_claim_wrong_stage_returns_none(self, store):
        store.create_task("item1", "download", {})
        assert store.claim("predict", "w") is None

    def test_claim_in_progress_not_reclaimed(self, store):
        store.create_task("item1", "download", {})
        store.claim("download", "worker-1")
        # Second claim should find nothing
        assert store.claim("download", "worker-2") is None


class TestCompleteAndFail:
    def test_complete_sets_done(self, store):
        store.create_task("item1", "download", {})
        task = store.claim("download", "w")
        store.complete(task.id, {"bytes": 100})
        counts = store.counts()
        assert counts["download"].get("done", 0) == 1

    def test_fail_sets_failed(self, store):
        store.create_task("item1", "download", {})
        task = store.claim("download", "w")
        store.fail(task.id, "network timeout")
        counts = store.counts()
        assert counts["download"].get("failed", 0) == 1


class TestCreateNext:
    def test_create_next_enqueues_predict(self, store):
        store.create_task("item1", "download", {})
        task = store.claim("download", "w")
        task.result = {"raw_bytes": 512}
        store.complete(task.id, task.result)
        store.create_next(task)
        predict_task = store.claim("predict", "w")
        assert predict_task is not None
        assert predict_task.stage == "predict"

    def test_create_next_terminal_stage_does_nothing(self, store):
        store.create_task("item1", "postprocess", {})
        task = store.claim("postprocess", "w")
        task.result = {"status": "complete"}
        store.complete(task.id, task.result)
        store.create_next(task)   # postprocess has no next stage
        counts = store.counts()
        assert counts.get("postprocess", {}).get("pending", 0) == 0


class TestConcurrentClaims:
    def test_two_workers_claim_different_tasks(self, store):
        store.create_task("a", "download", {})
        store.create_task("b", "download", {})

        claimed = []
        errors = []

        def worker(wid):
            task = store.claim("download", wid)
            if task:
                claimed.append(task.id)
            else:
                errors.append(wid)

        t1 = threading.Thread(target=worker, args=("w1",))
        t2 = threading.Thread(target=worker, args=("w2",))
        t1.start(); t2.start()
        t1.join();  t2.join()

        assert len(claimed) == 2
        assert len(set(claimed)) == 2   # each claimed a different task
        assert not errors

    def test_no_double_claim_under_contention(self, store):
        """100 threads race to claim a single task — exactly one should win."""
        store.create_task("item1", "download", {})
        winners = []
        lock = threading.Lock()

        def race():
            task = store.claim("download", "w")
            if task:
                with lock:
                    winners.append(task.id)

        threads = [threading.Thread(target=race) for _ in range(100)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert len(winners) == 1


class TestSeed:
    def test_seed_creates_download_tasks(self, store):
        items = [{"id": f"img_{i}", "url": f"http://x/{i}"} for i in range(5)]
        store.seed(items)
        counts = store.counts()
        assert counts["download"].get("pending", 0) == 5
