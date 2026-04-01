import time
import threading
import pytest
from etl.staged_pipeline import Stage, StagedPipeline, StagedPipelineResult


# ── Helpers ───────────────────────────────────────────────────────────────────

def double(x):
    return x * 2

def to_str(x):
    return str(x)

def boom(x):
    if x == 3:
        raise ValueError("item 3 exploded")
    return x


# ── Basic correctness ─────────────────────────────────────────────────────────

class TestStagedPipelineCorrectness:
    def test_single_stage_passes_all_items(self):
        result = (
            StagedPipeline("t")
            .add_stage(Stage("double", double))
            .run([1, 2, 3])
        )
        assert result.success
        assert result.items_in == 3
        assert result.items_out == 3

    def test_multi_stage_transforms_data(self):
        # double → str
        thread_results = []

        def collect_double(x):
            v = double(x)
            thread_results.append(v)
            return v

        result = (
            StagedPipeline("chain")
            .add_stage(Stage("double", collect_double))
            .add_stage(Stage("to_str", to_str))
            .run([1, 2, 3])
        )
        assert result.success
        assert result.items_out == 3
        assert set(thread_results) == {2, 4, 6}

    def test_empty_input(self):
        result = (
            StagedPipeline("empty")
            .add_stage(Stage("double", double))
            .run([])
        )
        assert result.items_in == 0
        assert result.items_out == 0

    def test_no_stages_raises(self):
        with pytest.raises(RuntimeError, match="No stages"):
            StagedPipeline("no_stages").run([1, 2])


# ── Parallelism ───────────────────────────────────────────────────────────────

class TestParallelism:
    def test_parallelism_2_uses_two_threads(self):
        """With parallelism=2, two items should run concurrently."""
        active_threads: set[str] = set()
        lock = threading.Lock()
        max_concurrent = [0]
        current_concurrent = [0]

        def slow_fn(x):
            with lock:
                current_concurrent[0] += 1
                if current_concurrent[0] > max_concurrent[0]:
                    max_concurrent[0] = current_concurrent[0]
                active_threads.add(threading.current_thread().name)
            time.sleep(0.1)
            with lock:
                current_concurrent[0] -= 1
            return x

        result = (
            StagedPipeline("par2")
            .add_stage(Stage("worker", slow_fn, parallelism=2))
            .run(list(range(6)))
        )
        assert result.success
        assert max_concurrent[0] == 2

    def test_parallelism_1_is_sequential_within_stage(self):
        """With parallelism=1, only one item should run at a time."""
        max_concurrent = [0]
        current_concurrent = [0]
        lock = threading.Lock()

        def slow_fn(x):
            with lock:
                current_concurrent[0] += 1
                if current_concurrent[0] > max_concurrent[0]:
                    max_concurrent[0] = current_concurrent[0]
            time.sleep(0.05)
            with lock:
                current_concurrent[0] -= 1
            return x

        result = (
            StagedPipeline("par1")
            .add_stage(Stage("worker", slow_fn, parallelism=1))
            .run(list(range(4)))
        )
        assert result.success
        assert max_concurrent[0] == 1


# ── Error handling ────────────────────────────────────────────────────────────

class TestErrorHandling:
    def test_stage_error_recorded_other_items_continue(self):
        result = (
            StagedPipeline("err")
            .add_stage(Stage("boom", boom))
            .run([1, 2, 3, 4])
        )
        assert not result.success
        assert len(result.errors) == 1
        assert "item 3 exploded" in result.errors[0]
        # Items 1,2,4 should still be processed
        assert result.items_out == 3

    def test_summary_shows_success(self):
        result = (
            StagedPipeline("ok")
            .add_stage(Stage("double", double))
            .run([1])
        )
        assert "SUCCESS" in result.summary()
        assert "ok" in result.summary()

    def test_summary_shows_failure(self):
        result = (
            StagedPipeline("fail")
            .add_stage(Stage("boom", boom))
            .run([3])
        )
        assert "FAILED" in result.summary()


# ── Throughput ────────────────────────────────────────────────────────────────

class TestThroughput:
    def test_3_stage_pipeline_finishes(self):
        result = (
            StagedPipeline("e2e")
            .add_stage(Stage("download",    lambda x: x,        parallelism=2))
            .add_stage(Stage("predict",     lambda x: x * 10,   parallelism=1))
            .add_stage(Stage("postprocess", lambda x: str(x),   parallelism=1))
            .run(list(range(10)))
        )
        assert result.success
        assert result.items_in == 10
        assert result.items_out == 10
