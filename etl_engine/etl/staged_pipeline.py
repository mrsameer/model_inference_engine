"""
StagedPipeline – multi-stage pipeline with per-stage parallelism.

Each stage runs in its own ThreadPoolExecutor.  Stages communicate via
bounded queues, providing natural back-pressure: a fast upstream stage
cannot flood a slower downstream stage.

             parallelism=2        parallelism=1       parallelism=1
items ──► [ Download stage ] ──► [ Predict stage ] ──► [ PostProc stage ] ──► results
                          queue              queue
"""
from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from etl.utils.logger import get_logger

logger = get_logger(__name__)

_SENTINEL = object()  # poison-pill that signals "no more items"


# ── Stage definition ──────────────────────────────────────────────────────────

@dataclass
class Stage:
    """One step in the pipeline.

    Args:
        name:        Human-readable label shown in logs.
        fn:          Callable that processes a single item.
        parallelism: Number of worker threads for this stage.
    """
    name: str
    fn: Callable[[Any], Any]
    parallelism: int = 1


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class StagedPipelineResult:
    name: str
    items_in: int = 0
    items_out: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        status = "SUCCESS" if self.success else f"FAILED ({len(self.errors)} errors)"
        return (
            f"StagedPipeline '{self.name}' [{status}] | "
            f"in={self.items_in} out={self.items_out} "
            f"duration={self.duration_seconds:.2f}s"
        )


# ── Per-stage driver ──────────────────────────────────────────────────────────

def _drive_stage(
    stage: Stage,
    in_q: queue.Queue,
    out_q: queue.Queue,
    errors: list[str],
) -> None:
    """
    Runs inside its own thread.  Submits items from *in_q* to a
    ThreadPoolExecutor (bounded by stage.parallelism), then forwards
    completed results to *out_q*.

    Two internal sub-threads keep submission and collection decoupled so
    that the executor can stay busy while earlier futures are being
    collected:

        submit-loop  ─►  [executor workers]  ─►  pending-queue
        collect-loop ─►  pending-queue            ─►  out_q
    """
    pending: queue.Queue[Future | object] = queue.Queue()

    # ── submit loop: feed items into the executor ─────────────────────────────
    def _submit_loop() -> None:
        with ThreadPoolExecutor(
            max_workers=stage.parallelism,
            thread_name_prefix=stage.name,
        ) as executor:
            while True:
                item = in_q.get()
                if item is _SENTINEL:
                    in_q.task_done()
                    break
                future: Future = executor.submit(stage.fn, item)
                pending.put(future)
                in_q.task_done()
        # Signal collect-loop that no more futures are coming
        pending.put(_SENTINEL)

    # ── collect loop: gather results and push downstream ─────────────────────
    def _collect_loop() -> None:
        while True:
            item = pending.get()
            if item is _SENTINEL:
                break
            try:
                result = item.result()       # blocks until the future completes
                out_q.put(result)
            except Exception as exc:
                logger.error("[%s] item failed: %s", stage.name, exc)
                errors.append(f"{stage.name}: {exc}")
        # Propagate sentinel downstream
        out_q.put(_SENTINEL)

    sub = threading.Thread(target=_submit_loop,  name=f"{stage.name}-submit",  daemon=True)
    col = threading.Thread(target=_collect_loop, name=f"{stage.name}-collect", daemon=True)
    sub.start()
    col.start()
    sub.join()
    col.join()


# ── StagedPipeline ────────────────────────────────────────────────────────────

class StagedPipeline:
    """
    Chains multiple :class:`Stage` objects so that each stage's output
    feeds the next stage's input via a bounded queue.

    Usage::

        result = (
            StagedPipeline("ml_pipeline")
            .add_stage(Stage("download",        download_fn,        parallelism=2))
            .add_stage(Stage("predict",         predict_fn,         parallelism=1))
            .add_stage(Stage("postprocess",     postprocess_fn,     parallelism=1))
            .run(items)
        )
        print(result.summary())
    """

    def __init__(self, name: str, queue_maxsize: int = 8):
        self.name = name
        self.queue_maxsize = queue_maxsize
        self._stages: list[Stage] = []

    def add_stage(self, stage: Stage) -> "StagedPipeline":
        self._stages.append(stage)
        return self

    def run(self, items: Iterable[Any]) -> StagedPipelineResult:
        if not self._stages:
            raise RuntimeError("No stages configured")

        result = StagedPipelineResult(name=self.name)
        errors: list[str] = []
        start = time.perf_counter()

        # Build one queue per inter-stage boundary
        # queues[0] is fed by the feeder; queues[-1] is drained by the collector
        n = len(self._stages)
        queues = [queue.Queue(maxsize=self.queue_maxsize) for _ in range(n + 1)]

        # ── feeder: push all input items into the first queue ─────────────────
        def _feeder() -> None:
            for item in items:
                queues[0].put(item)
                result.items_in += 1
            queues[0].put(_SENTINEL)

        feeder_thread = threading.Thread(target=_feeder, name="feeder", daemon=True)
        feeder_thread.start()

        # ── launch each stage in its own driver thread ────────────────────────
        stage_threads = []
        for i, stage in enumerate(self._stages):
            logger.info(
                "[%s] Starting stage '%s' (parallelism=%d)",
                self.name, stage.name, stage.parallelism,
            )
            t = threading.Thread(
                target=_drive_stage,
                args=(stage, queues[i], queues[i + 1], errors),
                name=f"driver-{stage.name}",
                daemon=True,
            )
            t.start()
            stage_threads.append(t)

        # ── collector: drain the final output queue ───────────────────────────
        final_results: list[Any] = []

        def _collector() -> None:
            while True:
                item = queues[-1].get()
                if item is _SENTINEL:
                    break
                final_results.append(item)
                result.items_out += 1

        collector_thread = threading.Thread(target=_collector, name="collector", daemon=True)
        collector_thread.start()

        # ── wait for everything to finish ─────────────────────────────────────
        feeder_thread.join()
        for t in stage_threads:
            t.join()
        collector_thread.join()

        result.duration_seconds = time.perf_counter() - start
        result.errors = errors
        logger.info(result.summary())
        return result
