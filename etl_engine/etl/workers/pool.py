"""
WorkerPool – launches all stage workers as independent threads.

Each worker runs its own poll loop; they share the same TaskStore but
are otherwise fully decoupled. The pool blocks until every worker has
processed all available tasks and the queues are drained, or until the
optional timeout expires.
"""
from __future__ import annotations

import threading
import time
from typing import Iterable

from etl.workers.base_worker import BaseWorker
from etl.db.task_store import TaskStore, STAGES
from etl.utils.logger import get_logger

logger = get_logger(__name__)


class WorkerPool:
    """
    Usage::

        pool = WorkerPool(task_store)
        pool.add(DownloadWorker(task_store))
        pool.add(PredictWorker(task_store))
        pool.add(PostProcessWorker(task_store))
        pool.run_until_done(timeout=120)
    """

    def __init__(self, task_store: TaskStore) -> None:
        self.task_store = task_store
        self._workers: list[BaseWorker] = []

    def add(self, worker: BaseWorker) -> "WorkerPool":
        self._workers.append(worker)
        return self

    def run_until_done(
        self,
        timeout: float = 300,
        idle_grace: float = 2.0,
    ) -> None:
        """
        Start all workers, then block until all tasks across all stages
        are in a terminal state (done / failed).

        Args:
            timeout:     Hard wall-clock limit in seconds.
            idle_grace:  How long all queues must stay empty before we
                         declare the pipeline finished and stop workers.
        """
        threads = [
            threading.Thread(target=w.run, name=f"worker-{w.stage}", daemon=True)
            for w in self._workers
        ]
        for t in threads:
            t.start()

        logger.info("WorkerPool started (%d workers)", len(self._workers))
        deadline = time.monotonic() + timeout
        idle_since: float | None = None

        try:
            while time.monotonic() < deadline:
                time.sleep(0.5)
                counts = self.task_store.counts()
                pending_total = sum(
                    s_counts.get("pending", 0) + s_counts.get("in_progress", 0)
                    for s_counts in counts.values()
                )

                if pending_total == 0:
                    if idle_since is None:
                        idle_since = time.monotonic()
                    elif time.monotonic() - idle_since >= idle_grace:
                        logger.info("All tasks complete — stopping workers")
                        break
                else:
                    idle_since = None   # reset: still work to do

                self._log_progress(counts)
        finally:
            for w in self._workers:
                w.stop()
            for t in threads:
                t.join(timeout=5)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _log_progress(self, counts: dict) -> None:
        parts = []
        for stage in STAGES:
            c = counts.get(stage, {})
            parts.append(
                f"{stage}: pending={c.get('pending',0)} "
                f"running={c.get('in_progress',0)} "
                f"done={c.get('done',0)} "
                f"failed={c.get('failed',0)}"
            )
        logger.info("progress | %s", "  |  ".join(parts))
