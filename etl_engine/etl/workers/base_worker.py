"""
BaseWorker – poll-based worker with bounded parallelism.

Each concrete worker:
  1. Continuously polls the task store for pending tasks in its stage.
  2. Claims one task atomically (the store guarantees no two workers
     claim the same task).
  3. Processes the task inside a thread from its own ThreadPoolExecutor
     (bounded to `parallelism` concurrent threads).
  4. On success  → marks the task done and enqueues the next stage.
     On exception → marks the task failed.

Back-pressure is enforced by a Semaphore: the poll loop acquires a
slot before claiming a task, so it never claims more tasks than it has
free worker threads.
"""
from __future__ import annotations

import os
import socket
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from etl.db.task_store import Task, TaskStore
from etl.utils.logger import get_logger

logger = get_logger(__name__)


class BaseWorker(ABC):
    # ── Subclasses must set these ─────────────────────────────────────────────
    stage: str
    parallelism: int = 1
    poll_interval: float = 0.5   # seconds between empty-queue polls

    def __init__(self, task_store: TaskStore) -> None:
        self.task_store = task_store
        self._stop = threading.Event()
        # Unique identity shown in logs and stored on claimed tasks
        self.worker_id = f"{socket.gethostname()}-{os.getpid()}-{self.stage}"

    # ── Subclasses implement this ─────────────────────────────────────────────

    @abstractmethod
    def process(self, payload: dict) -> dict:
        """Do the actual work; return a result dict passed to the next stage."""

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Start the poll loop (blocks until stop() is called)."""
        logger.info(
            "[%s] worker starting  parallelism=%d  poll=%.1fs",
            self.stage, self.parallelism, self.poll_interval,
        )
        sem = threading.Semaphore(self.parallelism)

        with ThreadPoolExecutor(
            max_workers=self.parallelism,
            thread_name_prefix=self.stage,
        ) as pool:
            while not self._stop.is_set():
                # Block until a worker slot is free (up to poll_interval)
                if not sem.acquire(timeout=self.poll_interval):
                    continue   # all slots busy — loop and re-check stop flag

                if self._stop.is_set():
                    sem.release()
                    break

                task = self.task_store.claim(self.stage, self.worker_id)
                if task:
                    pool.submit(self._handle, task, sem)
                else:
                    # Queue is empty — release slot and wait before retrying
                    sem.release()
                    self._stop.wait(self.poll_interval)

        logger.info("[%s] worker stopped", self.stage)

    def stop(self) -> None:
        """Signal the poll loop to exit after finishing in-flight tasks."""
        self._stop.set()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _handle(self, task: Task, sem: threading.Semaphore) -> None:
        thread = threading.current_thread().name
        logger.info("[%s] [%s] picked up task id=%d item=%s", self.stage, thread, task.id, task.item_id)
        try:
            result = self.process(task.payload)
            task.result = result
            self.task_store.complete(task.id, result)
            self.task_store.create_next(task)
            logger.info("[%s] [%s] done     task id=%d item=%s", self.stage, thread, task.id, task.item_id)
        except Exception as exc:
            self.task_store.fail(task.id, str(exc))
            logger.error("[%s] [%s] failed   task id=%d item=%s  error=%s",
                         self.stage, thread, task.id, task.item_id, exc)
        finally:
            sem.release()
