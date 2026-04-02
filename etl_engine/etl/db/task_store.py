"""
Task store — persistence layer for the worker pipeline.

Two implementations share the same interface:

  InMemoryTaskStore   – thread-safe in-process store for tests / demos
  PostgresTaskStore   – production store backed by PostgreSQL;
                        uses SELECT … FOR UPDATE SKIP LOCKED so multiple
                        workers can poll concurrently without ever claiming
                        the same task twice.
"""
from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ── Domain model ──────────────────────────────────────────────────────────────

STAGES = ("download", "predict", "postprocess")
NEXT_STAGE: dict[str, str | None] = {
    "download":    "predict",
    "predict":     "postprocess",
    "postprocess": None,          # terminal
}


@dataclass
class Task:
    id: int
    item_id: str
    stage: str
    status: str          # pending | in_progress | done | failed
    payload: dict
    result: dict | None = None
    error: str | None = None
    worker_id: str | None = None


# ── Abstract interface ────────────────────────────────────────────────────────

class TaskStore(ABC):
    @abstractmethod
    def create_task(self, item_id: str, stage: str, payload: dict) -> int:
        """Insert a new pending task; return its id."""

    @abstractmethod
    def claim(self, stage: str, worker_id: str) -> Task | None:
        """Atomically claim one pending task for *stage*.  Returns None if
        the queue is empty or all tasks are already claimed."""

    @abstractmethod
    def complete(self, task_id: int, result: dict) -> None:
        """Mark a task done and store its result."""

    @abstractmethod
    def fail(self, task_id: int, error: str) -> None:
        """Mark a task failed and store the error message."""

    @abstractmethod
    def counts(self) -> dict[str, dict[str, int]]:
        """Return {stage: {status: count}} for monitoring."""

    # ── Convenience helpers ───────────────────────────────────────────────────

    def seed(self, items: list[dict]) -> None:
        """Create one *download* task per item (bootstraps the pipeline)."""
        for item in items:
            self.create_task(item["id"], "download", item)

    def create_next(self, task: Task) -> None:
        """After a task completes, enqueue the next stage (if any)."""
        next_stage = NEXT_STAGE.get(task.stage)
        if next_stage and task.result is not None:
            self.create_task(task.item_id, next_stage, task.result)


# ── In-memory implementation (tests / demos without PostgreSQL) ───────────────

class InMemoryTaskStore(TaskStore):
    """Thread-safe in-process task store.

    The claim() method holds the lock for its entire duration, giving
    the same mutual-exclusion guarantee as Postgres FOR UPDATE SKIP LOCKED.
    """

    def __init__(self) -> None:
        self._tasks: list[Task] = []
        self._lock = threading.Lock()
        self._next_id = 1

    def create_task(self, item_id: str, stage: str, payload: dict) -> int:
        with self._lock:
            task = Task(
                id=self._next_id,
                item_id=item_id,
                stage=stage,
                status="pending",
                payload=payload,
            )
            self._tasks.append(task)
            self._next_id += 1
            return task.id

    def claim(self, stage: str, worker_id: str) -> Task | None:
        with self._lock:
            for task in self._tasks:
                if task.stage == stage and task.status == "pending":
                    task.status = "in_progress"
                    task.worker_id = worker_id
                    return task
            return None

    def complete(self, task_id: int, result: dict) -> None:
        with self._lock:
            task = self._by_id(task_id)
            task.status = "done"
            task.result = result

    def fail(self, task_id: int, error: str) -> None:
        with self._lock:
            task = self._by_id(task_id)
            task.status = "failed"
            task.error = error

    def counts(self) -> dict[str, dict[str, int]]:
        with self._lock:
            result: dict[str, dict[str, int]] = {s: {} for s in STAGES}
            for task in self._tasks:
                stage_counts = result.setdefault(task.stage, {})
                stage_counts[task.status] = stage_counts.get(task.status, 0) + 1
            return result

    def _by_id(self, task_id: int) -> Task:
        for t in self._tasks:
            if t.id == task_id:
                return t
        raise KeyError(f"Task {task_id} not found")


# ── PostgreSQL implementation ─────────────────────────────────────────────────

class PostgresTaskStore(TaskStore):
    """Production task store backed by PostgreSQL.

    Uses ``SELECT … FOR UPDATE SKIP LOCKED`` so N concurrent workers on
    the same stage never race to claim the same row.

    Requires the schema in migrations/tasks.sql to be applied first.
    """

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    def _connect(self):
        import psycopg2
        import psycopg2.extras
        conn = psycopg2.connect(self.dsn)
        conn.autocommit = False
        return conn

    def create_task(self, item_id: str, stage: str, payload: dict) -> int:
        import psycopg2.extras
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO tasks (item_id, stage, status, payload)
                    VALUES (%s, %s, 'pending', %s)
                    RETURNING id
                    """,
                    (item_id, stage, psycopg2.extras.Json(payload)),
                )
                task_id = cur.fetchone()[0]
            conn.commit()
        return task_id

    def claim(self, stage: str, worker_id: str) -> Task | None:
        import psycopg2.extras
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE tasks
                    SET    status    = 'in_progress',
                           worker_id = %s,
                           started_at = NOW(),
                           updated_at = NOW()
                    WHERE  id = (
                        SELECT id FROM tasks
                        WHERE  stage = %s AND status = 'pending'
                        ORDER  BY created_at ASC
                        FOR UPDATE SKIP LOCKED
                        LIMIT  1
                    )
                    RETURNING id, item_id, stage, status, payload, worker_id
                    """,
                    (worker_id, stage),
                )
                row = cur.fetchone()
            conn.commit()

        if row is None:
            return None
        return Task(
            id=row[0],
            item_id=row[1],
            stage=row[2],
            status=row[3],
            payload=row[4] or {},
            worker_id=row[5],
        )

    def complete(self, task_id: int, result: dict) -> None:
        import psycopg2.extras
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE tasks
                    SET status = 'done', result = %s,
                        finished_at = NOW(), updated_at = NOW()
                    WHERE id = %s
                    """,
                    (psycopg2.extras.Json(result), task_id),
                )
            conn.commit()

    def fail(self, task_id: int, error: str) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE tasks
                    SET status = 'failed', error = %s,
                        finished_at = NOW(), updated_at = NOW()
                    WHERE id = %s
                    """,
                    (error, task_id),
                )
            conn.commit()

    def counts(self) -> dict[str, dict[str, int]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT stage, status, COUNT(*)
                    FROM tasks
                    GROUP BY stage, status
                    """
                )
                rows = cur.fetchall()
        result: dict[str, dict[str, int]] = {s: {} for s in STAGES}
        for stage, status, count in rows:
            result.setdefault(stage, {})[status] = count
        return result
