"""
Demo: pull-based worker pipeline backed by an in-memory task store.

Three independent workers watch the task table for open tasks:

  DownloadWorker    parallelism=2  (CPU / network I/O)
  PredictWorker     parallelism=1  (GPU / compute)
  PostProcessWorker parallelism=1  (CPU / persistence)

Each worker polls continuously; when it finds a pending task in its
stage it claims it atomically, processes it, then enqueues the next
stage task.  Workers are otherwise completely independent — they don't
know about each other.

Run:
    cd /home/user/model_inference_engine/etl_engine
    python -m examples.run_workers

To use a real PostgreSQL instance instead of the in-memory store:
    USE_POSTGRES=true python -m examples.run_workers
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from etl.db import InMemoryTaskStore, PostgresTaskStore
from etl.workers import DownloadWorker, PredictWorker, PostProcessWorker
from etl.workers.pool import WorkerPool
from etl.config import settings

# ── Choose task store ─────────────────────────────────────────────────────────

if os.getenv("USE_POSTGRES", "false").lower() == "true":
    print("Using PostgresTaskStore")
    store = PostgresTaskStore(settings.postgres_dsn)
else:
    print("Using InMemoryTaskStore (no PostgreSQL required)")
    store = InMemoryTaskStore()

# ── Seed initial tasks ────────────────────────────────────────────────────────

ITEMS = [
    {"id": f"img_{i:03d}", "url": f"https://images.example.com/img_{i:03d}.jpg"}
    for i in range(1, 7)
]

print(f"\nSeeding {len(ITEMS)} items into 'download' stage...\n")
store.seed(ITEMS)

# ── Print initial state ───────────────────────────────────────────────────────

print("=" * 70)
print(f"  {'Stage':<14} {'Pending':>8} {'Running':>8} {'Done':>8} {'Failed':>8}")
print("-" * 70)
for stage, counts in store.counts().items():
    print(f"  {stage:<14} {counts.get('pending',0):>8} "
          f"{counts.get('in_progress',0):>8} "
          f"{counts.get('done',0):>8} "
          f"{counts.get('failed',0):>8}")
print("=" * 70)
print()

# ── Launch worker pool ────────────────────────────────────────────────────────

pool = (
    WorkerPool(store)
    .add(DownloadWorker(store))      # parallelism=2
    .add(PredictWorker(store))       # parallelism=1
    .add(PostProcessWorker(store))   # parallelism=1
)

pool.run_until_done(timeout=120, idle_grace=1.5)

# ── Final summary ─────────────────────────────────────────────────────────────

print()
print("=" * 70)
print("  Final task counts")
print("-" * 70)
print(f"  {'Stage':<14} {'Pending':>8} {'Running':>8} {'Done':>8} {'Failed':>8}")
print("-" * 70)
for stage, counts in store.counts().items():
    print(f"  {stage:<14} {counts.get('pending',0):>8} "
          f"{counts.get('in_progress',0):>8} "
          f"{counts.get('done',0):>8} "
          f"{counts.get('failed',0):>8}")
print("=" * 70)
