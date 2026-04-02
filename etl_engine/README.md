# Python + PostgreSQL ETL Engine

A lightweight, composable ETL (Extract → Transform → Load) framework built with Python and PostgreSQL.

## Features

- **Extractors**: CSV, JSON, Database (any SQLAlchemy DSN)
- **Transformers**: `DataCleaner` (dedup, strip, fill/drop nulls, rename), `DataValidator` (type casting, range checks, allowlist)
- **Loaders**: `PostgresLoader` with `append / replace / upsert / fail` write modes and automatic retry
- **Pipeline**: chainable, logs every step, returns a structured `PipelineResult`

## Project Structure

```
etl_engine/
├── etl/
│   ├── config.py          # Settings via env vars / .env
│   ├── pipeline.py        # Pipeline orchestrator
│   ├── extractors/        # CSV, JSON, DB extractors
│   ├── transformers/      # DataCleaner, DataValidator
│   ├── loaders/           # PostgresLoader
│   └── utils/logger.py
├── tests/                 # pytest test suite (23 tests)
├── examples/
│   ├── sample_data.csv
│   └── run_pipeline.py    # End-to-end demo
├── migrations/init.sql
└── docker-compose.yml
```

## Quick Start

### 1. Start PostgreSQL

```bash
docker compose up -d
```

### 2. Install

```bash
pip install -e ".[dev]"
```

### 3. Run the demo (mock, no DB needed)

```bash
MOCK_LOAD=true python -m examples.run_pipeline
```

### 4. Run against a real PostgreSQL instance

```bash
cp .env.example .env   # edit credentials if needed
python -m examples.run_pipeline
```

### 5. Run tests

```bash
pytest
```

## Usage Example

```python
from etl import Pipeline, CSVExtractor, DataCleaner, DataValidator, ColumnRule, PostgresLoader
from etl.config import settings

result = (
    Pipeline("sales_etl")
    .set_extractor(CSVExtractor("data/sales.csv"))
    .add_transformer(DataCleaner(drop_duplicates=True, fill_na={"revenue": 0}))
    .add_transformer(DataValidator(rules=[
        ColumnRule("id", dtype="int64", required=True),
        ColumnRule("revenue", dtype="float64", min_value=0),
        ColumnRule("region", allowed_values=["US", "EU", "APAC"]),
    ]))
    .set_loader(PostgresLoader(dsn=settings.postgres_dsn, table="sales", if_exists="upsert",
                               conflict_columns=["id"]))
    .run()
)

print(result.summary())
```

## Environment Variables

| Variable          | Default    | Description              |
|-------------------|------------|--------------------------|
| POSTGRES_HOST     | localhost  | PostgreSQL host          |
| POSTGRES_PORT     | 5432       | PostgreSQL port          |
| POSTGRES_DB       | etl_db     | Database name            |
| POSTGRES_USER     | etl_user   | Username                 |
| POSTGRES_PASSWORD | etl_pass   | Password                 |
| ETL_BATCH_SIZE    | 1000       | Upsert batch size        |
| ETL_LOG_LEVEL     | INFO       | Logging level            |
