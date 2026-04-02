"""
Demo: CSV → Clean → Validate → PostgreSQL

Run:
    cd etl_engine
    python -m examples.run_pipeline

Requires PostgreSQL (or use mock_load=True to skip the DB step).
"""
import sys
import os

# Allow running from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from etl import (
    Pipeline,
    CSVExtractor,
    DataCleaner,
    DataValidator,
    ColumnRule,
    PostgresLoader,
)
from etl.config import settings

MOCK_LOAD = os.getenv("MOCK_LOAD", "true").lower() == "true"


class PrintLoader:
    """A simple loader that just prints the DataFrame (no DB needed)."""

    def load(self, df):
        print("\n--- Transformed Data ---")
        print(df.to_string(index=False))
        print(f"\nTotal rows: {len(df)}")
        return len(df)


def build_pipeline() -> Pipeline:
    pipeline = Pipeline("customer_etl")

    pipeline.set_extractor(
        CSVExtractor("examples/sample_data.csv")
    )

    pipeline.add_transformer(
        DataCleaner(
            drop_duplicates=True,
            strip_strings=True,
            fill_na={"revenue": 0.0},
            drop_na_columns=["email"],
        )
    )

    pipeline.add_transformer(
        DataValidator(
            rules=[
                ColumnRule("id", dtype="int64", required=True),
                ColumnRule("name", required=True),
                ColumnRule("age", dtype="float64", min_value=0, max_value=120),
                ColumnRule("revenue", dtype="float64", min_value=0),
                ColumnRule("country", allowed_values=["US", "UK", "AU", "CA"]),
            ]
        )
    )

    if MOCK_LOAD:
        pipeline.set_loader(PrintLoader())
    else:
        pipeline.set_loader(
            PostgresLoader(
                dsn=settings.postgres_dsn,
                table="customers",
                if_exists="replace",
            )
        )

    return pipeline


if __name__ == "__main__":
    result = build_pipeline().run()
    print(f"\n{result.summary()}")
    sys.exit(0 if result.success else 1)
