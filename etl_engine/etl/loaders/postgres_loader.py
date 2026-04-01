from typing import Literal
import pandas as pd
from sqlalchemy import create_engine, text
from tenacity import retry, stop_after_attempt, wait_exponential
from etl.loaders.base import BaseLoader
from etl.utils.logger import get_logger

logger = get_logger(__name__)

IfExists = Literal["fail", "replace", "append", "upsert"]


class PostgresLoader(BaseLoader):
    """Load a DataFrame into a PostgreSQL table.

    Supports four write modes:
      - append  : INSERT rows (default)
      - replace : DROP + recreate table, then INSERT
      - upsert  : INSERT ... ON CONFLICT DO UPDATE
      - fail    : raise if table exists
    """

    def __init__(
        self,
        dsn: str,
        table: str,
        schema: str = "public",
        if_exists: IfExists = "append",
        conflict_columns: list[str] | None = None,
        batch_size: int = 1000,
        chunksize: int = 500,
    ):
        self.dsn = dsn
        self.table = table
        self.schema = schema
        self.if_exists = if_exists
        self.conflict_columns = conflict_columns or []
        self.batch_size = batch_size
        self.chunksize = chunksize

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def load(self, df: pd.DataFrame) -> int:
        if df.empty:
            logger.warning("Empty DataFrame – nothing to load")
            return 0

        engine = create_engine(self.dsn)
        try:
            if self.if_exists == "upsert":
                rows = self._upsert(df, engine)
            else:
                df.to_sql(
                    self.table,
                    engine,
                    schema=self.schema,
                    if_exists=self.if_exists,
                    index=False,
                    chunksize=self.chunksize,
                    method="multi",
                )
                rows = len(df)
        finally:
            engine.dispose()

        logger.info("Loaded %d rows → %s.%s", rows, self.schema, self.table)
        return rows

    def _upsert(self, df: pd.DataFrame, engine) -> int:
        if not self.conflict_columns:
            raise ValueError("conflict_columns required for upsert mode")

        cols = list(df.columns)
        col_names = ", ".join(f'"{c}"' for c in cols)
        placeholders = ", ".join(f":{c}" for c in cols)
        update_set = ", ".join(
            f'"{c}" = EXCLUDED."{c}"'
            for c in cols
            if c not in self.conflict_columns
        )
        conflict_target = ", ".join(f'"{c}"' for c in self.conflict_columns)

        sql = text(
            f'INSERT INTO "{self.schema}"."{self.table}" ({col_names}) '
            f"VALUES ({placeholders}) "
            f"ON CONFLICT ({conflict_target}) DO UPDATE SET {update_set}"
        )

        records = df.to_dict(orient="records")
        with engine.begin() as conn:
            for i in range(0, len(records), self.batch_size):
                conn.execute(sql, records[i : i + self.batch_size])

        return len(df)
