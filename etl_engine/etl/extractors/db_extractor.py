import pandas as pd
from sqlalchemy import create_engine, text
from etl.extractors.base import BaseExtractor
from etl.utils.logger import get_logger

logger = get_logger(__name__)


class DBExtractor(BaseExtractor):
    """Extract data from any SQLAlchemy-supported database."""

    def __init__(self, dsn: str, query: str, params: dict | None = None):
        self.dsn = dsn
        self.query = query
        self.params = params or {}

    def extract(self) -> pd.DataFrame:
        logger.info("Extracting from database with query: %.80s...", self.query)
        engine = create_engine(self.dsn)
        with engine.connect() as conn:
            df = pd.read_sql(text(self.query), conn, params=self.params)
        logger.info("Extracted %d rows from database", len(df))
        engine.dispose()
        return self.validate(df)
