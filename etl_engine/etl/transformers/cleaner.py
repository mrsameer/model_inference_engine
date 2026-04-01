import pandas as pd
from etl.transformers.base import BaseTransformer
from etl.utils.logger import get_logger

logger = get_logger(__name__)


class DataCleaner(BaseTransformer):
    """Handles common data cleaning tasks."""

    def __init__(
        self,
        drop_duplicates: bool = True,
        strip_strings: bool = True,
        fill_na: dict | None = None,
        drop_na_columns: list[str] | None = None,
        rename_columns: dict | None = None,
    ):
        self.drop_duplicates = drop_duplicates
        self.strip_strings = strip_strings
        self.fill_na = fill_na or {}
        self.drop_na_columns = drop_na_columns or []
        self.rename_columns = rename_columns or {}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        original_len = len(df)
        df = df.copy()

        if self.rename_columns:
            df = df.rename(columns=self.rename_columns)
            logger.debug("Renamed columns: %s", self.rename_columns)

        if self.strip_strings:
            str_cols = df.select_dtypes(include="object").columns
            df[str_cols] = df[str_cols].apply(lambda c: c.str.strip())

        if self.drop_na_columns:
            df = df.dropna(subset=self.drop_na_columns)

        if self.fill_na:
            df = df.fillna(self.fill_na)

        if self.drop_duplicates:
            df = df.drop_duplicates()

        logger.info(
            "Cleaned: %d → %d rows (dropped %d)", original_len, len(df), original_len - len(df)
        )
        return df
