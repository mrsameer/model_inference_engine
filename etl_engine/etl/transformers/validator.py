from dataclasses import dataclass, field
import pandas as pd
from etl.transformers.base import BaseTransformer
from etl.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ColumnRule:
    name: str
    dtype: str | None = None          # e.g. "int64", "float64", "object"
    required: bool = False            # disallow nulls
    min_value: float | None = None
    max_value: float | None = None
    allowed_values: list | None = None


class DataValidator(BaseTransformer):
    """Validates a DataFrame against a set of column rules.

    Rows failing validation are dropped and logged.
    """

    def __init__(self, rules: list[ColumnRule], raise_on_empty: bool = True):
        self.rules = rules
        self.raise_on_empty = raise_on_empty

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        invalid_mask = pd.Series(False, index=df.index)

        for rule in self.rules:
            if rule.name not in df.columns:
                logger.warning("Column '%s' not found – skipping rule", rule.name)
                continue

            col = df[rule.name]

            if rule.dtype:
                try:
                    df[rule.name] = col.astype(rule.dtype)
                    col = df[rule.name]
                except (ValueError, TypeError):
                    logger.error("Cannot cast column '%s' to %s", rule.name, rule.dtype)
                    invalid_mask |= True

            if rule.required:
                invalid_mask |= col.isna()

            if rule.min_value is not None:
                invalid_mask |= col < rule.min_value

            if rule.max_value is not None:
                invalid_mask |= col > rule.max_value

            if rule.allowed_values is not None:
                invalid_mask |= ~col.isin(rule.allowed_values)

        invalid_count = invalid_mask.sum()
        if invalid_count:
            logger.warning("Dropping %d invalid rows", invalid_count)
            df = df[~invalid_mask]

        if self.raise_on_empty and df.empty:
            raise ValueError("All rows failed validation")

        logger.info("Validation complete: %d rows passed", len(df))
        return df
