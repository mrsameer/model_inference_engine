from abc import ABC, abstractmethod
import pandas as pd


class BaseExtractor(ABC):
    """Base class for all extractors."""

    @abstractmethod
    def extract(self) -> pd.DataFrame:
        """Extract data and return as a DataFrame."""
        ...

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Hook for post-extraction validation. Override as needed."""
        if df.empty:
            raise ValueError(f"{self.__class__.__name__}: extracted DataFrame is empty")
        return df
