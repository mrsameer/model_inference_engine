from abc import ABC, abstractmethod
import pandas as pd


class BaseLoader(ABC):
    """Base class for all loaders."""

    @abstractmethod
    def load(self, df: pd.DataFrame) -> int:
        """Load DataFrame and return the number of rows written."""
        ...
