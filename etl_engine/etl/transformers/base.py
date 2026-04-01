from abc import ABC, abstractmethod
import pandas as pd


class BaseTransformer(ABC):
    """Base class for all transformers."""

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ...
