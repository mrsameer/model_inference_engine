from pathlib import Path
import pandas as pd
from etl.extractors.base import BaseExtractor
from etl.utils.logger import get_logger

logger = get_logger(__name__)


class CSVExtractor(BaseExtractor):
    def __init__(self, file_path: str | Path, **read_csv_kwargs):
        self.file_path = Path(file_path)
        self.read_csv_kwargs = read_csv_kwargs

    def extract(self) -> pd.DataFrame:
        if not self.file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")
        logger.info("Extracting from CSV: %s", self.file_path)
        df = pd.read_csv(self.file_path, **self.read_csv_kwargs)
        logger.info("Extracted %d rows from %s", len(df), self.file_path.name)
        return self.validate(df)
