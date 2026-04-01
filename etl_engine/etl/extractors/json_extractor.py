from pathlib import Path
import pandas as pd
from etl.extractors.base import BaseExtractor
from etl.utils.logger import get_logger

logger = get_logger(__name__)


class JSONExtractor(BaseExtractor):
    def __init__(self, file_path: str | Path, record_path: str | None = None):
        self.file_path = Path(file_path)
        self.record_path = record_path

    def extract(self) -> pd.DataFrame:
        if not self.file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {self.file_path}")
        logger.info("Extracting from JSON: %s", self.file_path)
        df = pd.read_json(self.file_path)
        if self.record_path:
            import json
            with open(self.file_path) as f:
                data = json.load(f)
            df = pd.json_normalize(data, record_path=self.record_path)
        logger.info("Extracted %d rows from %s", len(df), self.file_path.name)
        return self.validate(df)
