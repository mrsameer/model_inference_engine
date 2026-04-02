from dataclasses import dataclass, field
from typing import Any
import time
import pandas as pd

from etl.extractors.base import BaseExtractor
from etl.transformers.base import BaseTransformer
from etl.loaders.base import BaseLoader
from etl.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    name: str
    rows_extracted: int = 0
    rows_transformed: int = 0
    rows_loaded: int = 0
    duration_seconds: float = 0.0
    success: bool = False
    error: str | None = None

    def summary(self) -> str:
        status = "SUCCESS" if self.success else f"FAILED ({self.error})"
        return (
            f"Pipeline '{self.name}' [{status}] | "
            f"extracted={self.rows_extracted} "
            f"transformed={self.rows_transformed} "
            f"loaded={self.rows_loaded} "
            f"duration={self.duration_seconds:.2f}s"
        )


class Pipeline:
    """Orchestrates Extract → Transform → Load.

    Usage::

        pipeline = Pipeline("sales_etl")
        pipeline.set_extractor(CSVExtractor("sales.csv"))
        pipeline.add_transformer(DataCleaner(drop_duplicates=True))
        pipeline.add_transformer(DataValidator(rules=[...]))
        pipeline.set_loader(PostgresLoader(dsn, table="sales"))
        result = pipeline.run()
    """

    def __init__(self, name: str):
        self.name = name
        self._extractor: BaseExtractor | None = None
        self._transformers: list[BaseTransformer] = []
        self._loader: BaseLoader | None = None

    def set_extractor(self, extractor: BaseExtractor) -> "Pipeline":
        self._extractor = extractor
        return self

    def add_transformer(self, transformer: BaseTransformer) -> "Pipeline":
        self._transformers.append(transformer)
        return self

    def set_loader(self, loader: BaseLoader) -> "Pipeline":
        self._loader = loader
        return self

    def run(self) -> PipelineResult:
        result = PipelineResult(name=self.name)
        start = time.perf_counter()

        try:
            if self._extractor is None:
                raise RuntimeError("No extractor configured")
            if self._loader is None:
                raise RuntimeError("No loader configured")

            # Extract
            logger.info("[%s] Starting extraction", self.name)
            df = self._extractor.extract()
            result.rows_extracted = len(df)

            # Transform
            for transformer in self._transformers:
                logger.info(
                    "[%s] Running transformer: %s",
                    self.name,
                    transformer.__class__.__name__,
                )
                df = transformer.transform(df)
            result.rows_transformed = len(df)

            # Load
            logger.info("[%s] Loading %d rows", self.name, len(df))
            result.rows_loaded = self._loader.load(df)

            result.success = True
        except Exception as exc:
            result.error = str(exc)
            logger.error("[%s] Pipeline failed: %s", self.name, exc)
        finally:
            result.duration_seconds = time.perf_counter() - start

        logger.info(result.summary())
        return result
