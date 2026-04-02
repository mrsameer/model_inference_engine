import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from etl.pipeline import Pipeline, PipelineResult
from etl.extractors.base import BaseExtractor
from etl.transformers.base import BaseTransformer
from etl.loaders.base import BaseLoader


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_extractor(df: pd.DataFrame) -> BaseExtractor:
    m = MagicMock(spec=BaseExtractor)
    m.extract.return_value = df
    return m


def make_transformer(fn=None) -> BaseTransformer:
    m = MagicMock(spec=BaseTransformer)
    m.transform.side_effect = fn if fn else lambda df: df
    return m


def make_loader(rows: int = 0) -> BaseLoader:
    m = MagicMock(spec=BaseLoader)
    m.load.return_value = rows
    return m


# ── Pipeline ──────────────────────────────────────────────────────────────────

class TestPipeline:
    def _basic_df(self):
        return pd.DataFrame({"a": [1, 2, 3]})

    def test_successful_run(self):
        df = self._basic_df()
        pipeline = (
            Pipeline("test")
            .set_extractor(make_extractor(df))
            .add_transformer(make_transformer())
            .set_loader(make_loader(rows=3))
        )
        result = pipeline.run()
        assert result.success
        assert result.rows_extracted == 3
        assert result.rows_transformed == 3
        assert result.rows_loaded == 3

    def test_transformer_chain_applied_in_order(self):
        calls = []
        df = self._basic_df()

        def t1(d):
            calls.append(1)
            return d

        def t2(d):
            calls.append(2)
            return d

        pipeline = (
            Pipeline("order_test")
            .set_extractor(make_extractor(df))
            .add_transformer(make_transformer(t1))
            .add_transformer(make_transformer(t2))
            .set_loader(make_loader())
        )
        pipeline.run()
        assert calls == [1, 2]

    def test_missing_extractor_fails(self):
        pipeline = Pipeline("no_extractor").set_loader(make_loader())
        result = pipeline.run()
        assert not result.success
        assert "extractor" in (result.error or "").lower()

    def test_missing_loader_fails(self):
        pipeline = Pipeline("no_loader").set_extractor(make_extractor(self._basic_df()))
        result = pipeline.run()
        assert not result.success
        assert "loader" in (result.error or "").lower()

    def test_extractor_exception_captured(self):
        extractor = MagicMock(spec=BaseExtractor)
        extractor.extract.side_effect = RuntimeError("boom")
        pipeline = Pipeline("err_test").set_extractor(extractor).set_loader(make_loader())
        result = pipeline.run()
        assert not result.success
        assert "boom" in result.error

    def test_duration_recorded(self):
        pipeline = (
            Pipeline("timing")
            .set_extractor(make_extractor(self._basic_df()))
            .set_loader(make_loader())
        )
        result = pipeline.run()
        assert result.duration_seconds >= 0

    def test_summary_contains_name(self):
        pipeline = (
            Pipeline("my_pipeline")
            .set_extractor(make_extractor(self._basic_df()))
            .set_loader(make_loader(3))
        )
        result = pipeline.run()
        assert "my_pipeline" in result.summary()
