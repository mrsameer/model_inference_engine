import json
import pytest
import pandas as pd
from pathlib import Path
from etl.extractors.csv_extractor import CSVExtractor
from etl.extractors.json_extractor import JSONExtractor


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    p = tmp_path / "data.csv"
    p.write_text("id,name,value\n1,Alice,10\n2,Bob,20\n")
    return p


@pytest.fixture
def sample_json(tmp_path: Path) -> Path:
    p = tmp_path / "data.json"
    p.write_text(json.dumps([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]))
    return p


# ── CSVExtractor ─────────────────────────────────────────────────────────────

class TestCSVExtractor:
    def test_extracts_all_rows(self, sample_csv):
        df = CSVExtractor(sample_csv).extract()
        assert len(df) == 2

    def test_column_names(self, sample_csv):
        df = CSVExtractor(sample_csv).extract()
        assert list(df.columns) == ["id", "name", "value"]

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            CSVExtractor(tmp_path / "nonexistent.csv").extract()

    def test_empty_file_raises(self, tmp_path):
        p = tmp_path / "empty.csv"
        p.write_text("id,name\n")  # header only → empty DataFrame
        with pytest.raises(ValueError, match="empty"):
            CSVExtractor(p).extract()


# ── JSONExtractor ─────────────────────────────────────────────────────────────

class TestJSONExtractor:
    def test_extracts_all_rows(self, sample_json):
        df = JSONExtractor(sample_json).extract()
        assert len(df) == 2

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            JSONExtractor(tmp_path / "missing.json").extract()
