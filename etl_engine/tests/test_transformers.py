import pytest
import pandas as pd
from etl.transformers.cleaner import DataCleaner
from etl.transformers.validator import DataValidator, ColumnRule


# ── DataCleaner ───────────────────────────────────────────────────────────────

class TestDataCleaner:
    def _df(self):
        return pd.DataFrame(
            {
                "name": ["  Alice ", "Bob", "Alice", "  Bob  "],
                "score": [10, 20, 10, 20],
                "tag": [None, "x", None, "x"],
            }
        )

    def test_strips_strings(self):
        df = DataCleaner(drop_duplicates=False).transform(self._df())
        assert df["name"].tolist() == ["Alice", "Bob", "Alice", "Bob"]

    def test_drop_duplicates(self):
        df = DataCleaner(drop_duplicates=True).transform(self._df())
        assert len(df) == 2

    def test_fill_na(self):
        df = DataCleaner(fill_na={"tag": "default"}).transform(self._df())
        assert (df["tag"] == "default").any()

    def test_drop_na_columns(self):
        df = DataCleaner(drop_na_columns=["tag"], drop_duplicates=False).transform(self._df())
        assert df["tag"].notna().all()

    def test_rename_columns(self):
        df = DataCleaner(rename_columns={"name": "full_name"}, drop_duplicates=False).transform(
            self._df()
        )
        assert "full_name" in df.columns
        assert "name" not in df.columns


# ── DataValidator ─────────────────────────────────────────────────────────────

class TestDataValidator:
    def _df(self):
        return pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "age": [25, -5, 30, 200],
                "country": ["US", "UK", "XX", "AU"],
            }
        )

    def test_filters_out_of_range_values(self):
        rules = [ColumnRule("age", min_value=0, max_value=120)]
        df = DataValidator(rules).transform(self._df())
        assert (df["age"] >= 0).all()
        assert (df["age"] <= 120).all()

    def test_filters_disallowed_values(self):
        rules = [ColumnRule("country", allowed_values=["US", "UK", "AU"])]
        df = DataValidator(rules).transform(self._df())
        assert "XX" not in df["country"].values

    def test_required_drops_nulls(self):
        df = pd.DataFrame({"id": [1, 2, None, 4]})
        rules = [ColumnRule("id", required=True)]
        result = DataValidator(rules).transform(df)
        assert result["id"].notna().all()

    def test_raises_when_all_rows_invalid(self):
        df = pd.DataFrame({"age": [-1, -2]})
        rules = [ColumnRule("age", min_value=0)]
        with pytest.raises(ValueError, match="All rows failed"):
            DataValidator(rules, raise_on_empty=True).transform(df)

    def test_missing_column_skipped_gracefully(self):
        df = pd.DataFrame({"name": ["Alice"]})
        rules = [ColumnRule("missing_col", required=True)]
        # Should not raise; just warns
        result = DataValidator(rules).transform(df)
        assert len(result) == 1
