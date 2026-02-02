"""
Testy dla config_manager - zarządzanie konfiguracją
"""

import pytest
import json
import tempfile
import sys
from pathlib import Path

# Dodaj główny katalog do ścieżki
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestNormalizeDataframe:
    """Testy dla normalize_dataframe_columns()"""

    def test_normalize_views_column(self):
        """Powinno zmapować viewCount -> views"""
        import pandas as pd
        from config_manager import normalize_dataframe_columns

        df = pd.DataFrame({
            "title": ["Test 1", "Test 2"],
            "viewCount": [1000, 2000]
        })

        normalized = normalize_dataframe_columns(df)

        assert "views" in normalized.columns, "Powinno dodać kolumnę 'views'"
        assert normalized["views"].tolist() == [1000, 2000]

    def test_normalize_published_at_column(self):
        """Powinno zmapować publishedAt -> published_at"""
        import pandas as pd
        from config_manager import normalize_dataframe_columns

        df = pd.DataFrame({
            "title": ["Test"],
            "views": [100],
            "publishedAt": ["2024-01-01"]
        })

        normalized = normalize_dataframe_columns(df)

        assert "published_at" in normalized.columns, "Powinno dodać kolumnę 'published_at'"

    def test_preserve_existing_columns(self):
        """Jeśli kolumna już istnieje, nie powinna być nadpisana"""
        import pandas as pd
        from config_manager import normalize_dataframe_columns

        df = pd.DataFrame({
            "title": ["Test"],
            "views": [100],  # Już istnieje
            "viewCount": [200]  # Alias
        })

        normalized = normalize_dataframe_columns(df)

        # views powinno zachować oryginalną wartość (100), nie alias (200)
        assert normalized["views"].iloc[0] == 100

    def test_handle_none_input(self):
        """Powinno bezpiecznie obsłużyć None"""
        from config_manager import normalize_dataframe_columns

        result = normalize_dataframe_columns(None)
        assert result is None


class TestAtomicWrite:
    """Testy dla _atomic_write_json()"""

    def test_atomic_write_creates_file(self):
        """Powinno utworzyć plik JSON"""
        from config_manager import _atomic_write_json

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.json"
            data = {"key": "value", "number": 42}

            result = _atomic_write_json(test_file, data, "test")

            assert result is True, "Powinno zwrócić True przy sukcesie"
            assert test_file.exists(), "Plik powinien istnieć"

            # Sprawdź zawartość
            with open(test_file) as f:
                loaded = json.load(f)
            assert loaded == data

    def test_atomic_write_overwrites_existing(self):
        """Powinno nadpisać istniejący plik"""
        from config_manager import _atomic_write_json

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.json"

            # Pierwszy zapis
            _atomic_write_json(test_file, {"old": "data"}, "test")

            # Drugi zapis (nadpisanie)
            _atomic_write_json(test_file, {"new": "data"}, "test")

            with open(test_file) as f:
                loaded = json.load(f)

            assert loaded == {"new": "data"}


class TestScoringConfig:
    """Testy dla konfiguracji scoringu"""

    def test_default_scoring_config_exists(self):
        """DEFAULT_CONFIG powinien zawierać sekcję scoring"""
        from config_manager import AppConfig

        assert "scoring" in AppConfig.DEFAULT_CONFIG
        scoring = AppConfig.DEFAULT_CONFIG["scoring"]

        assert "threshold_pass" in scoring
        assert "threshold_border" in scoring
        assert "weight_data" in scoring
        assert "weight_llm" in scoring

    def test_scoring_weights_sum_to_one(self):
        """Wagi scoringu powinny sumować się do 1.0"""
        from config_manager import AppConfig

        scoring = AppConfig.DEFAULT_CONFIG["scoring"]
        total = scoring["weight_data"] + scoring["weight_metrics"] + scoring["weight_llm"]

        assert abs(total - 1.0) < 0.01, f"Wagi powinny sumować się do 1.0, suma = {total}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
