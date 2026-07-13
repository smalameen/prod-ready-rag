import tempfile
from pathlib import Path

import pytest

from src.loaders.factory import get_loader


class TestTextLoader:
    def test_load_txt(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("Hello, world!")
            path = f.name
        loader = get_loader(path)
        docs = loader.load(path)
        assert len(docs) == 1
        assert docs[0].text == "Hello, world!"
        assert docs[0].metadata["source_type"] == "txt"
        Path(path).unlink()

    def test_load_md(self):
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
            f.write("# Title\n\nContent here.")
            path = f.name
        loader = get_loader(path)
        docs = loader.load(path)
        assert len(docs) == 1
        assert "# Title" in docs[0].text
        assert docs[0].metadata["source_type"] == "md"
        Path(path).unlink()


class TestJSONLoader:
    def test_load_json_object(self):
        import json
        data = {"name": "Alice", "age": 30}
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(data, f)
            path = f.name
        loader = get_loader(path)
        docs = loader.load(path)
        assert len(docs) >= 1
        assert "Alice" in docs[0].text
        Path(path).unlink()

    def test_load_json_array(self):
        import json
        data = [{"name": "Alice"}, {"name": "Bob"}]
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(data, f)
            path = f.name
        loader = get_loader(path)
        docs = loader.load(path)
        assert len(docs) == 2
        Path(path).unlink()


class TestCSVLoader:
    def test_load_csv(self):
        import csv
        rows = [("name", "age"), ("Alice", "30"), ("Bob", "25")]
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", newline="", delete=False) as f:
            writer = csv.writer(f)
            writer.writerows(rows)
            path = f.name
        loader = get_loader(path)
        docs = loader.load(path)
        assert len(docs) == 2
        Path(path).unlink()


class TestFactory:
    def test_unsupported_extension(self):
        with pytest.raises(ValueError, match="Unsupported file extension"):
            get_loader("test.xyz")
