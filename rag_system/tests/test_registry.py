import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.ingestion.registry import compute_hash, is_ingested, mark_ingested


def test_compute_hash():
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("test content")
        path = f.name
    h = compute_hash(path)
    assert isinstance(h, str)
    assert len(h) == 64
    Path(path).unlink()


@patch("src.ingestion.registry.REGISTRY_PATH")
def test_mark_and_check_ingested(mock_registry_path):
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("test")
        path = f.name

    mock_registry_path.parent.mkdir = lambda parents, exist_ok: None
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as reg_f:
        json.dump({}, reg_f)
        reg_path = reg_f.name

    with patch("src.ingestion.registry.REGISTRY_PATH", Path(reg_path)):
        assert not is_ingested(path)
        mark_ingested(path)
        assert is_ingested(path)

    Path(path).unlink()
    Path(reg_path).unlink()
