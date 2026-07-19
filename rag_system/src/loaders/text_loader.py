import os
from pathlib import Path

from .base import BaseLoader, Document

_ENCODINGS = ["utf-8", "utf-8-sig", "cp1252", "latin-1", "cp1258", "iso-8859-1"]

def _try_decode(data: bytes) -> str:
    for enc in _ENCODINGS:
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return data.decode("utf-8", errors="replace")


class TextLoader(BaseLoader):
    def load(self, file_path: str) -> list[Document]:
        path = Path(file_path)
        raw = path.read_bytes()
        text = _try_decode(raw)
        doc = Document(
            text=text,
            metadata={
                "source_file": path.name,
                "source_type": path.suffix.lstrip("."),
                "document_id": str(path.stem),
                "created_at": self._get_created_at(path),
                "title": path.stem,
                "tags": [],
            },
        )
        return [doc]

    @staticmethod
    def _get_created_at(path: Path) -> str:
        try:
            import datetime
            mtime = os.path.getmtime(path)
            return datetime.datetime.fromtimestamp(mtime).isoformat()
        except Exception:
            return ""
