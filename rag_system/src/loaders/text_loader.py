import os
from pathlib import Path

from .base import BaseLoader, Document


class TextLoader(BaseLoader):
    def load(self, file_path: str) -> list[Document]:
        path = Path(file_path)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
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
