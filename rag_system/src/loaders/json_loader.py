import json
import os
from pathlib import Path
from typing import Any

from .base import BaseLoader, Document


class JSONLoader(BaseLoader):
    def load(self, file_path: str) -> list[Document]:
        path = Path(file_path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        docs = []
        records = data if isinstance(data, list) else [data]
        for i, record in enumerate(records):
            text = self._flatten(record)
            if text.strip():
                docs.append(
                    Document(
                        text=text,
                        metadata={
                            "source_file": path.name,
                            "source_type": "json",
                            "document_id": f"{path.stem}_{i}",
                            "created_at": self._get_created_at(path),
                            "title": f"{path.stem} - Item {i}",
                            "tags": [],
                        },
                    )
                )
        return docs

    def _flatten(self, obj: Any, prefix: str = "") -> str:
        lines = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, (dict, list)):
                    lines.append(self._flatten(v, key))
                else:
                    lines.append(f"{key}: {v}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                lines.append(self._flatten(item, f"{prefix}[{i}]"))
        else:
            lines.append(str(obj))
        return "\n".join(lines)

    @staticmethod
    def _get_created_at(path: Path) -> str:
        try:
            import datetime
            mtime = os.path.getmtime(path)
            return datetime.datetime.fromtimestamp(mtime).isoformat()
        except Exception:
            return ""
