import os
from pathlib import Path

import pandas as pd

from .base import BaseLoader, Document


class CSVLoader(BaseLoader):
    def load(self, file_path: str) -> list[Document]:
        path = Path(file_path)
        df = pd.read_csv(file_path)
        records = df.to_dict(orient="records")
        docs = []
        for i, record in enumerate(records):
            text = "\n".join(f"{k}: {v}" for k, v in record.items() if pd.notna(v))
            if text.strip():
                docs.append(
                    Document(
                        text=text,
                        metadata={
                            "source_file": path.name,
                            "source_type": "csv",
                            "document_id": f"{path.stem}_row_{i}",
                            "created_at": self._get_created_at(path),
                            "title": f"{path.stem} - Row {i}",
                            "tags": [],
                        },
                    )
                )
        return docs

    @staticmethod
    def _get_created_at(path: Path) -> str:
        try:
            import datetime
            mtime = os.path.getmtime(path)
            return datetime.datetime.fromtimestamp(mtime).isoformat()
        except Exception:
            return ""
