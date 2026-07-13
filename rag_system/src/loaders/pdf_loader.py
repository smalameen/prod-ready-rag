import os
from pathlib import Path

from .base import BaseLoader, Document


class PDFLoader(BaseLoader):
    def load(self, file_path: str) -> list[Document]:
        path = Path(file_path)
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        text = "\n\n".join(text_parts)
        doc = Document(
            text=text,
            metadata={
                "source_file": path.name,
                "source_type": "pdf",
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
