import logging
import os
import tempfile
from pathlib import Path
from typing import BinaryIO

try:
    from supabase import create_client
except ImportError:
    create_client = None


logger = logging.getLogger(__name__)


class SupabaseFileStorage:
    BUCKET_NAME = "rag-files"

    def __init__(self):
        supabase_url = os.environ["SUPABASE_URL"]
        supabase_key = os.environ["SUPABASE_SERVICE_KEY"]
        self._sb = create_client(supabase_url, supabase_key)

    def upload(self, file_content: bytes, filename: str, session_id: str) -> str:
        path = f"{session_id}/{filename}"
        self._sb.storage.from_(self.BUCKET_NAME).upload(
            path=path,
            file=file_content,
            file_options={"content-type": self._content_type(filename)},
        )
        logger.info(f"Uploaded {filename} to Supabase Storage")
        return path

    def download(self, path: str) -> str:
        data = self._sb.storage.from_(self.BUCKET_NAME).download(path)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(path).suffix)
        tmp.write(data)
        tmp.close()
        return tmp.name

    def delete(self, path: str):
        self._sb.storage.from_(self.BUCKET_NAME).remove([path])
        logger.info(f"Deleted {path} from Supabase Storage")

    def list_files(self, session_id: str) -> list[dict]:
        try:
            files = self._sb.storage.from_(self.BUCKET_NAME).list(session_id)
            return [{"name": f["name"], "size": f.get("metadata", {}).get("size", 0)} for f in files]
        except Exception:
            return []

    @staticmethod
    def _content_type(filename: str) -> str:
        ext = Path(filename).suffix.lower()
        return {
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".csv": "text/csv",
            ".json": "application/json",
            ".parquet": "application/octet-stream",
        }.get(ext, "application/octet-stream")
