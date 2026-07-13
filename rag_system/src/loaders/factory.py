from pathlib import Path

from .base import BaseLoader
from .csv_loader import CSVLoader
from .docx_loader import DocxLoader
from .json_loader import JSONLoader
from .parquet_loader import ParquetLoader
from .pdf_loader import PDFLoader
from .text_loader import TextLoader


_EXTENSION_MAP: dict[str, type[BaseLoader]] = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".pdf": PDFLoader,
    ".docx": DocxLoader,
    ".csv": CSVLoader,
    ".json": JSONLoader,
    ".parquet": ParquetLoader,
}


def get_loader(file_path: str) -> BaseLoader:
    ext = Path(file_path).suffix.lower()
    loader_cls = _EXTENSION_MAP.get(ext)
    if loader_cls is None:
        raise ValueError(f"Unsupported file extension: {ext}")
    return loader_cls()
