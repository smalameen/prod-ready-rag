import logging
import sys
from pathlib import Path
from typing import Optional

from .config import BASE_DIR


LOG_DIR = BASE_DIR / "logs"


def setup_logging(name: str = "rag_system", level: int = logging.INFO) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(LOG_DIR / "rag_system.log", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


class LatencyTracker:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.metrics: dict[str, float] = {}

    def record(self, stage: str, duration_sec: float):
        self.metrics[stage] = duration_sec
        self.logger.info(f"Latency [{stage}]: {duration_sec:.4f}s")

    def summary(self) -> str:
        parts = [f"{k}={v:.4f}s" for k, v in self.metrics.items()]
        return " | ".join(parts)


def log_llm_call(logger: logging.Logger, model: str, prompt_tokens: int, completion_tokens: int, cost: float):
    total_tokens = prompt_tokens + completion_tokens
    logger.info(
        f"LLM Call | Model={model} | "
        f"Prompt={prompt_tokens} | Completion={completion_tokens} | "
        f"Total={total_tokens} | Cost=${cost:.6f}"
    )
