import json
import logging
import time
from typing import Any

import requests
from langchain_openai import ChatOpenAI

from src.utils.config import get_openrouter_api_key
from src.utils.logging import LatencyTracker, log_llm_call


logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_SITE_URL = "https://github.com/rag-project"
DEFAULT_SITE_NAME = "Universal Local RAG"

PROMPT_TEMPLATE = """You are an assistant answering questions strictly using the provided context.

If the answer cannot be found in the context, reply:

"I could not find this information in the knowledge base."

Context:
{context}

Question:
{question}

Answer:"""


class OllamaGenerator:
    def __init__(self, config: dict[str, Any]):
        ollama_config = config.get("ollama", {})
        self.model = ollama_config.get("model", "llama3.2")
        self.base_url = ollama_config.get("base_url", "http://localhost:11434")
        self.temperature = ollama_config.get("temperature", 0.2)
        self.max_tokens = ollama_config.get("max_tokens", 1500)
        self.chat_url = f"{self.base_url.rstrip('/')}/api/chat"
        logger.info(f"OllamaGenerator initialized with model: {self.model} at {self.base_url}")

    def generate(
        self,
        question: str,
        context_chunks: list[dict[str, Any]],
        tracker: LatencyTracker | None = None,
    ) -> str:
        context_parts = []
        source_names: set[str] = set()

        for chunk in context_chunks:
            text = chunk.get("text", "")
            source = chunk.get("metadata", {}).get("source_file", "unknown")
            source_names.add(source)
            context_parts.append(f"[Source: {source}]\n{text}")

        context = "\n\n---\n\n".join(context_parts)
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
            "stream": False,
        }

        t0 = time.time()
        resp = requests.post(self.chat_url, json=payload, timeout=120)
        t1 = time.time()

        if tracker:
            tracker.record("llm_generation", t1 - t0)

        resp.raise_for_status()
        data = resp.json()
        answer = data["message"]["content"]

        source_lines = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sorted(source_names)))
        answer += f"\n\nSources:\n{source_lines}"

        return answer


class AnswerGenerator:
    def __init__(self, config: dict[str, Any]):
        or_config = config.get("openrouter", {})
        self.model = or_config.get("model", "openai/gpt-5-mini")
        self.temperature = or_config.get("temperature", 0.2)
        self.max_tokens = or_config.get("max_tokens", 2000)
        self.api_key = get_openrouter_api_key()

        self.llm = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self.api_key,
            base_url=OPENROUTER_BASE_URL,
            default_headers={
                "HTTP-Referer": DEFAULT_SITE_URL,
                "X-Title": DEFAULT_SITE_NAME,
            },
        )
        logger.info(f"AnswerGenerator initialized with model: {self.model}")

    def generate(
        self,
        question: str,
        context_chunks: list[dict[str, Any]],
        tracker: LatencyTracker | None = None,
    ) -> str:
        context_parts = []
        source_names: set[str] = set()

        for chunk in context_chunks:
            text = chunk.get("text", "")
            source = chunk.get("metadata", {}).get("source_file", "unknown")
            source_names.add(source)
            context_parts.append(f"[Source: {source}]\n{text}")

        context = "\n\n---\n\n".join(context_parts)

        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        t0 = time.time()
        response = self.llm.invoke(prompt)
        t1 = time.time()

        if tracker:
            tracker.record("llm_generation", t1 - t0)

        usage = response.usage_metadata if hasattr(response, "usage_metadata") else {}
        prompt_tokens = usage.get("input_tokens", 0) if usage else 0
        completion_tokens = usage.get("output_tokens", 0) if usage else 0

        cost = self._estimate_cost(prompt_tokens, completion_tokens)
        log_llm_call(logger, self.model, prompt_tokens, completion_tokens, cost)

        answer = response.content

        source_lines = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sorted(source_names)))
        answer += f"\n\nSources:\n{source_lines}"

        return answer

    @staticmethod
    def _estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens * 1.0 + completion_tokens * 2.0) / 1_000_000 * 0.15


def create_generator(config: dict[str, Any]):
    provider = config.get("llm_provider", "openrouter")
    if provider == "ollama":
        return OllamaGenerator(config)
    return AnswerGenerator(config)
