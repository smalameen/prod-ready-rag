from typing import Any


class ConversationMemory:
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.history: list[dict[str, str]] = []

    def add_exchange(self, question: str, answer: str):
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})
        if len(self.history) > self.window_size * 2:
            self.history = self.history[-(self.window_size * 2):]

    def get_history(self) -> list[dict[str, str]]:
        return self.history.copy()

    def format_history(self) -> str:
        lines = []
        for entry in self.history:
            role = entry["role"].capitalize()
            lines.append(f"{role}: {entry['content']}")
        return "\n".join(lines)

    def clear(self):
        self.history.clear()
