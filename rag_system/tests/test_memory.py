from src.memory.conversation import ConversationMemory


class TestConversationMemory:
    def test_add_exchange(self):
        mem = ConversationMemory(window_size=3)
        mem.add_exchange("What is X?", "X is Y")
        assert len(mem.history) == 2
        assert mem.history[0]["content"] == "What is X?"
        assert mem.history[1]["content"] == "X is Y"

    def test_window_trimming(self):
        mem = ConversationMemory(window_size=1)
        mem.add_exchange("Q1", "A1")
        mem.add_exchange("Q2", "A2")
        assert len(mem.history) == 2
        assert mem.history[0]["content"] == "Q2"
        assert mem.history[1]["content"] == "A2"

    def test_format_history(self):
        mem = ConversationMemory(window_size=5)
        mem.add_exchange("Hello", "Hi there")
        formatted = mem.format_history()
        assert "User: Hello" in formatted
        assert "Assistant: Hi there" in formatted

    def test_clear(self):
        mem = ConversationMemory(window_size=5)
        mem.add_exchange("Q", "A")
        mem.clear()
        assert len(mem.history) == 0
