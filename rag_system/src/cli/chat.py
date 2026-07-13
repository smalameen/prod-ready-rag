import logging

from src.embeddings.embedder import get_embedding_model
from src.generation.generator import AnswerGenerator
from src.memory.conversation import ConversationMemory
from src.retrieval.pipeline import RetrievalPipeline
from src.retrieval.vector_store import VectorStore
from src.utils.config import load_config
from src.utils.logging import LatencyTracker, setup_logging


logger = logging.getLogger(__name__)


def main():
    setup_logging()
    config = load_config()

    embedder = get_embedding_model(config.get("embedding", {}).get("model"))
    vector_store = VectorStore()

    retriever = RetrievalPipeline(vector_store, embedder, config)
    generator = AnswerGenerator(config)
    memory = ConversationMemory(window_size=5)

    print("RAG Assistant Started")
    print("Type 'exit' to quit.")
    print()

    while True:
        try:
            user_input = input("\nYou:\n").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if user_input.lower() in ("clear", "reset"):
            memory.clear()
            print("Conversation memory cleared.")
            continue

        tracker = LatencyTracker(logger)

        chunks = retriever.retrieve(user_input, tracker=tracker)
        answer = generator.generate(user_input, chunks, tracker)

        memory.add_exchange(user_input, answer)

        print(f"\nAssistant:\n{answer}")

        logger.info(f"Chat tracker summary: {tracker.summary()}")


if __name__ == "__main__":
    main()
