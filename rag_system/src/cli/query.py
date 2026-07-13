import logging

from src.embeddings.embedder import get_embedding_model
from src.generation.generator import AnswerGenerator
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

    print("Query Mode")
    print("Type 'exit' to quit.")
    print()

    while True:
        try:
            question = input("Question:\n").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break

        tracker = LatencyTracker(logger)

        chunks = retriever.retrieve(question, tracker=tracker)
        if not chunks:
            print("I could not find this information in the knowledge base.")
            print()
            continue

        sources = set(
            c.get("metadata", {}).get("source_file", "unknown") for c in chunks
        )
        print("\nRetrieved Sources:")
        for i, s in enumerate(sorted(sources), 1):
            print(f"{i}. {s}")

        answer = generator.generate(question, chunks, tracker)
        print(f"\nAnswer:\n{answer}\n")

        logger.info(f"Query tracker summary: {tracker.summary()}")


if __name__ == "__main__":
    main()
