.PHONY: install ingest query chat status test clean docker-build docker-run

install:
	pip install -r requirements.txt

ingest:
	python ingest.py

query:
	python query.py

chat:
	python main.py

status:
	python status.py

test:
	python -m pytest tests/ -v

clean:
	rm -rf vectordb/chromadb/
	rm -rf embeddings/model_cache/
	rm -rf data/processed/registry.json
	rm -rf logs/
	rm -rf __pycache__/
	rm -rf src/**/__pycache__/
	rm -rf tests/__pycache__/
	rm -rf .pytest_cache/

docker-build:
	docker compose build

docker-run:
	docker compose run --rm rag-system

docker-ingest:
	docker compose run --rm rag-system python ingest.py

docker-query:
	docker compose run --rm rag-system python query.py

api:
	python api_run.py
