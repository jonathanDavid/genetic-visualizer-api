.PHONY: install dev test lint docker-build docker-run

install:
	pip install -r requirements.txt

# Run the API locally with hot reload on http://localhost:8000
dev:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest

docker-build:
	docker build -t genetic-visualizer-api .

docker-run:
	docker run --rm -p 8000:8000 genetic-visualizer-api
