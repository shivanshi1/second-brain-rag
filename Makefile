.PHONY: install up down migrate migration api worker test lint fmt typecheck

install:
	pip install -e ".[ml,dev]"
	pre-commit install

up:
	docker compose up -d db redis

down:
	docker compose down

migrate:
	alembic upgrade head

migration:
	alembic revision --autogenerate -m "$(m)"

api:
	uvicorn secondbrain.api.main:app --reload --port 8000

worker:
	celery -A secondbrain.ingestion.celery_app.celery_app worker --loglevel=INFO --concurrency=1

test:
	pytest --cov=secondbrain

lint:
	ruff check . && ruff format --check .

fmt:
	ruff check --fix . && ruff format .

typecheck:
	mypy src
