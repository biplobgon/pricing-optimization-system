.PHONY: install install-dev check-env test lint format api

install:
	pip install -r requirements/base.txt

install-dev:
	pip install -r requirements/dev.txt

check-env:
	python scripts/check_environment.py

test:
	pytest tests/

lint:
	flake8 src tests

format:
	black src tests scripts

api:
	uvicorn src.api.main:app --reload
