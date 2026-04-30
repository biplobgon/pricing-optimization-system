.PHONY: install test lint format api

install:
	pip install -r requirements/base.txt

test:
	pytest tests/

lint:
	flake8 src tests

format:
	black src tests scripts

api:
	uvicorn src.api.main:app --reload
