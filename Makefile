.PHONY: install dev backtest plan test lint
install:
	pip install -e .
dev:
	pip install -e ".[dev]"
backtest:
	atlasforecast backtest
plan:
	atlasforecast plan
test:
	pytest -q --cov=atlasforecast --cov-report=term-missing
lint:
	ruff check src tests
