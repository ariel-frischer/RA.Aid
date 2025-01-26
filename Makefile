.PHONY: test setup-dev setup-hooks check fix help

help:
	@echo "Available commands:"
	@echo "  test         - Run tests with coverage reporting"
	@echo "  check        - Run ruff linter checks"
	@echo "  fix          - Run ruff auto-fixes"
	@echo "  setup-dev    - Install development dependencies"
	@echo "  setup-hooks  - Install git pre-commit hooks"

test:
	# for future consideration append  --cov-fail-under=80 to fail test coverage if below 80%
	python -m pytest --cov=ra_aid --cov-report=term-missing --cov-report=html

check:
	ruff check

fix:
	ruff check --fix

setup-dev:
	pip install -e ".[dev]"

setup-hooks: setup-dev
	pre-commit install
