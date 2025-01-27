.PHONY: test pytest setup-dev setup-hooks check fix help

help:
	@echo "Development Commands:"
	@echo "  make test         - Run tests with pytest and generate coverage reports"
	@echo "  make check        - Run ruff linter to check code quality"
	@echo "  make fix          - Auto-fix code style issues (imports, formatting, and linting)"
	@echo "  make fix-basic    - Quick auto-fix of basic linting issues only"
	@echo ""
	@echo "Setup Commands:"
	@echo "  make setup-dev    - Install all development dependencies"
	@echo "  make setup-hooks  - Install git pre-commit hooks (runs setup-dev first)"

test:
	# for future consideration append  --cov-fail-under=80 to fail test coverage if below 80%
	python -m pytest --cov=ra_aid --cov-report=term-missing --cov-report=html

check:
	ruff check

fix:
	ruff check . --select I --fix # First sort imports
	ruff format .
	ruff check --fix

fix-basic:
	ruff check --fix

setup-dev:
	pip install -e ".[dev]"

setup-hooks: setup-dev
	pre-commit install
