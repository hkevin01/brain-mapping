# Makefile for Brain Mapping Toolkit

.PHONY: help install dev-install test lint format clean docs docker

help:
	@echo "Brain Mapping Toolkit - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  install         Install package for production"
	@echo "  dev-install     Install package in development mode"
	@echo "  install-gpu     Install with GPU support"
	@echo "  install-all     Install with all optional dependencies"
	@echo ""
	@echo "Development:"
	@echo "  test           Run tests"
	@echo "  lint           Run linting"
	@echo "  format         Format code with black"
	@echo "  type-check     Run type checking with mypy"
	@echo ""
	@echo "Documentation:"
	@echo "  docs           Build documentation"
	@echo "  docs-serve     Serve documentation locally"
	@echo ""
	@echo "Utilities:"
	@echo "  clean          Clean build artifacts"
	@echo "  docker         Build Docker image"
	@echo "  docker-run     Run in Docker container"

# Installation commands
install:
	pip install .

dev-install:
	pip install -e ".[dev]"

install-gpu:
	pip install -e ".[gpu,dev]"

install-all:
	pip install -e ".[all,dev]"

# Development commands
test:
	pytest tests/ -v --cov=brain_mapping --cov-report=html --cov-report=term

lint:
	flake8 src/brain_mapping tests/
	mypy src/brain_mapping

format:
	black src/brain_mapping tests/
	isort src/brain_mapping tests/

type-check:
	mypy src/brain_mapping

# Documentation
docs:
	cd docs && make html

docs-serve:
	cd docs/_build/html && python -m http.server 8000

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Docker commands
docker:
	docker build -t brain-mapping-toolkit .

docker-run:
	docker run -it --rm \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/output:/app/output \
		brain-mapping-toolkit

# Quick start for new developers
setup-dev: dev-install
	@echo "Development environment setup complete!"
	@echo "Run 'make test' to verify installation"

# CI/CD helpers
ci-test: lint test
	@echo "All CI checks passed!"

release-check:
	python -m build
	twine check dist/*
