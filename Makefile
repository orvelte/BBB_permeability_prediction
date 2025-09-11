# Makefile for BBB Prediction Project

.PHONY: help install test clean run setup

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	pip install -r requirements.txt

install-dev:  ## Install development dependencies
	pip install -r requirements.txt
	pip install pytest black flake8

setup:  ## Set up the project (install dependencies and create directories)
	pip install -r requirements.txt
	mkdir -p data results/plots results/models

test:  ## Run tests
	python -m pytest tests/ -v

test-coverage:  ## Run tests with coverage
	python -m pytest tests/ --cov=src --cov-report=html

lint:  ## Run linting
	flake8 src/ tests/
	black --check src/ tests/

format:  ## Format code
	black src/ tests/

run:  ## Run the main analysis script
	python src/bbb.py

run-sample:  ## Run with sample data
	cp data/sample_data.csv data/BBB_datasets.csv
	python src/bbb.py

predict:  ## Predict BBB permeability for new molecules
	@echo "Usage: make predict SMILES='CC(=O)NC1=CC=C(C=C1)O'"
	@if [ -z "$(SMILES)" ]; then \
		echo "Please provide SMILES: make predict SMILES='your_smiles_here'"; \
	else \
		python src/predict_bbb.py $(SMILES); \
	fi

clean:  ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf results/plots/*.png
	rm -rf results/models/*.pkl
	rm -rf .pytest_cache/
	rm -rf htmlcov/

clean-data:  ## Clean up data files (be careful!)
	rm -f data/*.csv
	rm -f data/*.xlsx

jupyter:  ## Start Jupyter notebook server
	jupyter notebook notebooks/

docs:  ## Generate documentation
	@echo "Documentation is available in the docs/ directory"

package:  ## Create a distribution package
	python setup.py sdist bdist_wheel

install-package:  ## Install the package in development mode
	pip install -e .

uninstall:  ## Uninstall the package
	pip uninstall BBB_permeability_prediction -y
