# Author : Eshan Roy <eshanized@proton.me>
# SPDX-License-Identifier: MIT

.PHONY: test lint format check install clean

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

lint:
	ruff check m31r/ tests/

format:
	black m31r/ tests/ tools/

check: lint test
	@echo "All checks passed."

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/
