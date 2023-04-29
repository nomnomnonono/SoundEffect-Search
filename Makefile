.PHONY: format
format:
	poetry run pysen run format

.PHONY: lint
lint:
	poetry run pysen run lint

.PHONY: setup
setup:
	poetry install
	poetry run pip install librosa

.PHONY: run
run:
	poetry run python app.py
