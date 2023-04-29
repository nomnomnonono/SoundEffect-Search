.PHONY: format
format:
	poetry run pysen run format

.PHONY: lint
lint:
	poetry run pysen run lint

.PHONY: setup
setup:
	poetry install

.PHONY: run
run:
	poetry run python app.py
