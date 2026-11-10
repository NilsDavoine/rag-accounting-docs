# One command to install and index:   make setup
PY := .venv/bin/python
export PYTEST_DISABLE_PLUGIN_AUTOLOAD := 1

.PHONY: setup install db corpus index eval test web

setup: install db corpus index   ## venv + dependencies + PostgreSQL/pgvector + corpus + index

install:
	uv venv -p 3.11 .venv
	uv pip install -p $(PY) -r requirements.txt --index-strategy unsafe-best-match

db:
	docker compose up -d --wait

corpus:
	$(PY) -m data.generate_corpus

index:
	$(PY) -m rag.index

eval:            ## writes eval/results.md
	$(PY) -m eval.run_eval

test:
	$(PY) -m pytest

web:
	$(PY) -m uvicorn app.web:app --port 8000
