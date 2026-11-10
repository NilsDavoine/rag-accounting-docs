# One command, from a fresh clone to the web page:   make demo
PY := .venv/bin/python
export PYTEST_DISABLE_PLUGIN_AUTOLOAD := 1

.PHONY: demo setup install db corpus index eval eval-claude test test-fast web

demo: setup web   ## install, start the database, build both indexes, serve http://localhost:8000

setup: install db corpus index   ## venv + dependencies + PostgreSQL/pgvector + corpus + index

install:
	uv venv --allow-existing -p 3.11 .venv
	uv pip install -p $(PY) -r requirements.txt --index-strategy unsafe-best-match

db:
	docker compose up -d --wait

corpus:
	$(PY) -m data.generate_corpus
	$(PY) -m data.generate_hard

index:
	$(PY) -m rag.index
	$(PY) -m rag.index --hard

eval:            ## writes eval/results.md: ablation on both sets, then the extractive generation row
	$(PY) -m eval.run_eval
	$(PY) -m eval.run_generation --generator extractive

eval-claude:     ## adds the Claude Haiku 5.5 rows; needs ANTHROPIC_API_KEY, prints the estimated cost first
	$(PY) -m eval.run_generation --generator anthropic

test:            ## everything, including the isolation tests against the database
	$(PY) -m pytest

test-fast:       ## no database, no model
	$(PY) -m pytest -m "not db"

web:
	$(PY) -m uvicorn app.web:app --port 8000
