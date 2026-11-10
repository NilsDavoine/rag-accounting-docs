"""Pinned models and settings. Revisions are Hugging Face commit hashes, so a model
update upstream cannot silently change the benchmark."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "data" / "corpus"
LABELS_FILE = ROOT / "data" / "nature_labels.jsonl"
QUESTIONS_FILE = ROOT / "eval" / "questions.jsonl"
ARTIFACTS = ROOT / "artifacts"            # trained classifier + tuned thresholds (generated)

DATABASE_URL = os.environ.get("RAG_DATABASE_URL", "postgresql://rag:rag@localhost:5433/rag")

# Multilingual (FR/EN/DE among ~100 languages), 118M parameters, 384 dimensions.
EMBED_MODEL = "intfloat/multilingual-e5-small"
EMBED_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
EMBED_DIM = 384

# Multilingual cross-encoder trained on mMARCO, 118M parameters.
RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
RERANK_REVISION = "1427fd652930e4ba29e8149678df786c240d8825"

CANDIDATES = 50     # per retriever, before fusion
RRF_K = 60
FINAL_K = 8         # passages kept after reranking

# Generation backends (all optional): extractive (default, no model), ollama, anthropic.
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b-instruct")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-5-5")
