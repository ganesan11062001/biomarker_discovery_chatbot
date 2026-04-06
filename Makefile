# ── Biomarker Discovery Platform ─────────────────────────────────────────────
# Usage:
#   make install        install Python dependencies
#   make install-r      install R packages
#   make api            start FastAPI backend
#   make ui             start Streamlit UI
#   make dev            start both (requires two terminals or a process manager)
#   make test           run tests
#   make clean          remove generated outputs

PYTHON  = python
RSCRIPT = Rscript
UVICORN = uvicorn
STREAMLIT = streamlit

.PHONY: install install-r api ui dev test clean dirs

# ── Install ───────────────────────────────────────────────────────────────────

install:
	pip install -r requirements.txt

install-r:
	$(RSCRIPT) r_scripts/install_packages.R

# ── Run ───────────────────────────────────────────────────────────────────────

api:
	$(UVICORN) api.main:app --reload --reload-dir api --reload-dir core --reload-dir agents --reload-dir skills --reload-dir config --host 0.0.0.0 --port 8000

ui:
	API_BASE_URL=http://localhost:8000 $(STREAMLIT) run ui/app.py --server.port 8501

# Tip: open two terminals and run `make api` in one, `make ui` in the other.
dev:
	@echo "Open two terminals:"
	@echo "  Terminal 1:  make api"
	@echo "  Terminal 2:  make ui"

# ── Directories ───────────────────────────────────────────────────────────────

dirs:
	mkdir -p data/raw data/processed outputs logs

# ── Test ─────────────────────────────────────────────────────────────────────

test:
	pytest tests/ -v

# ── Clean ────────────────────────────────────────────────────────────────────

clean:
	rm -rf outputs/* data/processed/* data/raw/* logs/*
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
