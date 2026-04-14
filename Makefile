# ── Biomarker Discovery Platform ─────────────────────────────────────────────
# Works on macOS, Linux, and Windows (Git Bash / WSL).
#
# Usage:
#   make install    install Python dependencies
#   make api        start FastAPI backend  (port 8000)
#   make ui         start Streamlit UI     (port 8501)
#   make dev        print instructions for running both
#   make test       run tests
#   make dirs       create required directories
#   make clean      remove generated outputs

# ── OS detection ──────────────────────────────────────────────────────────────
# On Windows (native cmd/PowerShell) the OS variable is "Windows_NT".
# Git Bash and WSL report as non-Windows, so standard commands work there.
ifeq ($(OS),Windows_NT)
    PYTHON := python
else
    PYTHON := python3
endif

.PHONY: install api ui dev test clean dirs

# ── Install ───────────────────────────────────────────────────────────────────

install:
	$(PYTHON) -m pip install -r requirements.txt

# ── Run ───────────────────────────────────────────────────────────────────────

api:
	$(PYTHON) -m uvicorn api.main:app --reload \
		--reload-dir api \
		--reload-dir core \
		--reload-dir agents \
		--reload-dir skills \
		--reload-dir config \
		--host 0.0.0.0 --port 8000

ui:
	$(PYTHON) -m streamlit run ui/app.py --server.port 8501

# Open two terminals: `make api` in one, `make ui` in the other.
dev:
	@echo "Start the platform in two terminals:"
	@echo ""
	@echo "  Terminal 1:  make api"
	@echo "  Terminal 2:  make ui"
	@echo ""
	@echo "Then open http://localhost:8501"

# ── Directories (Python-based for cross-platform) ─────────────────────────────

dirs:
	$(PYTHON) -c "import pathlib; [pathlib.Path(p).mkdir(parents=True, exist_ok=True) for p in ['data/raw','data/processed','outputs','logs']]"

# ── Test ─────────────────────────────────────────────────────────────────────

test:
	$(PYTHON) -m pytest tests/ -v

# ── Clean (Python-based for cross-platform) ───────────────────────────────────

clean:
	$(PYTHON) -c "\
import shutil, pathlib; \
[shutil.rmtree(p, ignore_errors=True) or pathlib.Path(p).mkdir(parents=True, exist_ok=True) \
 for p in ['outputs','data/raw','data/processed','logs']]; \
[p.unlink() for p in pathlib.Path('.').rglob('*.pyc')]; \
[shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__') if p.is_dir()] \
"
