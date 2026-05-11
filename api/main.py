"""
api/main.py
FastAPI application entry point.

Start:  uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes import chat, sessions, upload, results
from config.settings import get_settings
from core.langgraph_workflow import get_workflow

settings = get_settings()


# ── Logging setup ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    settings.ensure_dirs()
    logger.info("Directories ready: raw=%s  processed=%s  outputs=%s",
                settings.data_raw_dir, settings.data_processed_dir, settings.output_dir)
    # Pre-warm the LangGraph workflow (compiles the graph and instantiates agents)
    get_workflow()
    logger.info("LangGraph workflow compiled and ready.")
    # Rehydrate any session checkpoints from disk so the user can resume after
    # a server restart (GenoMAS-style checkpoint recovery).
    from core.session_manager import SessionManager
    n_loaded = SessionManager.load_from_disk()
    if n_loaded:
        logger.info("Rehydrated %d session(s) from disk checkpoint.", n_loaded)
    yield
    # Shutdown (nothing to clean up for now)
    logger.info("Shutting down Biomarker Discovery Platform.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Proteomics Biomarker Discovery Platform",
    description=(
        "Multi-agent AI system for proteomics biomarker discovery. "
        "Supports Olink NPX, label-free MS, TMT, and generic intensity matrices."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow Streamlit UI (localhost:8501) and any local origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501", "http://127.0.0.1:8501",   # legacy Streamlit
        "http://localhost:3000", "http://127.0.0.1:3000",   # Next.js dev
        "*",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated output files (plots, CSVs) as static assets
app.mount("/static", StaticFiles(directory=settings.output_dir, check_dir=False), name="static")

# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(chat.router,     prefix="/chat",     tags=["Chat"])
app.include_router(upload.router,   prefix="/upload",   tags=["Upload"])
app.include_router(results.router,  prefix="/results",  tags=["Results"])
app.include_router(sessions.router, prefix="/sessions", tags=["Sessions"])


# ── Utility endpoints ─────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "ok",
        "env": settings.app_env,
        "version": "1.0.0",
    }
