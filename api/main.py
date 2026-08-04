"""
api/main.py
FastAPI application entry point.

Start:  uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
    # Posit Connect proxies content under /content/<guid>/ and strips the
    # prefix before forwarding. Setting root_path lets FastAPI build correct
    # absolute URLs for /docs, OpenAPI, and redirects.
    root_path=os.getenv("FASTAPI_ROOT_PATH", ""),
)

# CORS — allow Streamlit UI (localhost:8501) and local Next.js dev only.
# Do NOT include "*" — it silently overrides all listed origins and allows
# any cross-origin request, including from malicious sites.
_extra_origins = [o.strip() for o in os.getenv("CORS_EXTRA_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501", "http://127.0.0.1:8501",   # Streamlit UI
        "http://localhost:3000", "http://127.0.0.1:3000",   # Next.js dev
        *_extra_origins,
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Optional API-key authentication ──────────────────────────────────────────
# When CONNECT_API_KEY is set on the API side (e.g. Posit Connect deployment),
# every request must carry "Authorization: Key <token>". Local development
# (CONNECT_API_KEY unset) is unaffected.
_API_KEY = os.getenv("CONNECT_API_KEY", "").strip()
_NO_AUTH_PATHS = {"/health", "/docs", "/redoc", "/openapi.json"}

@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    if _API_KEY and request.url.path not in _NO_AUTH_PATHS:
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Key ") or auth[4:].strip() != _API_KEY:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid or missing API key."},
            )
    return await call_next(request)

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
