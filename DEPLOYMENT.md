# Deployment Guide

---

## Local Development

### 1. Clone and install

```bash
git clone https://github.com/ganesan11062001/biomarker_discovery_chatbot.git
cd biomarker_discovery_chatbot
make install
```

### 2. Install kaleido for static PNG export

```bash
pip install kaleido
```

Kaleido is required to export plots as high-resolution PNGs. If it is absent,
the pipeline falls back to HTML-only output — all plots still render in the UI.

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Azure OpenAI (required)
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-key>
AZURE_OPENAI_API_VERSION=2024-08-01-preview

# Model deployments (one GPT-4o deployment can serve all agents)
AZURE_DEPLOYMENT_CHAT=gpt-4o
AZURE_DEPLOYMENT_INGESTION=gpt-4o
AZURE_DEPLOYMENT_BIOMARKER=gpt-4o
AZURE_DEPLOYMENT_ENRICHMENT=gpt-4o
AZURE_DEPLOYMENT_VISUALIZATION=gpt-4o

# Directories
DATA_RAW_DIR=data/raw
DATA_PROCESSED_DIR=data/processed
OUTPUT_DIR=outputs

# Server
API_HOST=0.0.0.0
API_PORT=8000
API_BASE_URL=http://localhost:8000

# LangSmith (optional but recommended)
LANGSMITH_API_KEY=<your-key>
LANGSMITH_PROJECT=biomarker-discovery
LANGSMITH_TRACING=true

# App
APP_ENV=development
```

### 4. Create data directories

```bash
make dirs
```

### 5. Run tests

```bash
make test
```

### 6. Start services

```bash
# Terminal 1
make api       # FastAPI on http://localhost:8000

# Terminal 2
make ui        # Streamlit on http://localhost:8501
```

---

## Production Deployment

### Option A — Single server (systemd)

#### FastAPI backend

Create `/etc/systemd/system/biomarker-api.service`:

```ini
[Unit]
Description=BiomarkerAI FastAPI Backend
After=network.target

[Service]
User=www-data
WorkingDirectory=/opt/biomarker_discovery_chatbot
EnvironmentFile=/opt/biomarker_discovery_chatbot/.env
ExecStart=/opt/biomarker_discovery_chatbot/venv/bin/python -m uvicorn api.main:app \
    --host 0.0.0.0 --port 8000 --workers 2
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

#### Streamlit UI

Create `/etc/systemd/system/biomarker-ui.service`:

```ini
[Unit]
Description=BiomarkerAI Streamlit UI
After=biomarker-api.service

[Service]
User=www-data
WorkingDirectory=/opt/biomarker_discovery_chatbot
EnvironmentFile=/opt/biomarker_discovery_chatbot/.env
ExecStart=/opt/biomarker_discovery_chatbot/venv/bin/python -m streamlit run ui/app.py \
    --server.port 8501 --server.headless true --server.address 0.0.0.0
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable biomarker-api biomarker-ui
sudo systemctl start biomarker-api biomarker-ui
```

---

### Option B — Docker Compose

Create `docker-compose.yml` in the project root:

```yaml
version: "3.9"

services:
  api:
    build: .
    command: python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
    ports:
      - "8000:8000"
    env_file: .env
    volumes:
      - ./data:/app/data
      - ./outputs:/app/outputs
    restart: unless-stopped

  ui:
    build: .
    command: python -m streamlit run ui/app.py --server.port 8501 --server.headless true --server.address 0.0.0.0
    ports:
      - "8501:8501"
    env_file: .env
    environment:
      - API_BASE_URL=http://api:8000
    depends_on:
      - api
    restart: unless-stopped
```

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

# System deps for kaleido (Chromium-based PNG renderer)
RUN apt-get update && apt-get install -y \
    chromium-driver \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip install kaleido

COPY . .
RUN python -c "import pathlib; [pathlib.Path(p).mkdir(parents=True, exist_ok=True) for p in ['data/raw','data/processed','outputs','logs']]"

EXPOSE 8000 8501
```

Deploy:

```bash
docker compose up -d
```

---

### Option C — Azure App Service (API) + Azure Static Web Apps (UI)

**API — Azure App Service (Linux, Python 3.11)**

```bash
# Create App Service plan
az appservice plan create \
  --name biomarker-plan \
  --resource-group <rg> \
  --sku B2 --is-linux

# Create web app
az webapp create \
  --name biomarker-api \
  --resource-group <rg> \
  --plan biomarker-plan \
  --runtime "PYTHON:3.11"

# Set startup command
az webapp config set \
  --name biomarker-api \
  --resource-group <rg> \
  --startup-file "python -m uvicorn api.main:app --host 0.0.0.0 --port 8000"

# Push env vars
az webapp config appsettings set \
  --name biomarker-api \
  --resource-group <rg> \
  --settings @env_settings.json   # JSON file with all .env vars

# Deploy from GitHub
az webapp deployment source config \
  --name biomarker-api \
  --resource-group <rg> \
  --repo-url https://github.com/ganesan11062001/biomarker_discovery_chatbot \
  --branch main --manual-integration
```

**UI — Run Streamlit as a second App Service** using the same approach with startup command:

```bash
python -m streamlit run ui/app.py --server.port 8000 --server.headless true --server.address 0.0.0.0
```

Set `API_BASE_URL` to the API App Service URL in the UI app settings.

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `AZURE_OPENAI_ENDPOINT` | Yes | — | Azure OpenAI resource URL |
| `AZURE_OPENAI_API_KEY` | Yes | — | Azure OpenAI API key |
| `AZURE_OPENAI_API_VERSION` | Yes | `2024-08-01-preview` | API version |
| `AZURE_DEPLOYMENT_CHAT` | Yes | `gpt-4o` | Deployment name for chat/orchestration agent |
| `AZURE_DEPLOYMENT_INGESTION` | No | `gpt-4o` | Deployment for ingestion agent |
| `AZURE_DEPLOYMENT_BIOMARKER` | No | `gpt-4o` | Deployment for biomarker agent |
| `AZURE_DEPLOYMENT_ENRICHMENT` | No | `gpt-4o` | Deployment for enrichment agent |
| `AZURE_DEPLOYMENT_VISUALIZATION` | No | `gpt-4o` | Deployment for visualization agent |
| `API_HOST` | No | `0.0.0.0` | Bind address for uvicorn |
| `API_PORT` | No | `8000` | Port for FastAPI |
| `API_BASE_URL` | No | `http://localhost:8000` | URL the Streamlit UI calls |
| `DATA_RAW_DIR` | No | `data/raw` | Raw upload storage |
| `DATA_PROCESSED_DIR` | No | `data/processed` | Processed data storage |
| `OUTPUT_DIR` | No | `outputs` | Excel, PNG, HTML, and JSON plot output dir |
| `MAX_FILE_SIZE_MB` | No | `200` | Upload size limit |
| `TOP_N_BIOMARKERS` | No | `50` | Biomarkers in Excel top sheet |
| `ADJ_PVAL_CUTOFF` | No | `0.05` | Default adjusted p-value significance threshold |
| `LOG2FC_CUTOFF` | No | `1.0` | Default log₂FC significance threshold |
| `MISSING_VALUE_THRESHOLD` | No | `0.5` | Max missing fraction per protein |
| `LANGSMITH_API_KEY` | No | — | LangSmith API key |
| `LANGSMITH_PROJECT` | No | `biomarker-discovery` | LangSmith project name |
| `LANGSMITH_TRACING` | No | `true` | Enable LangSmith tracing |
| `APP_ENV` | No | `development` | `development` or `production` |
| `LOG_LEVEL` | No | `INFO` | Logging level |

---

## Health Check

```bash
curl http://localhost:8000/health
# {"status":"ok","env":"development","version":"1.0.0"}
```

---

## Useful Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/redoc` | ReDoc |
| `POST` | `/chat/session` | Create a new analysis session |
| `POST` | `/chat/` | Send a message; runs the full LangGraph pipeline |
| `POST` | `/upload/` | Upload a proteomics file (CSV / XLSX / XLS) |
| `GET` | `/results/{session_id}` | Fetch full session state |
| `GET` | `/results/{session_id}/excel` | Download formatted Excel biomarker results |
| `GET` | `/results/{session_id}/file?path=<path>` | Serve any generated output file (PNG, HTML, JSON, CSV) |

---

## Visualization Output Format

Each plot is saved in three formats:

| Format | Extension | Purpose |
|---|---|---|
| PNG | `.png` | High-resolution static image (2× scale via kaleido) |
| HTML | `.html` | Standalone interactive chart (full Plotly.js embedded) |
| JSON | `.json` | Plotly figure JSON loaded by `st.plotly_chart()` for in-app interactivity |

The 16 available plot types are:

| Plot | Standard suite | Triggered by |
|---|---|---|
| `volcano` | Supervised | DEA results |
| `ma_plot` | Supervised | DEA results |
| `waterfall` | Supervised | DEA results |
| `heatmap` | Supervised | Top biomarkers + wide data |
| `pca` | Supervised | Wide data |
| `sample_correlation` | Supervised | Wide data |
| `boxplot` | Supervised | Wide data |
| `violin` | Supervised | Wide data |
| `topn_bar` | Supervised | Top biomarkers |
| `paired_lines` | Paired only | Paired design |
| `anova_multigroup` | ANOVA only | ≥3 groups |
| `cv_distribution` | Unsupervised | All modes |
| `fc_heatmap` | Supervised | Top biomarkers |
| `rescue_bar` | Pooled only | Pooled design |
| `silac_ratio_dist` | SILAC only | SILAC data |
| `pathway_dotplot` | When enrichment available | Pathway results |

---

## Scaling Considerations

- **Sessions are in-memory** — restarting the API loses all sessions. For production, replace `SessionManager` with a Redis backend.
- **Uvicorn workers** — set `--workers` to `(2 × CPU cores) + 1` for concurrency. LangGraph state is not shared between workers, so each worker needs its own session store.
- **File storage** — `data/raw` and `outputs` are local disk. Mount a shared volume (NFS, Azure Files, S3) when running multiple workers or containers.
- **LLM costs** — each chat turn makes 1–3 Azure OpenAI calls. Monitor token usage in LangSmith or Azure portal.
- **PNG export** — kaleido spawns a Chromium process per plot. On memory-constrained servers, consider disabling PNG export (`kaleido` absent = HTML-only fallback) and relying on in-app Plotly rendering instead.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `KeyError: session_id` | Session expired after server restart | Re-upload your file — session data is in-memory only |
| Plots not showing in UI | `OUTPUT_DIR` not mounted or wrong path | Confirm `outputs/` exists and is writable; check `GET /results/{sid}` returns `plot_paths` |
| Plots show as PNG only (no interactive expand) | Normal behaviour — interactive expander renders below chat | Click **🔬 Explore plots interactively** section below the last message |
| PNG export produces blank files | kaleido not installed | `pip install kaleido`; plots fall back to HTML if kaleido absent |
| `gseapy` enrichment fails | Enrichr API unreachable | Check internet connectivity; enrichment returns empty gracefully |
| `wrap_openai` warning on startup | LangSmith not installed or key missing | Install `langsmith` or set `LANGSMITH_TRACING=false` |
| Analysis returns no proteins | All proteins filtered by missing-value threshold | Lower `MISSING_VALUE_THRESHOLD` in `.env` (default 0.5) |
| Upload rejected | Wrong file extension | Accepted: `.csv`, `.xlsx`, `.xls` — rename `.txt`/`.tsv` to `.csv` |
| File exceeds size limit | File > `MAX_FILE_SIZE_MB` | Increase `MAX_FILE_SIZE_MB` in `.env` (default 200 MB) |
