# Proteomics Biomarker Discovery Platform

A multi-agent AI system for biomarker discovery from proteomics data.
Built on **LangGraph**, **FastAPI**, **Streamlit**, and **Azure OpenAI**.

---

## Architecture

```
Upload (CSV/Excel)
       │
       ▼
IngestionAgent       — column detection, data normalisation
       │
       ▼
ChatAgent            — intent detection, LangGraph routing
       │
  ┌────┴──────────────────────────────────────┐
  ▼                                           ▼
BiomarkerAgent                         (general chat)
  │
  │  OmicsSkillRegistry
  │  ├── ProteomicsAnalysisSkill  ← implemented
  │  ├── TranscriptomicsSkill     ← planned
  │  └── MetabolomicsSkill        ← planned
  │
  ▼
Excel Report  +  LLM Summary
       │
  ┌────┴─────────────┐
  ▼                  ▼
EnrichmentAgent  VisualizationAgent
(KEGG / GO)      (plots, reports)
```

## Quick Start

### 1. Prerequisites

- Python 3.10+
- Azure OpenAI resource with a GPT-4o deployment

### 2. Install

```bash
make install
```

### 3. Configure

```bash
cp .env.example .env
```

Edit `.env` and fill in:

```env
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-key>
AZURE_DEPLOYMENT_CHAT=gpt-4o
AZURE_DEPLOYMENT_BIOMARKER=gpt-4o
```

### 4. Create directories

```bash
make dirs
```

### 5. Run

Open two terminals:

```bash
# Terminal 1 — API backend
make api

# Terminal 2 — Streamlit UI
make ui
```

Open **http://localhost:8501** in your browser.

---

## Workflow

1. **Upload** a proteomics matrix (CSV or Excel, rows = proteins, columns = samples)
2. **Assign groups** — use the sidebar multiselect to assign sample columns to Group 1 and Group 2
3. **Run Analysis** — click the button or type in the chat
4. **Download** the formatted Excel report with ranked biomarkers, QC metrics, and parameters

---

## Supported Data Types

| Type | Detection |
|------|-----------|
| Olink NPX | Max intensity ≤ 20 |
| MS LFQ | Max intensity > 1000 |
| Generic intensity matrix | Anything else |

---

## Adding a New Omic Type

The platform is designed for multi-omic extensibility.
To add support for (e.g.) transcriptomics:

**1. Create a skill**

```python
# skills/transcriptomics_analysis.py
from skills.base_skill import BaseOmicsSkill, OmicsAnalysisResult

class TranscriptomicsSkill(BaseOmicsSkill):
    @property
    def omic_type(self) -> str:
        return "transcriptomics"

    def execute(self, **kwargs) -> OmicsAnalysisResult:
        # load counts, run DESeq2-equivalent, export Excel …
        return OmicsAnalysisResult(
            omic_type="transcriptomics",
            top_biomarkers=[...],
            n_significant=42,
            excel_path="/outputs/results.xlsx",
            qc_summary={...},
            error=None,
        )
```

**2. Register it in `BiomarkerAgent.__init__`**

```python
# agents/biomarker_agent.py
from skills.transcriptomics_analysis import TranscriptomicsSkill

self._registry.register(TranscriptomicsSkill())
```

**3. Set `omic_type` in state**

The client sets `state["omic_type"] = "transcriptomics"` before calling the API.
No other changes needed — `BiomarkerAgent` routes automatically.

---

## Project Structure

```
├── agents/                  LangGraph agent nodes
│   ├── base_agent.py        Azure OpenAI LLM helpers
│   ├── biomarker_agent.py   Multi-omic dispatcher (OmicsSkillRegistry)
│   ├── chat_agent.py        Intent detection & routing
│   ├── enrichment_agent.py  Pathway enrichment (KEGG / GO)
│   ├── ingestion_agent.py   Data loading & column detection
│   └── visualization_agent.py  Plots & reports
├── api/                     FastAPI application
│   ├── main.py
│   └── routes/
│       ├── chat.py          POST /chat/
│       ├── results.py       GET  /results/{session_id}
│       └── upload.py        POST /upload/
├── config/
│   └── settings.py          Pydantic settings (Azure, analysis defaults)
├── core/
│   ├── langgraph_workflow.py  StateGraph compilation
│   ├── session_manager.py     Thread-safe session store
│   └── state.py               BiomarkerState TypedDict
├── prompts/                 LLM system prompts (one per agent)
├── r_scripts/               R scripts for optional enrichment/viz
│   ├── pathway_enrichment.R
│   └── plot_volcano.R
├── skills/                  Omic analysis skills
│   ├── base_skill.py        BaseOmicsSkill + OmicsAnalysisResult
│   ├── omics_registry.py    OmicsSkillRegistry
│   ├── load_data.py         DataLoadingSkill
│   ├── proteomics_analysis.py  ProteomicsAnalysisSkill
│   ├── run_enrichment.py    PathwaySkill (R backend)
│   └── run_visualization.py    ReportingSkill (R backend)
├── tests/                   pytest test suite
├── ui/
│   ├── app.py               Streamlit application
│   └── components/
├── .env.example             Environment variable template
├── Makefile                 Cross-platform build targets
└── requirements.txt
```

---

## Running Tests

```bash
make test
```

---

## Analysis Methods

| Step | Method |
|------|--------|
| Missing value filter | Proteins with > 50% NaN removed |
| Log2 transform | Applied when max intensity > 100 |
| Imputation | Half-minimum per protein |
| Differential analysis | Welch t-test (unequal variance) |
| Multiple testing | Benjamini-Hochberg FDR |
| Unsupervised ranking | Coefficient of variation (CV%) |

### Significance tiers (supervised mode)

| Tier | adj. p-value | \|log2FC\| |
|------|-------------|-----------|
| Highly Significant | < 0.01 | ≥ 1.0 |
| Significant | < 0.05 | ≥ 1.0 |
| Trend | < 0.10 | any |
