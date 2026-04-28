# BiomarkerAI — Proteomics Multi-Agent Platform

A conversational AI system for biomarker discovery from proteomics data.
Built on **LangGraph**, **FastAPI**, **Streamlit**, and **Azure OpenAI**.
Observability powered by **LangSmith**.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Streamlit UI  (localhost:8501)                             │
│  Upload file · Chat · View plots · Download Excel           │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP
┌────────────────────────▼────────────────────────────────────┐
│  FastAPI Backend  (localhost:8000)                          │
│  POST /upload/   POST /chat/   GET /results/{session_id}    │
└────────────────────────┬────────────────────────────────────┘
                         │ LangGraph invoke()
┌────────────────────────▼────────────────────────────────────┐
│  LearningAgent  — sole LangGraph node (orchestrator)        │
│                                                             │
│  _make_decision()  →  LLM (json_mode=True, confidence gate) │
│                                                             │
│  action: answer           →  _answer()  (grounded chat)     │
│  action: run_analysis     →  BiomarkerAgent                 │
│  action: run_enrichment   →  EnrichmentAgent                │
│  action: run_visualization→  VisualizationAgent             │
│  action: show_code        →  returns stored analysis_code   │
│  action: modify_code      →  LLM rewrites analysis_code     │
│  action: query_database   →  ProteinLookupSkill (UniProt)   │
└─────────────────────────────────────────────────────────────┘
         │              │               │
┌────────▼──────┐ ┌─────▼──────┐ ┌─────▼──────────────────┐
│BiomarkerAgent │ │Enrichment  │ │VisualizationAgent       │
│               │ │Agent       │ │                         │
│PooledFC skill │ │PathwaySkill│ │ProteomicsPlotSuite      │
│Proteomics     │ │(UniProt +  │ │11 plot types            │
│  Analysis     │ │ Enrichr)   │ │(volcano, PCA, heatmap…) │
│CV ranking     │ └────────────┘ └─────────────────────────┘
└───────────────┘
         │
┌────────▼──────────────────────────────────────────────────┐
│  SessionManager  —  in-memory BiomarkerState per session  │
│  LangSmith       —  full trace tree for every LLM call    │
└───────────────────────────────────────────────────────────┘
```

---

## How a Request Flows

1. **Upload** — user drops a CSV or Excel file. `IngestionAgent` detects column types, sample columns, pooled vs replicate design, and builds a `label_map`. All metadata is stored in `BiomarkerState`.

2. **Chat** — every user message goes to the `LearningAgent` orchestrator.
   - `_make_decision()` calls the LLM with `json_mode=True` and returns a `DecisionSchema` JSON with `action`, `confidence`, and optional group assignments.
   - Decisions with `confidence < 0.7` are demoted to `"answer"` to prevent accidental analysis triggers.
   - The orchestrator dispatches to the appropriate specialist or answers directly.

3. **Analysis** — `BiomarkerAgent` selects the right skill:
   - **Pooled design** (n=1 per group) → `PooledFoldChangeSkill` — log₂FC + pairwise contrasts
   - **Supervised** (groups with replicates) → `ProteomicsAnalysisSkill` — Welch t-test + BH FDR
   - **Unsupervised** (no groups) → CV ranking
   - Produces a ranked Excel report and stores `top_biomarkers` in state.

4. **Enrichment** — `EnrichmentAgent` submits the top protein list to UniProt (gene symbol resolution, capped at 500 proteins) then Enrichr via gseapy (KEGG, GO, Reactome, WikiPathways). Results stored in `state["pathways"]`.

5. **Visualization** — `VisualizationAgent` generates up to 11 plot types. Plots anchor to the analysis message in the chat history and stay there across follow-up turns.

6. **LangSmith** — `wrap_openai()` auto-traces every LLM call. `@traceable` spans name each agent phase. All traces nest under a root `LangGraph` trace.

---

## Quick Start

### Prerequisites

- Python 3.9+
- Azure OpenAI resource with a GPT-4o deployment
- (Optional) LangSmith account for observability

### 1. Install

```bash
make install
```

### 2. Configure

```bash
cp .env.example .env
```

Fill in your credentials:

```env
# Required
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-key>
AZURE_OPENAI_API_VERSION=2024-08-01-preview
AZURE_DEPLOYMENT_CHAT=gpt-4o

# Optional — LangSmith observability
LANGSMITH_API_KEY=<your-key>
LANGSMITH_PROJECT=biomarker-discovery
LANGSMITH_TRACING=true
```

### 3. Create directories

```bash
make dirs
```

### 4. Run

```bash
# Terminal 1 — API backend
make api

# Terminal 2 — UI
make ui
```

Open **http://localhost:8501**

---

## Supported Data Types

| Type | Auto-detected when |
|------|-------------------|
| Olink NPX | Max intensity ≤ 20 |
| MS LFQ | Max intensity > 1,000 |
| MS TMT | Intensity range 100–1,000 |
| Generic matrix | Anything else |
| Pooled design | Single sample per group (label_map detected) |

Accepted file formats: `.csv`, `.xlsx`, `.xls`

---

## Analysis Methods

### Supervised (groups with replicates)

| Step | Method |
|------|--------|
| Missing value filter | Proteins with > 50% NaN removed |
| Log₂ transform | Applied when max intensity > 100 |
| Imputation | Half-minimum per protein |
| Differential analysis | Welch t-test (unequal variance) |
| Multiple testing | Benjamini-Hochberg FDR |
| Log₂FC cap | ±20 (prevents Excel/downstream issues) |

Significance tiers:

| Tier | adj. p | \|log₂FC\| |
|------|--------|-----------|
| Highly significant | < 0.01 | ≥ 1.0 |
| Significant | < 0.05 | ≥ 1.0 |
| Trend | < 0.10 | any |

### Pooled (n=1 per group, no replicates)

- Log₂ fold-change with pseudocount (+1)
- All pairwise contrasts auto-generated from the label map
- Generic rescue score: sum of positive fold-changes across all contrasts

### Unsupervised

- Coefficient of variation (CV%) ranking
- No group assignment needed

---

## Pathway Enrichment

Libraries queried per organism:

| Organism | Libraries |
|----------|-----------|
| Human | KEGG 2021, GO BP 2023, Reactome 2022, WikiPathways 2023 |
| Mouse | KEGG 2019 Mouse, GO BP 2023, WikiPathways 2019 Mouse |
| Rat | Mouse libraries (proxy — warning shown to user) |

---

## Plots

| Plot | Available in |
|------|-------------|
| Volcano | Supervised |
| MA plot | Supervised |
| Heatmap | All modes |
| PCA | All modes |
| Boxplot | All modes |
| Sample correlation | All modes |
| CV distribution | Unsupervised |
| FC heatmap | Pooled |
| Top-N bar | Pooled / unsupervised |
| Rescue bar | Pooled |
| Pathway dot plot | After enrichment |

Request specific plots in chat: *"show me a volcano plot"*, *"give me PCA and heatmap"*

---

## Hallucination Guards

1. **json_mode=True** — forces `response_format={"type":"json_object"}` on every decision call.
2. **DecisionSchema** (Pydantic) — validates action names, clamps confidence to [0,1], demotes low-confidence decisions to `"answer"`.
3. **Grounding anchors** — `_answer()` injects the actual `top_biomarkers` and `pathways` lists into the LLM system prompt with explicit citation constraints.

---

## Project Structure

```
├── agents/
│   ├── base_agent.py           Azure OpenAI client + LangSmith wrapping
│   ├── biomarker_agent.py      Multi-omic dispatcher (OmicsSkillRegistry)
│   ├── chat_agent.py           Lightweight Q&A agent
│   ├── enrichment_agent.py     Pathway enrichment + LLM interpretation
│   ├── ingestion_agent.py      File parsing, column detection, QC
│   ├── learning_agent.py       Orchestrator — DecisionSchema, routing, grounding
│   └── visualization_agent.py  Plot generation + LLM summary
├── api/
│   ├── main.py                 FastAPI app, CORS, lifespan
│   └── routes/
│       ├── chat.py             POST /chat/  POST /chat/session
│       ├── results.py          GET  /results/{session_id}
│       └── upload.py           POST /upload/
├── config/
│   └── settings.py             Pydantic settings (env-var backed)
├── core/
│   ├── langgraph_workflow.py   StateGraph (single-node, LearningAgent)
│   ├── session_manager.py      Thread-safe in-memory session store
│   ├── state.py                BiomarkerState TypedDict
│   └── tracing.py              LangSmith configure + metadata helpers
├── prompts/                    System prompts (one .txt per agent)
├── skills/
│   ├── base_skill.py           BaseOmicsSkill + OmicsAnalysisResult
│   ├── load_data.py            DataLoadingSkill
│   ├── omics_registry.py       OmicsSkillRegistry
│   ├── pooled_fold_change.py   PooledFoldChangeSkill
│   ├── protein_lookup.py       ProteinLookupSkill (UniProt REST)
│   ├── proteomics_analysis.py  ProteomicsAnalysisSkill (t-test + BH FDR)
│   ├── run_enrichment.py       PathwaySkill (gseapy / Enrichr)
│   └── run_visualization.py    ProteomicsPlotSuite (11 plot types)
├── tests/
│   ├── conftest.py
│   ├── test_agents/
│   ├── test_integration/
│   └── test_tracing.py
├── ui/
│   └── app.py                  Streamlit application
├── .env.example
├── DEPLOYMENT.md
├── Makefile
└── requirements.txt
```

---

## Adding a New Omic Type

1. Create `skills/transcriptomics_analysis.py`:

```python
from skills.base_skill import BaseOmicsSkill, OmicsAnalysisResult

class TranscriptomicsSkill(BaseOmicsSkill):
    @property
    def omic_type(self) -> str:
        return "transcriptomics"

    def execute(self, **kwargs) -> OmicsAnalysisResult:
        return OmicsAnalysisResult(
            omic_type="transcriptomics",
            top_biomarkers=[...],
            n_significant=42,
            excel_path="outputs/results.xlsx",
            qc_summary={},
            error=None,
        )
```

2. Register it in `agents/biomarker_agent.py`:

```python
from skills.transcriptomics_analysis import TranscriptomicsSkill
self._registry.register(TranscriptomicsSkill())
```

No other changes needed — `BiomarkerAgent` routes by `omic_type` automatically.

---

## Running Tests

```bash
make test
# or
python3 -m pytest tests/ -v --tb=short
```

105 tests covering: tracing, base agent, learning agent (decision schema, routing, grounding), and full end-to-end integration.

---

## LangSmith Observability

With `LANGSMITH_TRACING=true`, every run produces a trace tree:

```
LangGraph  (root)
└── learning_agent
    ├── orchestrator.decision   — action, confidence, group assignments
    └── orchestrator.answer / BiomarkerAgent / EnrichmentAgent / VisualizationAgent
```

View at https://smith.langchain.com → project `biomarker-discovery`.
