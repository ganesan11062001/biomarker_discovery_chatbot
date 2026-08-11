# BiomarkerAI — Proteomics Multi-Agent Platform

A conversational AI system for biomarker discovery from proteomics data.
Built on **LangGraph**, **Streamlit**, and **Azure OpenAI**.
Observability powered by **LangSmith**.

Deployed to production as a **single Streamlit app** on Posit Connect — see
[DEPLOYMENT.md](DEPLOYMENT.md) for the live app URL and redeploy steps.

---

## Architecture

`ui/app.py` calls the business logic in `core/backend_service.py` **directly,
in-process** — there is no HTTP hop between the UI and the agent workflow.
A FastAPI layer (`api/routes/*.py`) still exists as a thin, optional wrapper
around the same `backend_service` functions, kept for anyone who needs a
standalone REST API, but it is **not** part of the deployed app.

```
┌─────────────────────────────────────────────────────────────┐
│  Streamlit UI  (ui/app.py)                                   │
│  Upload file · Chat · View plots · Download Excel           │
└────────────────────────┬────────────────────────────────────┘
                         │ in-process function calls
┌────────────────────────▼────────────────────────────────────┐
│  core/backend_service.py                                     │
│  create_session · upload_file · run_chat_turn · get_*        │
│  (api/routes/*.py wraps these same functions over HTTP —      │
│   optional, not used by the deployed app)                     │
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
make ui
```

Open **http://localhost:8501** — this single command is enough; the UI calls
`core/backend_service.py` in-process, no separate API server needed.

`make api` still works if you want the optional standalone FastAPI layer
(`api/routes/*.py`) for external HTTP callers, but it is independent of the
Streamlit app and not required to use the chatbot.

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

`BiomarkerAgent` picks a `test_method` (auto-detected or LLM/user-specified) and dispatches to the matching routine in `skills/proteomics_analysis.py`:

| test_method | Scenario | Method |
|---|---|---|
| `welch` (default) | 2 groups, unpaired replicates | Welch t-test + BH FDR + Cohen's d |
| `limma` | 2 groups, unpaired replicates | limma eBayes (moderated t-stat) + BH FDR |
| `paired_t` | Pre/post, matched samples | Paired t-test on per-subject differences + Cohen's d_z |
| `anova` | ≥3 groups | One-way ANOVA + **Tukey HSD post-hoc** (flags which specific group pairs differ) |
| `dose_response` | Ordered dose/concentration groups | Linear trend test across ordered levels (monotonicity, not just omnibus significance) |
| `repeated_measures` | Time-course, same subjects across timepoints | Repeated-measures ANOVA, falls back to a mixed-effects model (`MixedLM`) for unbalanced designs |
| `linear_regression` / `logistic_regression` / `cox_regression` | Continuous / binary / survival clinical outcome | Per-protein regression vs. `clinical_outcome`; logistic adds ROC AUC |

Shared preprocessing for all supervised paths:

| Step | Method |
|------|--------|
| Missing value filter | Proteins with > 50% NaN removed (group-aware — kept if present in ≥ threshold of samples in *either* group) |
| Log₂ transform | Applied when max intensity > 100 |
| Imputation | Half-minimum per protein (computed per row, after saving the pre-imputation detection mask) |
| Multiple testing | Benjamini-Hochberg FDR |
| Log₂FC cap | ±20 (prevents Excel/downstream issues) |
| Effect size | Cohen's d (unpaired) / Cohen's d_z (paired) — sign always matches the log₂FC direction (m2 − m1) |

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

### Phosphoproteomics / PTM

- When `ptm_analysis=True`, enrichment adds kinase-substrate libraries (e.g. KEA2021, PhosphoSitePlus) alongside the standard pathway libraries.

### Unsupervised (no comparison groups)

- Ranks by coefficient of variation (CV%) when ≥ 3 total samples are available (variability is meaningful).
- **Single pooled sample (n=1) fallback**: when there aren't enough replicates to compute CV/MAD/IQR, those fields are `None` and proteins are instead ranked by raw `mean_expression` — so a true n=1 dataset still returns a descriptive ranking report instead of an empty result or a `KeyError`.

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
2. **DecisionSchema** (Pydantic) — validates action names, clamps confidence to [0,1], demotes decisions with `confidence < 0.7` to `"answer"`.
3. **Reason-quality guard** — GPT-4o self-reports ≥0.95 confidence almost every time, so the raw confidence gate rarely fires. Side-effect actions (`run_analysis`, `run_enrichment`, ...) additionally require a reason ≥ 8 characters, or they're demoted to `"answer"` too — catches confident-but-terse (likely misrouted) decisions.
4. **Multi-step routing** — compound imperatives ("compare X vs Y, then run pathway analysis and show the volcano plot") are returned as an ordered `action_sequence` and executed step-by-step (deduplicated per group-pair) instead of only running the first action and dropping the rest.
5. **Viz-override negative guard** — a deterministic override forces `"show volcano/heatmap/PCA"` phrases to `run_visualization`, but a regex exclusion prevents it from misfiring on conceptual questions like *"what does a volcano plot show?"*.
6. **Grounding anchors** — `_answer()` injects the actual `top_biomarkers` and `pathways` lists into the LLM system prompt with explicit citation constraints.

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
├── api/                        Optional standalone REST layer (not used by ui/app.py)
│   ├── main.py                 FastAPI app, CORS, lifespan
│   └── routes/
│       ├── chat.py             POST /chat/  POST /chat/session
│       ├── results.py          GET  /results/{session_id}
│       └── upload.py           POST /upload/
├── config/
│   └── settings.py             Pydantic settings (env-var backed)
├── core/
│   ├── backend_service.py      Business logic called directly by ui/app.py (create_session,
│   │                             upload_file, run_chat_turn, get_analysis_state, ...) — also
│   │                             reused by api/routes/*.py for the optional HTTP layer
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
│   ├── run_visualization.py    ProteomicsPlotSuite (on-demand plots, e.g. waterfall, ANOVA)
│   └── plotly_visuals.py       Auto-generated post-analysis suite (volcano, PCA, heatmap, boxplots)
├── tests/
│   ├── conftest.py
│   ├── test_agents/
│   ├── test_integration/
│   └── test_tracing.py
├── ui/
│   └── app.py                  Streamlit application — the deployed entrypoint
├── .env.example
├── DEPLOYMENT.md
├── manifest.json                Posit Connect deployment manifest
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
