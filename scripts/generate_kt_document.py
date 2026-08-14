"""
scripts/generate_kt_document.py
Generates docs/BiomarkerAI_KT_Document.docx — a Knowledge Transfer document
covering architecture, code walkthrough, deployment, and known-issue history.

Run:  .venv\\Scripts\\python.exe scripts\\generate_kt_document.py
"""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "docs" / "BiomarkerAI_KT_Document.docx"
DIAGRAM_PATH = ROOT / "scripts" / "architecture-diagram.png"

NAVY = RGBColor(0x1F, 0x3A, 0x5F)
GREY = RGBColor(0x55, 0x55, 0x55)
CODE_BG = "F2F2F2"


def shade_cell(cell, hex_color: str) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


def set_cell_text(cell, text: str, bold: bool = False, mono: bool = False, size: int = 10) -> None:
    cell.text = ""
    p = cell.paragraphs[0]
    run = p.add_run(text)
    run.bold = bold
    run.font.size = Pt(size)
    if mono:
        run.font.name = "Consolas"


class KTDoc:
    def __init__(self) -> None:
        self.doc = Document()
        self._style_document()

    def _style_document(self) -> None:
        normal = self.doc.styles["Normal"]
        normal.font.name = "Calibri"
        normal.font.size = Pt(10.5)
        for i, size in ((1, 20), (2, 15), (3, 12.5)):
            h = self.doc.styles[f"Heading {i}"]
            h.font.color.rgb = NAVY
            h.font.size = Pt(size)

    # ── content helpers ───────────────────────────────────────────────────
    def title_page(self, title: str, subtitle: str, meta: list[tuple[str, str]]) -> None:
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(title)
        run.font.size = Pt(30)
        run.font.bold = True
        run.font.color.rgb = NAVY

        p2 = self.doc.add_paragraph()
        p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run2 = p2.add_run(subtitle)
        run2.font.size = Pt(14)
        run2.font.color.rgb = GREY

        self.doc.add_paragraph()
        table = self.doc.add_table(rows=0, cols=2)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        for k, v in meta:
            row = table.add_row().cells
            set_cell_text(row[0], k, bold=True)
            set_cell_text(row[1], v)
        self.doc.add_page_break()

    def h1(self, text: str) -> None:
        self.doc.add_heading(text, level=1)

    def h2(self, text: str) -> None:
        self.doc.add_heading(text, level=2)

    def h3(self, text: str) -> None:
        self.doc.add_heading(text, level=3)

    def p(self, text: str = "") -> None:
        self.doc.add_paragraph(text)

    def bullets(self, items: list[str]) -> None:
        for item in items:
            self.doc.add_paragraph(item, style="List Bullet")

    def numbered(self, items: list[str]) -> None:
        for item in items:
            self.doc.add_paragraph(item, style="List Number")

    def code(self, text: str) -> None:
        table = self.doc.add_table(rows=1, cols=1)
        cell = table.rows[0].cells[0]
        shade_cell(cell, CODE_BG)
        cell.text = ""
        for i, line in enumerate(text.strip("\n").split("\n")):
            para = cell.paragraphs[0] if i == 0 else cell.add_paragraph()
            run = para.add_run(line if line else " ")
            run.font.name = "Consolas"
            run.font.size = Pt(9.5)
        self.doc.add_paragraph()

    def table(self, headers: list[str], rows: list[list[str]], widths: list[float] | None = None) -> None:
        t = self.doc.add_table(rows=1, cols=len(headers))
        t.style = "Light Grid Accent 1"
        for i, h in enumerate(headers):
            set_cell_text(t.rows[0].cells[i], h, bold=True)
            shade_cell(t.rows[0].cells[i], "DCE6F1")
        for row_data in rows:
            row = t.add_row().cells
            for i, val in enumerate(row_data):
                set_cell_text(row[i], val)
        self.doc.add_paragraph()

    def image(self, path: Path, width_cm: float = 16.0, caption: str | None = None) -> None:
        if path.exists():
            self.doc.add_picture(str(path), width=Cm(width_cm))
            last = self.doc.paragraphs[-1]
            last.alignment = WD_ALIGN_PARAGRAPH.CENTER
            if caption:
                cap = self.doc.add_paragraph()
                cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run = cap.add_run(caption)
                run.italic = True
                run.font.size = Pt(9)
                run.font.color.rgb = GREY
        else:
            self.p(f"[Diagram not found at {path}]")

    def page_break(self) -> None:
        self.doc.add_page_break()

    def save(self) -> None:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.doc.save(str(OUT_PATH))


def build() -> None:
    kt = KTDoc()

    # ── Title page ──────────────────────────────────────────────────────────
    kt.title_page(
        "BiomarkerAI",
        "Proteomics Multi-Agent Discovery Platform — Knowledge Transfer Document",
        [
            ("Document type", "Knowledge Transfer (KT)"),
            ("Repository", "biomarker_discovery_chatbot"),
            ("GitHub (org)", "github.com/PredictiveScience/biomarker_discovery_chatbot"),
            ("Production host", "Posit Connect — rndconnect.solidbio.com"),
            ("Live app GUID", "66ebc07d-a20e-4715-9fcb-0b20582b731b"),
            ("Date generated", "2026-08-11"),
        ],
    )

    # ── 1. Overview ─────────────────────────────────────────────────────────
    kt.h1("1. Project Overview")
    kt.p(
        "BiomarkerAI is a conversational, multi-agent system for biomarker discovery from "
        "proteomics data. A scientist uploads a protein expression matrix (Olink NPX, MS LFQ, "
        "MS TMT, or a generic numeric matrix) and interacts with the system through natural-"
        "language chat to run differential-expression analysis, pathway enrichment, and "
        "generate publication-style plots — without writing any code themselves."
    )
    kt.p("Core technologies:")
    kt.bullets([
        "LangGraph — orchestration graph; a single node (LearningAgent) makes all routing decisions via LLM reasoning.",
        "Azure OpenAI (GPT-4o) — powers intent classification, grounded chat answers, and code generation/review.",
        "Streamlit — the UI, and since the single-app consolidation, the deployed application itself.",
        "LangSmith — full trace tree observability for every LLM call and agent phase.",
        "pandas / scipy / statsmodels / limma (via Rscript) — statistical analysis engines.",
        "Plotly (+ kaleido where available) — all charting.",
        "gseapy — pathway enrichment against Enrichr (KEGG, GO, Reactome, WikiPathways).",
    ])
    kt.p(
        "The project previously ran as two separate deployed services (a Streamlit UI calling a "
        "FastAPI backend over HTTP). It has since been consolidated into a single Streamlit "
        "application that calls the same business logic in-process. This is the single most "
        "important architectural fact for anyone picking up this codebase — see Section 2."
    )

    # ── 2. Architecture ─────────────────────────────────────────────────────
    kt.h1("2. Architecture")
    kt.p(
        "ui/app.py calls the business logic in core/backend_service.py directly, in-process — "
        "there is no HTTP hop between the UI and the agent workflow in production. A FastAPI "
        "layer (api/routes/*.py) still exists as a thin, optional wrapper around the exact same "
        "backend_service functions, kept for anyone who needs a standalone REST API, but it is "
        "NOT part of the deployed app and is not required to run the chatbot."
    )
    kt.image(DIAGRAM_PATH, width_cm=16.5, caption="Figure 1 — Current production architecture (single Streamlit app on Posit Connect).")

    kt.h2("2.1 Request Flow (step by step)")
    kt.numbered([
        "Upload — the user drops a CSV/Excel file. IngestionAgent detects column types, sample "
        "columns, pooled-vs-replicate design, and builds a label_map. All metadata is stored in "
        "BiomarkerState.",
        "Chat — every user message goes to the LearningAgent orchestrator. Its _make_decision() "
        "method calls the LLM with json_mode=True and returns a DecisionSchema JSON containing "
        "an action, a confidence score, and optional group assignments.",
        "Routing guard — decisions with confidence < 0.7 are demoted to \"answer\" to prevent "
        "accidental/hallucinated analysis triggers. A second guard demotes side-effect actions "
        "(run_analysis, run_enrichment, ...) whose stated reason is under 8 characters.",
        "Analysis — BiomarkerAgent picks the right skill based on the experimental design: "
        "PooledFoldChangeSkill (n=1 per group), ProteomicsAnalysisSkill (replicated groups, "
        "Welch/limma/ANOVA/etc.), or CV-ranking (no groups at all). Produces a ranked Excel "
        "report and stores top_biomarkers in state.",
        "Enrichment — EnrichmentAgent submits the top protein list to UniProt for gene symbol "
        "resolution (capped at 500 proteins), then to Enrichr via gseapy across KEGG, GO, "
        "Reactome, and WikiPathways libraries. Results are stored in state['pathways'].",
        "Visualization — VisualizationAgent generates up to 16 plot types. Plots anchor to the "
        "analysis message in the chat history and stay pinned there across follow-up turns.",
        "LangSmith — wrap_openai() auto-traces every LLM call; @traceable spans name each agent "
        "phase. All traces nest under a single root 'LangGraph' trace per chat turn.",
    ])

    # ── 3. Repository structure ─────────────────────────────────────────────
    kt.h1("3. Repository Structure")
    kt.table(
        ["Path", "Purpose"],
        [
            ["agents/", "One class per pipeline stage: ingestion, biomarker analysis dispatch, "
                        "enrichment, visualization, learning (orchestrator), chat, code_reviewer, "
                        "domain_expert."],
            ["api/", "Optional standalone FastAPI layer. api/main.py + api/routes/*.py wrap the "
                     "same core/backend_service.py functions over HTTP. Not used by the deployed app."],
            ["config/settings.py", "Pydantic BaseSettings — reads .env locally; on Posit Connect, "
                                    "values are set via the content's Settings -> Vars panel instead."],
            ["core/backend_service.py", "THE key module. Framework-agnostic service layer: "
                                          "create_session, upload_file, run_chat_turn, "
                                          "get_analysis_state, get_download_payload, "
                                          "resolve_output_path, build_plot_artifacts, "
                                          "humanise_plot_title. Called directly by ui/app.py."],
            ["core/langgraph_workflow.py", "Compiles the single-node LangGraph StateGraph "
                                            "(START -> learning_agent -> END)."],
            ["core/session_manager.py", "Thread-safe in-memory store of BiomarkerState per "
                                          "session_id. Sessions are lost on process restart."],
            ["core/state.py", "BiomarkerState TypedDict — the single shared state dict threaded "
                               "through the whole pipeline, plus comparison-lookup helpers."],
            ["prompts/", "One .txt system prompt per agent."],
            ["skills/", "Stateless analysis/plotting implementations invoked by the agents, e.g. "
                        "proteomics_analysis.py (stats), run_enrichment.py (gseapy), "
                        "run_visualization.py + plotly_visuals.py (Plotly figures)."],
            ["ui/app.py", "The Streamlit application — the deployed production entrypoint."],
            ["manifest.json", "Posit Connect deployment manifest (generated by rsconnect)."],
            ["DEPLOYMENT.md", "Deployment runbook — local dev + the authoritative Posit Connect "
                                "production redeploy steps."],
            ["tests/", "pytest suite — unit + integration tests across agents/skills/core."],
        ],
    )

    kt.page_break()

    # ── 4. Core modules walkthrough ─────────────────────────────────────────
    kt.h1("4. Core Modules Walkthrough")

    kt.h2("4.1 core/backend_service.py — the service boundary")
    kt.p(
        "This module was extracted from the old api/routes/*.py handlers so the exact same "
        "logic could be called either over HTTP (optional FastAPI path) or directly in-process "
        "(the deployed Streamlit path). It intentionally contains no FastAPI/Starlette types in "
        "any public function signature."
    )
    kt.table(
        ["Function", "Purpose"],
        [
            ["create_session(disease_program, organism)", "Creates a new session via SessionManager; returns session_id."],
            ["delete_session(session_id)", "Deletes a session; raises BackendError(404) if not found."],
            ["upload_file(file_bytes, filename, session_id, ...)", "Validates extension/size, persists the raw file under data/raw/<session_id>/<uuid4hex><ext>, runs IngestionAgent, returns an UploadResponse-shaped dict."],
            ["run_chat_turn(session_id, message, ...)", "Invokes the LangGraph workflow for one chat turn; returns new_assistant_messages, intent, status, and plot metadata."],
            ["get_analysis_state(session_id)", "Returns the full current BiomarkerState for a session (used by results/debug views)."],
            ["resolve_output_path(path)", "Safely resolves a relative output path against OUTPUT_DIR, guarding against path traversal outside outputs/."],
            ["get_download_payload(session_id)", "Returns {content, mime, filename} for the session's Excel biomarker report download."],
            ["build_plot_artifacts(session_id, paths)", "Builds the list of {path, title, ...} plot dicts used by the (optional) FastAPI SSE streaming route."],
            ["humanise_plot_title(path)", "Converts a generated-plot filename into a clean display title — strips hex file-ids, numeric timestamp tokens, and filler words (biomarkers/pooled/plot). Shared by both ui/app.py and build_plot_artifacts() to avoid duplicated, inconsistent logic."],
        ],
    )

    kt.h2("4.2 agents/learning_agent.py — the orchestrator")
    kt.p(
        "LearningAgent is the sole LangGraph node. Every user message is passed through "
        "_make_decision(), which calls the LLM with response_format={'type':'json_object'} "
        "(json_mode=True) and validates the result against a Pydantic DecisionSchema."
    )
    kt.table(
        ["action", "Effect"],
        [
            ["answer", "Grounded chat — injects current top_biomarkers / pathways into the system prompt with citation constraints, then answers directly."],
            ["run_analysis", "Dispatches to BiomarkerAgent for the requested comparison/groups."],
            ["run_enrichment", "Dispatches to EnrichmentAgent (UniProt + Enrichr)."],
            ["run_visualization", "Dispatches to VisualizationAgent for requested plot type(s)."],
            ["show_code", "Returns the stored analysis_code for the current/most-recent analysis."],
            ["modify_code", "LLM rewrites the stored analysis_code per the user's request."],
            ["query_database", "Routes to ProteinLookupSkill (UniProt REST) for protein/gene lookups."],
        ],
    )
    kt.p("Hallucination / mis-routing guards implemented here:")
    kt.bullets([
        "json_mode=True forces valid JSON on every decision call.",
        "DecisionSchema (Pydantic) validates action names and clamps confidence to [0, 1].",
        "confidence < 0.7 demotes any decision to \"answer\".",
        "Side-effect actions additionally require a reason of at least 8 characters, or they are "
        "demoted to \"answer\" too — this catches confident-but-terse (likely misrouted) decisions, "
        "since GPT-4o self-reports >=0.95 confidence almost every time and the raw threshold "
        "rarely fires on its own.",
        "Compound imperatives (\"compare X vs Y, then run pathway analysis and show the volcano "
        "plot\") are returned as an ordered action_sequence and executed step by step instead of "
        "only running the first action.",
        "A deterministic override forces phrases like \"show volcano/heatmap/PCA\" to "
        "run_visualization, with a regex exclusion so it doesn't misfire on conceptual questions "
        "like \"what does a volcano plot show?\".",
    ])

    kt.h2("4.3 BiomarkerState (core/state.py)")
    kt.p(
        "A single TypedDict flows through the whole pipeline per session. Field names are "
        "intentionally generic (top_biomarkers, not top_proteins) so the same state shape can "
        "support other omic types in the future (transcriptomics, metabolomics, lipidomics are "
        "planned but not yet implemented)."
    )
    kt.bullets([
        "messages — LangGraph message history (Annotated with add_messages reducer).",
        "session_id, user_query, intent, active_agent — session/routing bookkeeping.",
        "omic_type — drives which analysis skill BiomarkerAgent dispatches to.",
        "analyses — a list of frozen snapshots, one per comparison run; find_analysis() resolves "
        "which entry a follow-up question refers to (exact id match, fuzzy 'g1 vs g2' match, or "
        "single-label match when unambiguous).",
        "top_biomarkers, pathways, plot_paths — the grounding data injected into chat answers.",
    ])

    kt.page_break()

    # ── 5. Analysis methods ─────────────────────────────────────────────────
    kt.h1("5. Analysis Methods")
    kt.p(
        "BiomarkerAgent picks a test_method (auto-detected from the experimental design, or "
        "explicitly requested by the user/LLM) and dispatches to the matching routine in "
        "skills/proteomics_analysis.py:"
    )
    kt.table(
        ["test_method", "Scenario", "Method"],
        [
            ["welch (default)", "2 groups, unpaired replicates", "Welch t-test + BH FDR + Cohen's d"],
            ["limma", "2 groups, unpaired replicates", "limma eBayes (moderated t-stat) + BH FDR, via an Rscript subprocess"],
            ["paired_t", "Pre/post, matched samples", "Paired t-test on per-subject differences + Cohen's d_z"],
            ["anova", "3+ groups", "One-way ANOVA + Tukey HSD post-hoc (flags which specific group pairs differ)"],
            ["dose_response", "Ordered dose/concentration groups", "Linear trend test across ordered levels (monotonicity, not just omnibus significance)"],
            ["repeated_measures", "Time-course, same subjects across timepoints", "Repeated-measures ANOVA; falls back to a mixed-effects model (MixedLM) for unbalanced designs"],
            ["linear/logistic/cox_regression", "Continuous / binary / survival clinical outcome", "Per-protein regression vs. clinical_outcome; logistic adds ROC AUC"],
        ],
    )
    kt.h2("5.1 Shared preprocessing (all supervised paths)")
    kt.table(
        ["Step", "Method"],
        [
            ["Missing value filter", "Proteins with > 50% NaN removed (group-aware — kept if present in >= threshold of samples in either group)"],
            ["Log2 transform", "Applied when max intensity > 100"],
            ["Imputation", "Half-minimum per protein (computed per row, after saving the pre-imputation detection mask)"],
            ["Multiple testing", "Benjamini-Hochberg FDR"],
            ["Log2FC cap", "+/-20 (prevents Excel/downstream issues)"],
            ["Effect size", "Cohen's d (unpaired) / Cohen's d_z (paired) — sign always matches the log2FC direction"],
        ],
    )
    kt.h2("5.2 Significance tiers")
    kt.table(
        ["Tier", "adj. p", "|log2FC|"],
        [
            ["Highly significant", "< 0.01", ">= 1.0"],
            ["Significant", "< 0.05", ">= 1.0"],
            ["Trend", "< 0.10", "any"],
        ],
    )
    kt.h2("5.3 Special cases")
    kt.bullets([
        "Pooled design (n=1 per group, no replicates) — log2 fold-change with a +1 pseudocount; "
        "all pairwise contrasts are auto-generated from the label map; a generic 'rescue score' "
        "sums positive fold-changes across all contrasts.",
        "Phosphoproteomics/PTM — when ptm_analysis=True, enrichment adds kinase-substrate "
        "libraries (e.g. KEA2021, PhosphoSitePlus) alongside the standard pathway libraries.",
        "Unsupervised (no comparison groups) — ranks by coefficient of variation (CV%) when >= 3 "
        "total samples are available. Single pooled sample (n=1) fallback: when there aren't "
        "enough replicates for CV/MAD/IQR, those fields are None and proteins are instead ranked "
        "by raw mean_expression, so a true n=1 dataset still returns a descriptive ranking "
        "report instead of an empty result or a KeyError.",
    ])

    # ── 6. Enrichment & Plots ────────────────────────────────────────────────
    kt.h1("6. Pathway Enrichment")
    kt.table(
        ["Organism", "Libraries queried"],
        [
            ["Human", "KEGG 2021, GO BP 2023, Reactome 2022, WikiPathways 2023"],
            ["Mouse", "KEGG 2019 Mouse, GO BP 2023, WikiPathways 2019 Mouse"],
            ["Rat", "Mouse libraries used as a proxy — a warning is shown to the user"],
        ],
    )

    kt.h1("7. Visualization / Plots")
    kt.p(
        "Two separate plotting code paths exist and both must be understood by anyone touching "
        "plot generation:"
    )
    kt.bullets([
        "skills/run_visualization.py — on-demand plots explicitly requested in chat (e.g. "
        "waterfall, anova_multigroup). Its _save() helper writes .json + .html + .png for every "
        "figure.",
        "skills/plotly_visuals.py — the automatic post-analysis suite (volcano, PCA, heatmap, "
        "boxplots, ...) built right after an analysis finishes, via "
        "agents/biomarker_agent.py -> build_visualisation_suite(). Its _save_fig() helper was "
        "historically missing the .json export — see Section 9.3 for why that mattered.",
    ])
    kt.p(
        "Every plot is saved in up to three formats: .png (static, via kaleido — requires "
        "Chromium, which is NOT available in the Posit Connect container, so this always fails "
        "in production and is an accepted limitation), .html (standalone interactive, full "
        "Plotly.js embedded), and .json (Plotly figure JSON, loaded by st.plotly_chart() for "
        "in-app interactivity — this is the fallback that MUST exist for a plot to render in "
        "production)."
    )
    kt.p("Request specific plots directly in chat, e.g. \"show me a volcano plot\" or \"give me PCA and heatmap\".")

    kt.page_break()

    # ── 8. Local dev ─────────────────────────────────────────────────────────
    kt.h1("8. Local Development Setup")
    kt.numbered([
        "Clone: git clone https://github.com/PredictiveScience/biomarker_discovery_chatbot.git",
        "Install: make install (or pip install -r requirements.txt)",
        "Configure: cp .env.example .env, then fill in AZURE_OPENAI_* and (optionally) LANGSMITH_* keys.",
        "Create directories: make dirs",
        "Run tests: make test",
        "Start the app: make ui — Streamlit on http://localhost:8501. This single command is "
        "the whole app; make api (FastAPI on :8000) is optional and independent.",
    ])

    # ── 9. Production deployment ─────────────────────────────────────────────
    kt.h1("9. Production Deployment — Posit Connect")
    kt.p(
        "Production runs as a single Posit Connect content item running ui/app.py directly "
        "(python-streamlit app mode, entrypoint declared in manifest.json). Server: "
        "https://rndconnect.solidbio.com."
    )
    kt.table(
        ["Field", "Value"],
        [
            ["Content GUID", "66ebc07d-a20e-4715-9fcb-0b20582b731b"],
            ["Direct URL", "https://rndconnect.solidbio.com/content/66ebc07d-a20e-4715-9fcb-0b20582b731b/"],
            ["Dashboard", "https://rndconnect.solidbio.com/connect/#/apps/66ebc07d-a20e-4715-9fcb-0b20582b731b"],
            ["rsconnect server nickname", "solidbio-rnd"],
        ],
    )
    kt.h2("9.1 Redeploy command")
    kt.code(
        ".venv\\Scripts\\rsconnect.exe deploy manifest --name solidbio-rnd "
        "--app-id 66ebc07d-a20e-4715-9fcb-0b20582b731b manifest.json"
    )
    kt.h2("9.2 Operational gotchas (learned the hard way — read before redeploying)")
    kt.bullets([
        "PowerShell exit code 1 is a known false alarm. rsconnect writes its (verbose but "
        "successful) log to stderr, which PowerShell wraps as a NativeCommandError. Always check "
        "the log text for 'Deployment completed successfully.' rather than trusting the exit "
        "code.",
        "Environment variables are never bundled. AZURE_OPENAI_*, LANGSMITH_*, etc. must be set "
        "once via the content's Settings -> Vars panel in the Connect dashboard, not via a .env "
        "file on the server.",
        "Git-backed content cannot be redeployed by bundle upload. If a content item is "
        "configured to auto-deploy from a linked git repo/branch, manual "
        "'rsconnect deploy manifest' fails with: \"Uploading a content bundle is not allowed for "
        "this application since it is managed by git.\" The current content is a plain "
        "bundle-deployed item, not git-backed.",
        "No Chromium in the Connect container — kaleido (static PNG export) always fails there. "
        "This is expected; the app falls back to interactive Plotly (.json) rendering. Every "
        "plot-saving code path MUST write a .json sidecar or it will silently fail to render in "
        "production.",
        "Regenerate manifest.json before redeploying if dependencies or the file list changed, "
        "and verify .env is not present in the resulting file list — it must never be bundled.",
    ])
    kt.h2("9.3 Orphaned content items (pending decision)")
    kt.p(
        "Two older content items predate the single-app consolidation and are no longer "
        "updated: the original git-backed Streamlit app (bc427f95-f0be-4e52-a971-6e5335ad1eeb) "
        "and a standalone FastAPI deployment (4047059f-eaf9-47e6-9b88-99d15ab14579). A decision "
        "to archive or delete them in the Connect dashboard is still outstanding."
    )

    # ── 10. Env vars ─────────────────────────────────────────────────────────
    kt.h1("10. Environment Variables Reference")
    kt.table(
        ["Variable", "Required", "Default", "Description"],
        [
            ["AZURE_OPENAI_ENDPOINT", "Yes", "—", "Azure OpenAI resource URL"],
            ["AZURE_OPENAI_API_KEY", "Yes", "—", "Azure OpenAI API key"],
            ["AZURE_OPENAI_API_VERSION", "Yes", "2024-08-01-preview", "API version"],
            ["AZURE_DEPLOYMENT_CHAT", "Yes", "gpt-4o", "Deployment for the chat/orchestration agent"],
            ["AZURE_DEPLOYMENT_INGESTION/BIOMARKER/ENRICHMENT/VISUALIZATION", "No", "gpt-4o", "Per-agent model overrides — allows routing different agents to different models"],
            ["DATA_RAW_DIR / DATA_PROCESSED_DIR / OUTPUT_DIR", "No", "data/raw, data/processed, outputs", "Storage locations"],
            ["MAX_FILE_SIZE_MB", "No", "200", "Upload size limit"],
            ["LANGSMITH_API_KEY / LANGSMITH_PROJECT / LANGSMITH_TRACING", "No", "—, biomarker-discovery, true", "Observability"],
            ["APP_ENV / LOG_LEVEL", "No", "development, INFO", "Runtime mode / logging verbosity"],
            ["API_HOST / API_PORT / API_BASE_URL", "No", "0.0.0.0, 8000, http://localhost:8000", "Only relevant to the optional standalone FastAPI layer"],
        ],
    )
    kt.p(
        "Note: a handful of vars sometimes seen in local .env files — ANALYSIS_PROFILE, "
        "STRICT_TEST_METHOD, STRICT_ADJ_PVAL_CUTOFF, STRICT_LOG2FC_CUTOFF, "
        "STRICT_MISSING_VALUE_THRESHOLD, STRICT_TOP_N_BIOMARKERS, STREAMLIT_HOST_REACT_UI, "
        "REACT_UI_URL — are NOT currently read anywhere in the codebase (verified by full-repo "
        "grep). They appear to be reserved for planned-but-unwired features and can be safely "
        "ignored until/unless that work is picked up."
    )

    kt.page_break()

    # ── 11. Git workflow ─────────────────────────────────────────────────────
    kt.h1("11. Git Workflow & Credential Handling")
    kt.table(
        ["Remote", "URL", "Notes"],
        [
            ["origin", "github.com/PredictiveScience/biomarker_discovery_chatbot", "Solid Biosciences org repo — canonical remote"],
            ["origin2", "github.com/ganesan11062001/biomarker_discovery_chatbot", "Personal fork"],
        ],
    )
    kt.p(
        "Security note: the org remote's URL previously had a GitHub PAT embedded in plaintext. "
        "It was stripped from the stored git config. The stored 'origin' remote now has NO "
        "credentials — a plain 'git push origin main' will fail with 'repository not found'. "
        "Pushes to the org repo are done by passing the token inline in the push URL only "
        "(git push https://<token>@github.com/PredictiveScience/biomarker_discovery_chatbot.git "
        "main), and the token is never re-persisted into the stored remote configuration. This "
        "convention should be followed for any future push to origin unless a proper credential "
        "helper is configured."
    )

    # ── 12. Known issues / bug history ──────────────────────────────────────
    kt.h1("12. Known Issue History (useful precedent for future debugging)")
    kt.p(
        "These production bugs were found and fixed via live user testing (real tracebacks and "
        "screenshots), not static analysis — a reminder that dict-shape mismatches in a loosely "
        "typed pipeline (dict[str, Any] state) are often invisible to linters/type checkers."
    )
    kt.h2("12.1 Swallowed exceptions on upload")
    kt.p(
        "ui/app.py's upload handler only caught BackendError; any other exception crashed the "
        "whole Streamlit render with an unstyled traceback. Fixed by broadening the except "
        "clause and adding proper logging + a user-facing st.error()."
    )
    kt.h2("12.2 KeyError: 'response' in chat turns")
    kt.p(
        "Leftover from the old HTTP-era ChatResponse schema ({'response': '...'}). "
        "core.backend_service.run_chat_turn() actually returns new_assistant_messages (a list), "
        "not a flat response string. Fixed by joining new_assistant_messages the same way the "
        "legacy FastAPI JSON route already did."
    )
    kt.h2("12.3 Garbled plot titles")
    kt.p(
        "Plot filenames carry a lot of non-descriptive baggage (a hex upload-id, the literal "
        "word 'pooled', and a timestamp), e.g. "
        "biomarkers_70cf1386839f4f61a6aaef96748f1508_pooled_20260806_162616_anova_multigroup. "
        "A naive title-case of everything after the first underscore left the hex id and "
        "timestamp visible in the UI. Fixed with a shared humanise_plot_title() helper in "
        "core/backend_service.py that tokenizes the filename and drops purely numeric tokens, "
        "hex-looking tokens (8+ hex chars), and known filler words."
    )
    kt.h2("12.4 Heatmap / Boxplots failing to load entirely")
    kt.p(
        "Root cause: skills/plotly_visuals.py's _save_fig() (used for the automatic "
        "post-analysis Heatmap/Boxplots) only wrote .html and .png — never a .json. Since "
        "kaleido/PNG export always fails in the Connect container (no Chromium), and the "
        "Streamlit UI's fallback rendering path requires a .json sidecar, BOTH paths failed for "
        "these two plot types specifically. Other on-demand plots used a different helper "
        "(run_visualization.py's _save()) that already wrote all three formats, which is why "
        "only Heatmap/Boxplots were affected. Fixed by adding fig.to_json() output to "
        "_save_fig(), matching the existing pattern."
    )

    # ── 13. Testing ──────────────────────────────────────────────────────────
    kt.h1("13. Testing")
    kt.code("make test\n# or\npython -m pytest tests/ -v --tb=short")
    kt.p(
        "The suite covers tracing, base agent behaviour, learning agent routing/grounding "
        "(decision schema, confidence gates), and full end-to-end integration scenarios. If the "
        "full run appears to hang near the end, it is usually LangSmith trying (and failing) to "
        "flush traces under test load — set LANGSMITH_TRACING=false for local test runs to avoid "
        "the apparent hang; this does not indicate a broken key/project."
    )

    # ── 14. Troubleshooting ──────────────────────────────────────────────────
    kt.h1("14. Troubleshooting Guide")
    kt.table(
        ["Symptom", "Likely cause", "Fix"],
        [
            ["KeyError: session_id", "Session expired after a server restart", "Re-upload the file — session data is in-memory only"],
            ["Plots not showing in UI", "OUTPUT_DIR not writable, or missing .json sidecar", "Confirm outputs/ exists and is writable; confirm the plot's save helper writes a .json"],
            ["PNG export produces blank files / always fails on Connect", "kaleido has no Chromium in that container", "Expected — rely on the .json/interactive Plotly fallback, not PNG, in production"],
            ["gseapy enrichment fails", "Enrichr API unreachable", "Check connectivity; enrichment returns empty gracefully rather than crashing"],
            ["Analysis returns no proteins", "All proteins filtered by the missing-value threshold", "Lower MISSING_VALUE_THRESHOLD (default 0.5)"],
            ["Upload rejected", "Unsupported file extension", "Accepted: .csv, .xlsx, .xls — rename .txt/.tsv to .csv"],
            ["rsconnect deploy fails: 'managed by git'", "Target content item is git-backed", "Push to its linked branch instead, or deploy to a different (bundle-managed) content item"],
            ["rsconnect deploy exits 1 but logs 'Deployment completed successfully'", "PowerShell wraps rsconnect's stderr log as a NativeCommandError", "Benign — check the log text, not the exit code"],
        ],
    )

    kt.h1("15. Further Reading")
    kt.bullets([
        "README.md — architecture summary, quick start, supported data types, analysis methods.",
        "DEPLOYMENT.md — the authoritative, most detailed Posit Connect redeploy runbook.",
        "scripts/architecture-diagram.png / .mmd — the architecture diagram used in this document.",
    ])

    kt.save()
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    build()
