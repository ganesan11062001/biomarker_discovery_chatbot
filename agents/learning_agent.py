"""
agents/learning_agent.py
Master Orchestrator — LLM-driven reasoning over full session state.

Replaces keyword routing entirely. On every message the LearningAgent:
  1. Sends a structured context block to the LLM and receives a JSON decision
  2. Extracts comparison groups from natural language ("compare WT vs mdx")
  3. Routes to the correct specialist agent with the right parameters
  4. Runs ALL pairwise group comparisons when no specific groups are named
  5. Answers any question — on-topic or off-topic — with full session context

All specialist agents live inside this orchestrator. The LangGraph graph has
a single node: learning_agent → END.
"""
from __future__ import annotations

import json
import logging
import re
from itertools import combinations
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, field_validator

from agents.base_agent import BaseAgent
from config.settings import get_settings
from core.state import BiomarkerState
from core.tracing import get_trace_metadata

settings = get_settings()
logger   = logging.getLogger(__name__)

# ── LangSmith @traceable (graceful no-op if not installed) ───────────────────

try:
    from langsmith import traceable as _traceable
    from langsmith.run_helpers import get_current_run_tree as _get_run_tree
    _LANGSMITH_AVAILABLE = True
except ImportError:
    def _traceable(**_kw):          # type: ignore[misc]
        def _wrap(fn):
            return fn
        return _wrap
    def _get_run_tree():            # type: ignore[misc]
        return None
    _LANGSMITH_AVAILABLE = False


# ── Pydantic schema for LLM decision output ──────────────────────────────────

_VALID_ACTIONS = {
    "load_data", "run_analysis", "run_all_comparisons",
    "run_enrichment", "run_visualization",
    "show_code", "modify_code", "query_database",
    "ask_clarification",
    "answer",
}


class DecisionSchema(BaseModel):
    """
    Validates the JSON object returned by the decision LLM call.
    Invalid actions are caught before any pipeline step is triggered,
    and low-confidence decisions are demoted to 'answer' automatically.
    """
    action:          str            = "answer"
    group1_label:    Optional[str]  = None
    group1_samples:  List[str]      = []
    group2_label:    Optional[str]  = None
    group2_samples:  List[str]      = []
    requested_plots: List[str]      = []
    confidence:      float          = 1.0
    reason:          str            = ""

    # Analysis parameter overrides — extracted from user message when present.
    # Null means "use the session default"; populated value overrides it.
    adj_pval_cutoff:   Optional[float] = None   # e.g. 0.01, 0.05, 0.10
    log2fc_cutoff:     Optional[float] = None   # log2 scale, e.g. 1.0 = 2-fold
    missing_threshold: Optional[float] = None   # fraction, e.g. 0.5 = 50% allowed missing
    top_n:             Optional[int]   = None   # number of proteins to report

    # Extended test method — extracted when user explicitly requests a test type
    test_method:  Optional[str]                     = None  # "welch"|"limma"|"paired_t"|"anova"
    is_paired:    Optional[bool]                    = None  # True for matched/before-after designs
    all_groups:   Optional[Dict[str, List[str]]]    = None  # ANOVA: {group: [cols], ...}
    omic_type:    Optional[str]                     = None  # "proteomics_silac" etc.

    # Clarification question — only used when action == "ask_clarification".
    # Write a complete, kind, professional question the user sees verbatim.
    clarification_question: Optional[str]           = None

    @field_validator("action")
    @classmethod
    def _validate_action(cls, v: str) -> str:
        return v if v in _VALID_ACTIONS else "answer"

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, v) -> float:
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 1.0

    @field_validator("adj_pval_cutoff", mode="before")
    @classmethod
    def _clamp_pval(cls, v) -> Optional[float]:
        if v is None:
            return None
        try:
            return max(1e-6, min(1.0, float(v)))
        except (TypeError, ValueError):
            return None

    @field_validator("log2fc_cutoff", mode="before")
    @classmethod
    def _clamp_lfc(cls, v) -> Optional[float]:
        if v is None:
            return None
        try:
            return max(0.0, min(20.0, float(v)))
        except (TypeError, ValueError):
            return None

    @field_validator("missing_threshold", mode="before")
    @classmethod
    def _clamp_missing(cls, v) -> Optional[float]:
        if v is None:
            return None
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return None

    @field_validator("top_n", mode="before")
    @classmethod
    def _clamp_topn(cls, v) -> Optional[int]:
        if v is None:
            return None
        try:
            return max(1, min(5000, int(v)))
        except (TypeError, ValueError):
            return None

    @field_validator("test_method", mode="before")
    @classmethod
    def _validate_test_method(cls, v) -> Optional[str]:
        if v is None:
            return None
        valid = {"auto", "welch", "limma", "paired_t", "anova"}
        s = str(v).lower().strip()
        return s if s in valid else None

    @field_validator("omic_type", mode="before")
    @classmethod
    def _validate_omic_type(cls, v) -> Optional[str]:
        if v is None:
            return None
        valid = {"proteomics", "proteomics_pooled", "proteomics_silac"}
        s = str(v).lower().strip()
        return s if s in valid else None


# ── Prompts ───────────────────────────────────────────────────────────────────

_DECISION_SYSTEM_PROMPT = """\
You are the master orchestrator for a proteomics biomarker discovery AI platform.

Given the session state and a user message, output a single JSON decision object
that drives the pipeline. Choose exactly one action:

  "load_data"           — user wants to upload / load a new data file
  "run_analysis"        — run biomarker analysis for a SPECIFIC comparison the user named
  "run_all_comparisons" — run ALL pairwise group comparisons (user said "all" or didn't specify groups)
  "run_enrichment"      — run KEGG / GO pathway enrichment on current biomarker results
  "run_visualization"   — generate plots, heatmaps, charts, or a report
  "show_code"           — user wants to see the reproducible analysis code
  "modify_code"         — user wants to change, alter, or extend the analysis code
  "query_database"      — look up protein info, gene names, UniProt annotation, or convert IDs
  "ask_clarification"   — ask the user a focused, professional question before proceeding
  "answer"              — answer a question, explain something, or have a conversation

Analysis routing — the pipeline automatically selects the right method:
  • Regular proteomics (CSV or standard Excel, any number of replicates) →
      Test method auto-selected: limma eBayes for n≤4 per group, Welch t-test for n≥5.
      If ≥2 samples per group: supervised differential expression (log₂FC, Cohen's d, adj. p-value).
      If no group labels given: unsupervised CV/MAD/IQR variability ranking.
  • Pooled n=1 design (MaxQuant Excel with Identifier Info label sheet, one sample per label) →
      log₂ fold-change across all pairwise contrasts (no p-values, by design).
  • SILAC data (H/L or H/M ratios detected) → SilacAnalysisSkill (omic_type="proteomics_silac").
  • DIA/Spectronaut output → automatically reshaped, then ProteomicsAnalysisSkill.
  • Multi-batch TMT with IRS → IRS normalisation applied automatically when plex structure detected.

TEST METHOD EXTRACTION (populate "test_method" when user explicitly requests one):
  "limma"    — user says "limma", "moderated t-test", "eBayes", "small n", "few replicates"
  "welch"    — user says "Welch t-test", "standard t-test", "regular t-test"
  "paired_t" — user says "paired", "matched samples", "before/after", "pre/post", "same subject"
  "anova"    — user says "ANOVA", "more than 2 groups", "multiple groups simultaneously", "F-test"
  Leave null for "auto" (default; pipeline auto-selects limma vs Welch by sample size).

MULTI-GROUP ANOVA (populate "all_groups" when user has >2 groups for ANOVA):
  When user says "compare WT, KO, and HET" or "ANOVA across all 4 groups":
    • Set test_method = "anova"
    • Set all_groups = {"WT": ["WT_1","WT_2"], "KO": ["KO_1","KO_2"], "HET": ["HET_1","HET_2"]}
    • Leave group1_samples / group2_samples empty

PAIRED DESIGN (set is_paired = true when user describes matched samples):
  "compare before and after treatment for each patient" → is_paired = true
  "paired t-test with samples matched by patient ID" → is_paired = true, test_method = "paired_t"

SILAC (set omic_type when user specifies data type):
  "my data is SILAC" / "heavy/light ratios" → omic_type = "proteomics_silac"

Decision rules (in priority order):
1.  Questions ("what is X", "explain X", "how does Y work", "what did the analysis find") → "answer"
2.  Off-topic messages → "answer"
3.  No data loaded yet → "answer" (tell user to upload a file first)
4.  "show code" / "give me the code" / "what code was used" → "show_code"
5.  Re-run with new parameter values (see below) → "run_analysis" + fill parameter fields
6.  "change the code to use X method" / "add a step to the script" → "modify_code"
7.  "look up proteins" / "get gene names" / "annotate" / "UniProt" / "convert IDs" → "query_database"
8.  Pathway / enrichment / KEGG / GO → "run_enrichment"
9.  Plot / visualize / chart / heatmap / volcano / report → "run_visualization"
10. Pooled design AND "run analysis" with no specific group pair → "run_all_comparisons"
11. "run analysis" / "analyze" / "find biomarkers" with NO specific group pair named → "run_all_comparisons"
12. "run analysis" / "analyze" with SPECIFIC group names (e.g. "Disease vs Control") → "run_analysis"
    - Set group1_label, group1_samples, group2_label, group2_samples from available_columns.
    - Leave sample lists empty if you cannot confidently match column names.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CLARIFICATION PHILOSOPHY  —  ask the user rather than assume anything uncertain
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You are a professional scientific collaborator. Your primary responsibility is to
run the *right* analysis, not the fastest one. Whenever anything in the user's
request or the session data is ambiguous — at any stage of the pipeline — pause
and ask a clear, focused question before proceeding.

USE "ask_clarification" whenever you are genuinely uncertain about:
  • Which groups or samples to compare
  • What experimental design the data represents (paired, independent, time-series)
  • Which statistical method is most appropriate given the context
  • How many groups should be analysed and whether simultaneously or pairwise
  • What a column, label, or parameter means in the user's experiment
  • Whether a detected data feature (SILAC ratios, TMT batches, pooled design,
    Spectronaut output) matches the user's actual experiment
  • What the user wants to do next when their message is ambiguous
  • Any detail where guessing wrong would produce misleading biological results

This applies across the ENTIRE workflow — data loading, group assignment,
statistical testing, enrichment, visualisation, and interpretation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO WRITE EVERY CLARIFICATION QUESTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tone: Kind, warm, professional. Never imply the user did anything wrong.
Structure:
  1. One sentence acknowledging what you *can* see or have understood.
  2. The specific question — concrete, not vague.
  3. If there are distinct choices, list them as numbered options with a brief
     explanation of what each one does scientifically.
  4. A short closing that makes it easy to reply (e.g. "Just let me know!" or
     "Feel free to reply with whichever fits your experiment.").

Ask only ONE question per turn. If multiple things are unclear, ask about the
most critical one first and resolve the others in subsequent turns.

Use markdown formatting (bold for key terms, inline code for column names,
bullet/number lists for options) — the question is shown verbatim in chat.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMMON SITUATIONS THAT WARRANT CLARIFICATION (non-exhaustive — use judgement)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• Groups not specified and column names don't clearly imply them
• Sample names suggest a paired/longitudinal design (e.g. `Pat1_Pre`, `Pat1_Post`)
  but the user hasn't confirmed it
• Three or more groups detected — should analysis be ANOVA or pairwise?
• SILAC ratio data — single-condition (H/L vs 1) or two-condition (comparing ratios)?
• Multi-batch TMT detected but no reference channel column can be identified
• User requests a specific comparison but some named columns don't exist in the data
• User asks to "re-run with different thresholds" but doesn't specify which thresholds
• Enrichment requested but no significant proteins from a previous analysis are available
• Visualisation requested but it's unclear which plot types or which comparison to show
• Any request where two reasonable interpretations would produce different results
• User's message is genuinely ambiguous (e.g. "compare the groups" with no prior context)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHEN NOT TO ASK (proceed without clarification)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• The user's current or recent message already contains the answer
• The intent is unambiguous and the pipeline can proceed correctly without guessing
• The user is asking a general question (not requesting an action) — just answer it
• Pooled MaxQuant design is auto-detected — the routing is determined automatically
• The user is responding to a clarification you already asked — use their answer now
• The detail is minor and the default is scientifically reasonable for most cases
  (e.g. auto-selecting limma for small n is always appropriate — no need to ask)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PARAMETER EXTRACTION (for run_analysis and run_all_comparisons):
Extract analysis thresholds from the user message and populate the relevant fields.
Leave a field null if the user did not mention it.

  adj_pval_cutoff   — adjusted p-value / FDR threshold (0.0–1.0)
    Examples: "p-value < 0.01" → 0.01 | "5% FDR" → 0.05 | "stricter p-value of 0.001" → 0.001

  log2fc_cutoff     — minimum |log₂ fold-change| (already in log2 scale)
    Examples: "log2FC > 1.5" → 1.5 | "2-fold change" → 1.0 (log2(2)=1) | "FC threshold 2 on log2 scale" → 2.0

  missing_threshold — maximum fraction of missing values allowed per protein (0.0–1.0)
    Examples: "allow 60% missing" → 0.6 | "require 70% valid values" → 0.3 | "30% missing threshold" → 0.3

  top_n             — number of top proteins/biomarkers to report (integer)
    Examples: "top 100 proteins" → 100 | "show me 200 results" → 200

IMPORTANT: When the user says "re-run with p<0.01", "change threshold to 0.01 and re-analyze",
"use stricter cutoffs and run again", or similar → action = "run_analysis" (NOT "modify_code").
Only use "modify_code" when the user wants to change the code itself (add a step, change an algorithm, etc.)
that CANNOT be done by adjusting a threshold value.

For "run_analysis" populate groups only when you can confidently match column names to group labels.

OUTPUT: valid JSON only — no markdown fences, no prose, no trailing text.
{
  "action": "<action>",
  "group1_label": "<label or null>",
  "group1_samples": [],
  "group2_label": "<label or null>",
  "group2_samples": [],
  "requested_plots": [],
  "confidence": 0.95,
  "reason": "<one sentence explaining the decision>",
  "adj_pval_cutoff": null,
  "log2fc_cutoff": null,
  "missing_threshold": null,
  "top_n": null,
  "test_method": null,
  "is_paired": null,
  "all_groups": null,
  "omic_type": null,
  "clarification_question": null
}

For "ask_clarification": set "clarification_question" to the full question text
(markdown supported). Leave all group/param fields null. Set confidence ≥ 0.9.

"confidence" is a float 0.0–1.0. Decisions with confidence < 0.7 are auto-demoted to "answer".

For "run_visualization": populate "requested_plots" with canonical plot names the user asked for.
Leave it empty [] if the user wants all standard plots or was not specific.
Available plot names: volcano, ma_plot, heatmap, pca, boxplot, sample_correlation,
cv_distribution, fc_heatmap, topn_bar, rescue_bar, pathway_dotplot.
"""

_ANSWER_SYSTEM_PROMPT = """\
You are an expert AI assistant embedded in a proteomics biomarker discovery platform.

You have deep knowledge of:
- Proteomics: Olink NPX, MaxQuant LFQ, TMT, iTRAQ, spectral counting
- Statistics: t-tests, ANOVA, fold-change, FDR correction (BH, Bonferroni)
- Bioinformatics: pathway enrichment, KEGG, GO, differential expression
- General science, machine learning, and statistics

ANTI-HALLUCINATION RULES (strictly enforced):

RULE 1 — SESSION DATA IS THE ONLY SOURCE OF TRUTH FOR "MY" QUESTIONS:
  When the user asks anything about their uploaded data — including but not limited to:
    "describe my data", "what samples do I have", "what groups were detected",
    "how many proteins", "summarize my dataset", "what is in my file",
    "what did the analysis find", "what are my top biomarkers"
  — you MUST answer ONLY from the session context provided below.
  Do NOT supplement, guess, or fill gaps with general proteomics knowledge.
  If the session context does not contain the answer, say:
    "I don't have that information yet — [action needed, e.g. run analysis first]."
  Never describe what a "typical" proteomics dataset looks like as if it were the user's data.

RULE 2 — GROUNDED RESULTS ONLY:
  When referencing analysis results (proteins, fold-change values, p-values, pathways),
  ONLY cite values explicitly listed in the grounded data sections of the session context.
  If a specific value is not in the session data, say "that value is not in the current results."

RULE 3 — GENERAL SCIENCE QUESTIONS ARE FREE:
  For questions that are clearly general / off-topic (not about the user's specific file
  or session), answer freely from your training knowledge. Examples: "what is a t-test?",
  "how does KEGG enrichment work?", "explain PCA". These do NOT involve the user's data.

RULE 4 — NO INVENTED IDENTIFIERS:
  Do NOT invent protein names, gene symbols, accession IDs, or pathway names
  that are not grounded in the session context or your verified training knowledge.

RULE 5 — FORMAT:
  Use markdown formatting. Be concise and precise. For session-data summaries,
  present actual numbers from the context (n_proteins, n_samples, sample_columns, etc.)
  rather than generic descriptions.
"""

_GROUP_INFERENCE_PROMPT = """\
You are a bioinformatics data analyst.
Below are sample column names from a proteomics dataset.
Identify biological groups by looking for common naming patterns:
  - Explicit group names (WT, KO, Disease, Control, mdx, uDys5, etc.)
  - Numeric suffixes indicating replicates (_1, _2, _3 or .1 .2 .3)
  - Common prefixes

Return a JSON object mapping group name → list of column names.
Example: {"WT": ["WT_1","WT_2","WT_3"], "KO": ["KO_1","KO_2","KO_3"]}
If you cannot detect groups, return {}.
Output ONLY the JSON object, nothing else.
"""


# ── Module-level helpers ──────────────────────────────────────────────────────

def _truncate(text: str, max_len: int = 600) -> str:
    """Truncate a string to max_len characters with ellipsis indicator."""
    text = str(text)
    return text if len(text) <= max_len else text[:max_len] + "…[truncated]"


def _recent_messages(
    messages: list,
    n: int = 20,
    truncate_at: int = 600,
) -> list:
    """Return the last n messages as plain dicts with content truncated."""
    result = []
    for m in messages:
        if isinstance(m, dict):
            role    = m.get("role", "assistant")
            content = m.get("content", "")
        elif hasattr(m, "content"):
            msg_type = getattr(m, "type", "") or type(m).__name__.lower()
            role = "assistant" if ("ai" in msg_type or "assistant" in msg_type) else "user"
            content = str(m.content)
        else:
            continue
        if role in ("user", "assistant"):
            result.append({"role": role, "content": _truncate(content, truncate_at)})
    return result[-n:]


class LearningAgent(BaseAgent):
    """
    Single-node orchestrator that uses LLM reasoning to coordinate all
    specialist agents. Handles intent, group extraction, multi-comparison,
    and general Q&A in one place.
    """

    def __init__(self) -> None:
        super().__init__(
            deployment_name=settings.azure_deployment_chat,
            system_prompt_path="prompts/chat_agent.txt",
        )
        self._specialists: Dict[str, Any] = {}

    # ── Lazy specialist access ────────────────────────────────────────────────

    def _specialist(self, name: str) -> Any:
        if name not in self._specialists:
            if name == "ingestion":
                from agents.ingestion_agent import IngestionAgent
                self._specialists[name] = IngestionAgent()
            elif name == "biomarker":
                from agents.biomarker_agent import BiomarkerAgent
                self._specialists[name] = BiomarkerAgent()
            elif name == "enrichment":
                from agents.enrichment_agent import EnrichmentAgent
                self._specialists[name] = EnrichmentAgent()
            elif name == "visualization":
                from agents.visualization_agent import VisualizationAgent
                self._specialists[name] = VisualizationAgent()
        return self._specialists[name]

    # ── LLM decision ─────────────────────────────────────────────────────────

    @_traceable(run_type="chain", name="orchestrator.decision",
                tags=["biomarker-discovery", "decision"])
    def _make_decision(self, state: BiomarkerState) -> Dict[str, Any]:
        """
        Return a validated action decision dict from the LLM.

        LangSmith span: orchestrator.decision
          • input  — structured context block sent to the LLM
          • output — validated DecisionSchema dict
          • LLM child span auto-created by wrap_openai (latency + tokens)

        Hallucination guards applied here:
          1. json_mode=True forces valid JSON from the model (no markdown fences)
          2. DecisionSchema validates the action field against the allowed set
          3. Confidence < 0.7 is demoted to "answer" to prevent misrouted actions
        """
        sample_cols = state.get("sample_columns") or []
        label_map   = state.get("label_map") or {}
        top_bm      = state.get("top_biomarkers") or []
        g1_samps    = state.get("group1_samples") or []
        g2_samps    = state.get("group2_samples") or []
        g1_lbl      = state.get("group1_label") or ""
        g2_lbl      = state.get("group2_label") or ""

        ctx  = "SESSION STATE:\n"
        ctx += f"  data_loaded: {bool(state.get('data_type'))}\n"
        ctx += f"  data_type: {state.get('data_type', 'none')}\n"
        ctx += f"  n_proteins: {state.get('n_proteins', 0)}\n"
        ctx += f"  n_samples: {state.get('n_samples', 0)}\n"
        ctx += f"  is_pooled_design: {state.get('is_pooled_design', False)}\n"
        ctx += f"  omic_type: {state.get('omic_type', 'none')}\n"
        ctx += f"  data_type: {state.get('data_type', 'none')}\n"
        ctx += f"  analysis_complete: {state.get('n_significant') is not None}\n"
        ctx += f"  n_significant: {state.get('n_significant', 'none')}\n"
        ctx += f"  analysis_mode: {state.get('analysis_mode', 'none')}\n"
        ctx += f"  has_analysis_code: {bool(state.get('analysis_code'))}\n"
        ctx += f"  has_plots: {bool(state.get('plot_paths'))}\n"
        ctx += f"  enrichment_done: {bool(state.get('pathways'))}\n"
        ctx += f"  status: {state.get('status', 'ready')}\n"
        ctx += f"  is_paired: {state.get('is_paired', False)}\n"
        ctx += f"  test_method_set: {(state.get('analysis_params') or {}).get('test_method', 'auto')}\n"

        # TMT batch structure hint
        tmt = state.get("tmt_batches")
        if tmt:
            has_ref = all(v.get("reference") for v in tmt.values())
            ctx += f"  tmt_batches_detected: {list(tmt.keys())} | all_refs_found: {has_ref}\n"
        else:
            ctx += "  tmt_batches_detected: none\n"

        # ALL sample columns — the LLM must see these to populate group_samples correctly.
        ctx += f"  all_sample_columns ({len(sample_cols)} total): {sample_cols[:100]}\n"
        if label_map:
            ctx += f"  pooled_label_map: {label_map}\n"

        # Current group assignments (from chat or ingestion auto-detection).
        if g1_samps or g2_samps:
            ctx += (
                f"  currently_assigned_groups:\n"
                f"    '{g1_lbl}' ({len(g1_samps)} samples): {g1_samps}\n"
                f"    '{g2_lbl}' ({len(g2_samps)} samples): {g2_samps}\n"
            )
        else:
            ctx += "  currently_assigned_groups: none\n"

        # Inferred group count from column patterns (helps LLM decide on ANOVA vs pairwise)
        all_groups_state = state.get("all_groups")
        if all_groups_state:
            ctx += f"  all_groups_assigned: {list(all_groups_state.keys())}\n"

        if top_bm:
            ctx += f"  top_5_biomarkers: {[b.get('protein','') for b in top_bm[:5]]}\n"

        # Include last 5 conversation turns so the LLM knows what was recently discussed
        recent = _recent_messages(state.get("messages") or [], n=5)
        if recent:
            ctx += "\nRECENT CONVERSATION:\n"
            for m in recent:
                role    = m.get("role", "?")
                content = _truncate(m.get("content", ""), 200)
                ctx += f"  [{role}]: {content}\n"

        ctx += f"\nCURRENT USER MESSAGE: \"{state.get('user_query', '')}\""

        messages = [
            {"role": "system", "content": _DECISION_SYSTEM_PROMPT},
            {"role": "user",   "content": ctx},
        ]
        try:
            # json_mode=True: forces valid JSON output — no markdown fences,
            # no preamble — eliminating the most common structured-output failure.
            raw = self._call_llm(
                messages, max_tokens=350, temperature=0.0, json_mode=True
            ).strip()

            # Validate + coerce with Pydantic (catches unknown actions, bad types)
            decision_obj = DecisionSchema.model_validate(json.loads(raw))

            # Confidence gate: distrust low-confidence routing decisions
            if decision_obj.confidence < 0.7:
                self.logger.warning(
                    "Low-confidence decision (%.2f) for action=%s — demoting to 'answer'",
                    decision_obj.confidence, decision_obj.action,
                )
                decision_obj.action = "answer"

            decision = decision_obj.model_dump()
            self.logger.info(
                "Decision: action=%s | confidence=%.2f | reason=%s",
                decision["action"], decision["confidence"], decision.get("reason", ""),
            )
            return decision
        except Exception as exc:
            self.logger.warning("Decision LLM failed (%s) — defaulting to answer.", exc)
            return {"action": "answer", "confidence": 1.0, "reason": "LLM fallback"}

    # ── Group column matching ─────────────────────────────────────────────────

    @staticmethod
    def _match_columns_by_label(label: str, columns: List[str]) -> List[str]:
        """
        Return columns whose names match a group label by prefix or substring.

        Examples
        --------
        label="WT",      columns=["WT_1","WT_2","KO_1"] → ["WT_1","WT_2"]
        label="Control", columns=["ctrl_A","ctrl_B","dis_A"] → ["ctrl_A","ctrl_B"]
        """
        if not label or not columns:
            return []
        lbl = label.strip().lower()
        # Exact prefix (most reliable — "WT" → "WT_1")
        hits = [c for c in columns if c.lower().startswith(lbl)]
        if hits:
            return hits
        # Substring (fallback — "control" → "sample_control_1")
        hits = [c for c in columns if lbl in c.lower()]
        return hits

    # ── Group inference ───────────────────────────────────────────────────────

    def _infer_groups(self, sample_columns: List[str]) -> Dict[str, List[str]]:
        """Ask LLM to detect group structure from column naming patterns."""
        if not sample_columns:
            return {}
        prompt = _GROUP_INFERENCE_PROMPT + f"\nColumn names:\n{sample_columns}"
        messages = [
            {"role": "system", "content": "You are a bioinformatics data analyst."},
            {"role": "user",   "content": prompt},
        ]
        try:
            raw = self._call_llm(messages, max_tokens=400, temperature=0.0).strip()
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
            groups = json.loads(raw)
            if isinstance(groups, dict) and groups:
                self.logger.info("Inferred groups: %s", {k: len(v) for k, v in groups.items()})
                return groups
        except Exception as exc:
            self.logger.warning("Group inference failed: %s", exc)
        return {}

    # ── Multi-comparison ──────────────────────────────────────────────────────

    def _run_all_comparisons(self, state: BiomarkerState) -> BiomarkerState:
        """
        Route to the appropriate analysis for the loaded data:

        • Pooled n=1 design (is_pooled_design=True, no replicates):
          Delegate entirely to BiomarkerAgent → PooledFoldChangeSkill.
          That skill computes all pairwise log₂FC contrasts in a single pass.

        • Standard proteomics (replicated groups):
          Infer groups from column names, then run Welch t-test + BH FDR for
          every pairwise combination via ProteomicsAnalysisSkill.
        """
        is_pooled = state.get("is_pooled_design") or state.get("omic_type") == "proteomics_pooled"

        # For truly pooled n=1 designs, PooledFoldChangeSkill handles everything
        if is_pooled:
            label_map = state.get("label_map") or {}
            groups = list(label_map.values()) if label_map else ["all groups"]
            state["messages"].append({
                "role": "assistant",
                "content": (
                    f"Pooled n=1 design detected. Running log₂ fold-change analysis "
                    f"across all pairwise contrasts for groups: **{', '.join(groups)}** …"
                ),
            })
            return self._specialist("biomarker").run(state)

        # Standard replicated proteomics — infer groups and run pairwise DEA
        sample_cols = state.get("sample_columns") or []
        groups = self._infer_groups(sample_cols)

        if len(groups) < 2:
            state["messages"].append({
                "role": "assistant",
                "content": (
                    "I couldn't automatically detect groups from your column names. "
                    "Please tell me which groups to compare, e.g.: "
                    "*'compare Control_1, Control_2, Control_3 vs Disease_1, Disease_2, Disease_3'*."
                ),
            })
            return state

        group_names = list(groups.keys())
        pairs = list(combinations(group_names, 2))
        state["messages"].append({
            "role": "assistant",
            "content": (
                f"Running standard proteomics differential expression analysis "
                f"(**{len(pairs)} pairwise comparisons** across {len(group_names)} groups: "
                f"{', '.join(group_names)}).\n\n"
                f"Pipeline: log₂ transform → median normalisation → group-aware filter "
                f"→ half-min imputation → Welch t-test → BH FDR."
            ),
        })

        biomarker = self._specialist("biomarker")
        summary_lines: List[str] = []

        for g1_name, g2_name in pairs:
            state["group1_label"]   = g1_name
            state["group1_samples"] = groups[g1_name]
            state["group2_label"]   = g2_name
            state["group2_samples"] = groups[g2_name]
            state["analysis_mode"]  = "supervised"
            # Force standard proteomics analysis regardless of what was loaded
            state["omic_type"]      = "proteomics"

            state = biomarker.run(state)
            n_sig = state.get("n_significant", 0)
            top3 = [b.get("protein", "") for b in (state.get("top_biomarkers") or [])[:3]]
            summary_lines.append(
                f"- **{g1_name} vs {g2_name}**: {n_sig} significant | "
                f"top: {', '.join(top3)}"
            )

        state["messages"].append({
            "role": "assistant",
            "content": "**All pairwise comparisons complete:**\n" + "\n".join(summary_lines),
        })
        return state

    # ── Code display & modification ──────────────────────────────────────────

    def _show_code(self, state: BiomarkerState) -> BiomarkerState:
        code = state.get("analysis_code")
        if not code:
            state["messages"].append({
                "role": "assistant",
                "content": (
                    "No analysis code available yet. Run an analysis first and the "
                    "reproducible Python script will be generated automatically."
                ),
            })
        else:
            state["messages"].append({
                "role": "assistant",
                "content": (
                    "Here is the reproducible Python script used for this analysis:\n\n"
                    "```python\n" + code + "\n```\n\n"
                    "_You can run this script directly or ask me to modify it._"
                ),
            })
        state["status"] = "answered"
        return state

    def _modify_code(self, state: BiomarkerState) -> BiomarkerState:
        """Use LLM to alter the stored analysis code based on the user's request."""
        code = state.get("analysis_code")
        if not code:
            state["messages"].append({
                "role": "assistant",
                "content": "No analysis code to modify yet. Run an analysis first.",
            })
            state["status"] = "answered"
            return state

        user_request = state.get("user_query", "")
        prompt = (
            "You are an expert Python bioinformatician.\n\n"
            "Here is the current proteomics analysis script:\n\n"
            "```python\n" + code + "\n```\n\n"
            f"User request: {user_request}\n\n"
            "Modify the script according to the user's request. "
            "Keep all imports, parameters, and structure intact unless the request explicitly changes them. "
            "Output ONLY the complete modified Python script — no explanation, no markdown fences."
        )
        messages = [
            {"role": "system", "content": "You are an expert Python bioinformatician."},
            {"role": "user",   "content": prompt},
        ]
        try:
            modified = self._call_llm(messages, max_tokens=2000, temperature=0.0).strip()
            # Strip markdown fences if the LLM added them (Python 3.8-compatible)
            modified = re.sub(r"^```(?:python)?\s*", "", modified, flags=re.MULTILINE)
            modified = re.sub(r"\s*```$", "", modified, flags=re.MULTILINE).strip()
            state["analysis_code"] = modified
            state["messages"].append({
                "role": "assistant",
                "content": (
                    "Here is the modified analysis script:\n\n"
                    "```python\n" + modified + "\n```"
                ),
            })
        except Exception as exc:
            self.logger.warning("Code modification LLM failed: %s", exc)
            state["messages"].append({
                "role": "assistant",
                "content": f"Failed to modify the code: {exc}",
            })
        state["status"] = "answered"
        return state

    # ── Database query (UniProt) ──────────────────────────────────────────────

    def _query_database(self, state: BiomarkerState) -> BiomarkerState:
        """
        Look up top biomarkers via UniProt REST API.
        Returns gene annotations, resolved names, and reproducible code.
        """
        from skills.protein_lookup import ProteinLookupSkill
        from config.settings import get_settings

        protein_source = state.get("top_biomarkers") or state.get("top_proteins")
        if not protein_source:
            state["messages"].append({
                "role": "assistant",
                "content": (
                    "No biomarker results to look up. Run an analysis first so I have "
                    "a protein list to annotate."
                ),
            })
            state["status"] = "answered"
            return state

        protein_list = [p.get("protein", "") for p in protein_source if p.get("protein")]
        n_total = len(protein_list)

        state["messages"].append({
            "role": "assistant",
            "content": f"Querying UniProt for {n_total} proteins — please wait…",
        })

        skill = ProteinLookupSkill()
        _s = get_settings()
        result = skill.execute(
            protein_list=protein_list,
            organism=state.get("organism") or "human",
            output_dir=_s.output_dir,
        )

        annotations = result["annotations"]
        gene_symbols = result["gene_symbols"]
        n_resolved   = result["n_resolved"]
        code         = result.get("analysis_code", "")

        # Build a markdown table of annotations
        if annotations:
            rows = ["| Accession | Gene | Protein | Organism | Reviewed |",
                    "|-----------|------|---------|----------|----------|"]
            for a in annotations[:20]:
                rows.append(
                    f"| {a.get('accession','')} | {a.get('gene','')} | "
                    f"{a.get('protein_name','')[:40]} | {a.get('organism','')[:20]} | "
                    f"{'✓' if a.get('reviewed') else ''} |"
                )
            table = "\n".join(rows)
        else:
            table = "_No accession IDs detected — used regex extraction only._"

        msg = (
            f"**UniProt annotation complete**\n\n"
            f"- Proteins submitted: **{n_total}**\n"
            f"- Resolved via UniProt API: **{n_resolved}**\n"
            f"- Gene symbols extracted: **{len(gene_symbols)}**\n\n"
            f"{table}\n\n"
        )
        if gene_symbols:
            msg += f"**Gene symbols** (first 20): `{', '.join(gene_symbols[:20])}`\n\n"
        if code:
            msg += (
                "---\n**Reproducible lookup code:**\n\n"
                "```python\n" + code + "\n```\n"
            )

        state["messages"].append({"role": "assistant", "content": msg})
        state["status"] = "answered"
        return state

    # ── General answer ────────────────────────────────────────────────────────

    @_traceable(run_type="chain", name="orchestrator.answer",
                tags=["biomarker-discovery", "answer"])
    def _answer(self, state: BiomarkerState) -> BiomarkerState:
        """
        Answer any question using full session context + LLM knowledge.

        LangSmith span: orchestrator.answer
          • Injects actual biomarker list as a grounding anchor so the LLM
            cannot fabricate protein names or statistics that differ from what
            was computed.
        """
        ctx = ["## Session context (ONLY use this when answering questions about the user's data)"]
        if state.get("data_type"):
            sample_cols = state.get("sample_columns") or []
            label_map   = state.get("label_map") or {}
            g1          = state.get("group1_label")
            g2          = state.get("group2_label")
            g1_samps    = state.get("group1_samples") or []
            g2_samps    = state.get("group2_samples") or []

            ctx += [
                f"- Data loaded: YES",
                f"- Proteins: {state.get('n_proteins','?')}",
                f"- Samples: {state.get('n_samples','?')}",
                f"- Data type: {state.get('data_type','?')}",
                f"- Omic type: {state.get('omic_type','proteomics')}",
                f"- Pooled design: {state.get('is_pooled_design', False)}",
                f"- Organism: {state.get('organism', 'not set')}",
                f"- Disease program: {state.get('disease_program', 'General')}",
                f"- Sample columns (first 20): {sample_cols[:20]}",
            ]
            if label_map:
                ctx.append(f"- Pooled groups (label map): { {k: v for k, v in label_map.items()} }")
            elif g1 and g2:
                ctx += [
                    f"- Group 1: {g1} — {len(g1_samps)} samples: {g1_samps}",
                    f"- Group 2: {g2} — {len(g2_samps)} samples: {g2_samps}",
                ]
            else:
                ctx.append("- Groups: not yet assigned")

            if state.get("n_significant") is not None:
                top5 = [b.get("protein","") for b in (state.get("top_biomarkers") or [])[:5]]
                ctx += [
                    "- Analysis complete: YES",
                    f"- Significant biomarkers: {state.get('n_significant')}",
                    f"- Top 5 proteins: {top5}",
                    f"- Method: {state.get('analysis_mode','?')} mode",
                    f"- Comparison: {g1 or '?'} vs {g2 or '?'}",
                ]
            else:
                ctx.append("- Analysis complete: NO — analysis has not been run yet")
            if state.get("pathways"):
                top3pw = [p.get("pathway","") for p in state["pathways"][:3]]
                ctx.append(f"- Enrichment done: YES — top pathways: {top3pw}")
            else:
                ctx.append("- Enrichment done: NO")
            if state.get("plot_paths"):
                ctx.append(f"- Plots generated: {len(state['plot_paths'])}")
        else:
            ctx.append("- Data loaded: NO — user has not uploaded a file yet")

        # ── Grounding anchor: inject actual values so LLM cannot hallucinate ──
        # This is the primary hallucination guard for session-specific claims.
        if state.get("top_biomarkers"):
            ctx.append("\n## Grounded biomarker data (cite ONLY from this list)")
            for b in (state.get("top_biomarkers") or [])[:25]:
                protein = b.get("protein", "")
                lfc     = b.get("log2_fold_change", b.get("rescue_score", "?"))
                adjp    = b.get("adj_p_value", "?")
                ctx.append(f"  - {protein}  log2FC={lfc}  adj_p={adjp}")
            ctx.append(
                "CRITICAL: Do not mention any protein name, fold-change value, or "
                "p-value that is not listed above."
            )

        if state.get("pathways"):
            ctx.append("\n## Grounded pathway data (cite ONLY from this list)")
            for p in (state.get("pathways") or [])[:10]:
                ctx.append(
                    f"  - {p.get('pathway','')}  "
                    f"adj_p={p.get('p_adjust', p.get('adj_p','?'))}"
                )

        # Last 20 messages, with long content truncated to avoid token overflow
        history = _recent_messages(state.get("messages") or [], n=20, truncate_at=600)

        messages_for_llm = [
            {"role": "system", "content": _ANSWER_SYSTEM_PROMPT + "\n\n" + "\n".join(ctx)},
            *history,
        ]
        try:
            response = self._call_llm(messages_for_llm, max_tokens=700)
        except Exception as exc:
            self.logger.warning("Answer LLM failed: %s", exc)
            response = "Sorry, I encountered an error. Please try again."

        state["messages"].append({"role": "assistant", "content": response})
        state["intent"]       = "answer"
        state["active_agent"] = "learning_agent"
        state["status"]       = "answered"
        return state

    # ── Main entry point ──────────────────────────────────────────────────────

    @_traceable(run_type="chain", name="learning_agent",
                tags=["biomarker-discovery", "orchestrator"])
    def run(self, state: BiomarkerState) -> BiomarkerState:
        # Attach compact session metadata to the active LangSmith run so
        # every trace is filterable by session_id, data_type, etc.
        rt = _get_run_tree()
        if rt is not None:
            try:
                rt.extra.setdefault("metadata", {}).update(get_trace_metadata(state))
            except Exception:
                pass

        user_query = state.get("user_query", "")
        state["messages"].append({"role": "user", "content": user_query})

        decision = self._make_decision(state)
        action   = decision.get("action", "answer")
        state["intent"]       = action
        state["active_agent"] = "learning_agent"

        # ── Capture analysis parameter overrides from the decision ─────────────
        # Merge any non-null params from this decision into the session overrides.
        # Existing overrides are preserved so values set in earlier turns carry
        # forward until the user explicitly changes them.
        _param_keys = (
            "adj_pval_cutoff", "log2fc_cutoff", "missing_threshold", "top_n",
            "test_method",
        )
        new_params = {k: decision[k] for k in _param_keys
                      if k in decision and decision[k] is not None}
        if new_params:
            existing = dict(state.get("analysis_params") or {})
            existing.update(new_params)
            state["analysis_params"] = existing
            self.logger.info("Analysis params updated: %s", existing)

        # ── Persist state-level fields from decision ──────────────────────────
        if decision.get("is_paired") is not None:
            state["is_paired"] = bool(decision["is_paired"])
        if decision.get("all_groups"):
            state["all_groups"] = decision["all_groups"]
        if decision.get("omic_type"):
            state["omic_type"] = decision["omic_type"]

        # ── Clarification question ────────────────────────────────────────────
        if action == "ask_clarification":
            question = (
                decision.get("clarification_question")
                or decision.get("reason")
                or "Could you provide a bit more detail so I can run the most appropriate analysis for you?"
            )
            state["messages"].append({"role": "assistant", "content": question})
            state["status"] = "awaiting_clarification"
            self.logger.info("Clarification requested; awaiting user reply.")
            return state

        # ── Load data ─────────────────────────────────────────────────────────
        if action == "load_data":
            return self._specialist("ingestion").run(state)

        # ── Specific comparison (groups named by user) ─────────────────────────
        if action == "run_analysis":
            g1_label   = decision.get("group1_label")
            g1_samples = decision.get("group1_samples") or []
            g2_label   = decision.get("group2_label")
            g2_samples = decision.get("group2_samples") or []
            all_cols   = state.get("sample_columns") or []

            # Fallback: LLM gave group labels but couldn't match column names →
            # try pattern-matching the label against actual column names.
            if g1_label and not g1_samples:
                g1_samples = self._match_columns_by_label(g1_label, all_cols)
                if g1_samples:
                    self.logger.info(
                        "Pattern-matched '%s' → %s", g1_label, g1_samples
                    )
            if g2_label and not g2_samples:
                g2_samples = self._match_columns_by_label(g2_label, all_cols)
                if g2_samples:
                    self.logger.info(
                        "Pattern-matched '%s' → %s", g2_label, g2_samples
                    )

            if g1_samples and g2_samples:
                # Successfully identified both groups — update state and run
                state["group1_label"]   = g1_label or "Group1"
                state["group1_samples"] = g1_samples
                state["group2_label"]   = g2_label or "Group2"
                state["group2_samples"] = g2_samples

            elif (g1_label or g2_label) and not (
                (state.get("group1_samples") or []) and (state.get("group2_samples") or [])
            ):
                # User asked for a specific comparison but we have no group assignments.
                # Ask for clarification instead of silently running unsupervised.
                preview = all_cols[:15]
                more    = f" … (+{len(all_cols)-15} more)" if len(all_cols) > 15 else ""
                state["messages"].append({
                    "role": "assistant",
                    "content": (
                        f"I couldn't automatically match the group names you specified "
                        f"(**{g1_label}** / **{g2_label}**) to sample column names.\n\n"
                        f"Available columns: `{preview}{more}`\n\n"
                        "Please type the exact column names, e.g.:\n"
                        "*'compare Control_1, Control_2, Control_3 vs "
                        "Disease_1, Disease_2, Disease_3'*"
                    ),
                })
                state["status"] = "answered"
                return state

            # Groups resolved (either from this turn or already in state) → run analysis
            return self._specialist("biomarker").run(state)

        # ── All pairwise comparisons ───────────────────────────────────────────
        if action == "run_all_comparisons":
            return self._run_all_comparisons(state)

        # ── Enrichment ────────────────────────────────────────────────────────
        if action == "run_enrichment":
            return self._specialist("enrichment").run(state)

        # ── Visualisation ──────────────────────────────────────────────────────
        if action == "run_visualization":
            requested_plots = decision.get("requested_plots") or []
            return self._specialist("visualization").run(state, requested_plots=requested_plots or None)

        # ── Code display ──────────────────────────────────────────────────────
        if action == "show_code":
            return self._show_code(state)

        # ── Code modification ─────────────────────────────────────────────────
        if action == "modify_code":
            return self._modify_code(state)

        # ── Database query (UniProt) ───────────────────────────────────────────
        if action == "query_database":
            return self._query_database(state)

        # ── Answer (default) ──────────────────────────────────────────────────
        return self._answer(state)
