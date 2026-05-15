"""
agents/learning_agent.py
Master Orchestrator — LLM-driven reasoning over full session state.

Replaces keyword routing entirely. On every message the LearningAgent:
  1. Sends a structured context block to the LLM and receives a JSON decision
  2. Extracts comparison groups from natural language ("compare GroupA vs GroupB")
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
from typing import Any, Dict, List, Optional, Tuple

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
    "load_data", "run_analysis", "run_all_comparisons", "run_full_pipeline",
    "run_enrichment", "run_visualization",
    "show_code", "modify_code", "query_database",
    "query_data",
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
                         (e.g. "compare DMD Quad vs BL6 Quad")
  "run_all_comparisons" — run ALL pairwise group comparisons WITHOUT enrichment or viz
                         (rarely used directly — most users want run_full_pipeline)
  "run_full_pipeline"   — first-time auto-analysis: data summary + all pairwise
                         comparisons (or pooled fold-change for n=1 designs) +
                         pathway enrichment + plots, in one shot. Use this when
                         the user asks for "the full analysis", "run analysis"
                         (generic, no specific pair), "do everything", "give me
                         a comprehensive analysis", "analyse the data", etc.
                         After the pipeline runs, follow-up turns can use
                         run_analysis for specific drill-downs.
  "run_enrichment"      — run KEGG / GO pathway enrichment on current biomarker results
  "run_visualization"   — generate plots, heatmaps, charts, or a report
  "show_code"           — user wants to see the reproducible analysis code
  "modify_code"         — user wants to change, alter, or extend the analysis code
  "query_database"      — look up protein info, gene names, UniProt annotation, or convert IDs
  "query_data"          — answer cell-level / aggregation questions about the user's FILE itself.
                           Use whenever the answer requires reading specific values, cells, rows,
                           sheet structure, or counts directly from the uploaded data. Examples
                           (with generic placeholders — substitute the user's actual tokens):
                             • "what is the <metric> for <protein X> in sample <Y>?"
                             • "how many sheets does this file have?"
                             • "what are the column headers in the <sheet name> sheet?"
                             • "which proteins have a value of 0 in sample <Y>?"
                             • "is <protein X> detected in the <group> group?"
                             • "what is the molecular weight of <protein X>?"
                             • "how many contaminant proteins (CON__) are in the dataset?"
                             • "what is the largest / smallest / highest-MW protein in the file?"
                           DO NOT use for general explanations or for biomarker results from a
                           completed analysis — those go through "answer".
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
4.  "show code" / "give me the code" / "what code was used" / "show me the query" /
    "what query did you run" / "how did you get that answer" / "show the SQL" → "show_code"
    (The handler picks between the analysis script and the most recent
    data-query snippet based on the user's phrasing.)
5.  Re-run with new parameter values (see below) → "run_analysis" + fill parameter fields
6.  "change the code to use X method" / "add a step to the script" → "modify_code"
7.  "look up proteins" / "get gene names" / "annotate" / "UniProt" / "convert IDs" → "query_database"
7b. Specific values, sheet structure, cell content, accession lookups, MW, intensity counts,
    or detection-of-X-in-sample-Y questions about the uploaded FILE → "query_data"
    (Distinguish from "answer": if the question is about a concrete value in the file
    rather than a concept, definition, or analysis result, prefer "query_data".)
7c. MULTI-QUESTION MESSAGES: Note — when the user pastes 2+ questions in one message,
    the orchestrator splits them automatically and routes each question through this
    same decision step. So just answer for the SINGLE question you receive; do not
    worry about "the rest".
8.  Pathway / enrichment / KEGG / GO → "run_enrichment"
9.  Plot / visualize / chart / heatmap / volcano / report → "run_visualization"
10. Data uploaded but NO analysis yet AND user says "run analysis", "analyse the
    data", "do the analysis", "full analysis", "run all", "give me everything",
    "comprehensive analysis" — anything generic without a named group pair →
    "run_full_pipeline"
11. After a full pipeline has already run, "run analysis" with no specific pair
    repeats the pipeline (still "run_full_pipeline").
12. "run analysis" / "analyze" with SPECIFIC group names (e.g. "Disease vs Control",
    "DMD Quad vs BL6 Quad") → "run_analysis"
    - Set group1_label, group1_samples, group2_label, group2_samples from available_columns.
    - Leave sample lists empty if you cannot confidently match column names.
13. "run all pairwise comparisons WITHOUT enrichment / plots" (explicit, rare) →
    "run_all_comparisons"

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
  NOTE: Cell-level questions about the uploaded file (spectral counts, intensities, accessions,
  sheet contents, MW, "is X detected in sample Y") are handled by a separate `query_data`
  action that runs pandas against the file directly. If you receive such a question in this
  `answer` flow, it has already been routed here for a reason — you should describe what the
  session context DOES contain (sheet names, columns, row counts) and tell the user the value
  itself can be retrieved by re-asking the question explicitly.

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
  - Repeated tokens or shared substrings across columns (these usually mark a
    biological condition — accept whatever tokens the data actually contains;
    don't assume the names of strains, treatments, or tissues)
  - Numeric suffixes indicating replicates (_1, _2, _3 or .1 .2 .3 or trailing digits)
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


_QUESTION_TAG_RE = re.compile(
    r"\s*(?:Counts?\s*&\s*values?|Zeros?\s*&\s*nulls?|Edge\s*cases?|Metadata)\s*[⌄▾▼v]?\s*$",
    re.IGNORECASE,
)
_LIST_PREFIX_RE = re.compile(r"^\s*[\d]+[\.\)]\s*|^\s*[•▪●\-\*]\s*")


def _extract_questions(text: str) -> List[str]:
    """
    Split a (potentially long) user message into individual questions.

    Heuristics:
    - Split on newlines first.
    - Strip leading list markers ("1.", "•", "-", "*", "2)") and trailing
      categorisation tags ("Counts & values ⌄") that came from a UI.
    - A line is treated as a question if it ends with "?" (or "？").
    - If newline-split yields nothing useful, fall back to splitting the
      whole text on "?" so a paragraph like
        "What is X? What is Y? How many Z?"
      is broken into three.

    Returns the cleaned question strings; empty list when none are found.
    """
    if not text:
        return []

    questions: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = _LIST_PREFIX_RE.sub("", line)
        line = _QUESTION_TAG_RE.sub("", line)
        line = line.strip()
        if not line:
            continue
        if line.endswith("?") or line.endswith("？"):
            questions.append(line)

    # Fallback: paragraph with multiple "?" on a single line
    if len(questions) < 2 and text.count("?") >= 2:
        parts = [p.strip() for p in re.split(r"(?<=\?)\s+", text) if p.strip()]
        questions = [_QUESTION_TAG_RE.sub("", p).strip()
                     for p in parts if p.endswith("?")]

    return questions


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

        # ── Column-friendly-label mapping (for the decision LLM) ───────────
        # Without this, the decision LLM sees only raw column names ("SpC A",
        # "SpC B", …) and routes any question mentioning a friendly group name
        # ("DMD Soleus") to ask_clarification — even though the mapping IS
        # known. With it, the LLM can confidently route to query_data.
        column_groups_dec = state.get("column_group_labels") or {}
        if column_groups_dec:
            # Build a compact reverse map for the decision context
            reverse_dec: Dict[str, List[str]] = {}
            for col, label in column_groups_dec.items():
                reverse_dec.setdefault(label, []).append(col)
            ctx += "  column_group_labels (friendly_name → real_column(s)):\n"
            for label, cols in reverse_dec.items():
                ctx += f"    '{label}' → {cols}\n"
            ctx += (
                "  → IMPORTANT: questions that mention any of the friendly "
                "names above ARE answerable via query_data. Do NOT ask the "
                "user to clarify how friendly names map to columns — the "
                "mapping is shown above.\n"
            )

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
        """Ask LLM to detect group structure from column naming patterns.

        Mirrors IngestionAgent's sanitiser: drops any 1-sample "group" the LLM
        produces (a group requires ≥ 2 samples to be statistically meaningful)
        and rejects degenerate inferences entirely.
        """
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
                cleaned = self._sanitize_groups(groups, sample_columns)
                self.logger.info("Inferred groups (cleaned): %s",
                                 {k: len(v) for k, v in cleaned.items()})
                return cleaned
        except Exception as exc:
            self.logger.warning("Group inference failed: %s", exc)
        return {}

    @staticmethod
    def _sanitize_groups(
        groups: Dict[str, List[str]],
        sample_columns: List[str],
    ) -> Dict[str, List[str]]:
        """Drop single-sample groups; require ≥ 2 surviving groups total."""
        valid_cols = set(sample_columns)
        cleaned: Dict[str, List[str]] = {}
        for name, samples in groups.items():
            if not isinstance(samples, list):
                continue
            real = [c for c in samples if c in valid_cols]
            if len(real) >= 2:
                cleaned[str(name)] = real
        return cleaned if len(cleaned) >= 2 else {}

    # ── Full auto-pipeline ────────────────────────────────────────────────────

    @_traceable(run_type="chain", name="orchestrator.run_full_pipeline",
                tags=["biomarker-discovery", "full_pipeline"])
    def _run_full_pipeline(self, state: BiomarkerState) -> BiomarkerState:
        """End-to-end first-time analysis:
              1. Emit a concise data summary so the user sees what's loaded.
              2. Run all pairwise comparisons (Welch / limma when replicated,
                 pooled log₂FC when n=1 per group — handled by
                 ``_run_all_comparisons`` → BiomarkerAgent).
              3. Pathway enrichment on the top biomarkers if any survived.
              4. Visualisation suite (volcano / heatmap / PCA / etc.).
              5. Final closing message inviting drill-downs.
        Subsequent turns can run a single ``run_analysis`` for specific
        pairs without re-doing this whole pipeline.
        """
        # ── 1. Data summary ──────────────────────────────────────────────────
        summary = self._build_data_summary(state)
        if summary:
            state["messages"].append({"role": "assistant", "content": summary})

        # ── 2. Analyses (this populates top_biomarkers / pathways path-of-data) ──
        state = self._run_all_comparisons(state)

        has_biomarkers = bool(
            state.get("top_biomarkers") or state.get("top_proteins")
        )

        # ── 3. Enrichment (only when we have a biomarker list to query) ──────
        if has_biomarkers:
            try:
                state = self._specialist("enrichment").run(state)
            except Exception as exc:
                self.logger.warning("Pipeline enrichment step failed: %s", exc)
                state["messages"].append({
                    "role": "assistant",
                    "content": f"⚠ Pathway enrichment skipped: {exc}",
                })
        else:
            state["messages"].append({
                "role": "assistant",
                "content": (
                    "_Pathway enrichment skipped — no significant biomarkers "
                    "were produced upstream._"
                ),
            })

        # ── 4. Visualisation ─────────────────────────────────────────────────
        if has_biomarkers or state.get("plot_paths"):
            try:
                state = self._specialist("visualization").run(state)
            except Exception as exc:
                self.logger.warning("Pipeline viz step failed: %s", exc)
                state["messages"].append({
                    "role": "assistant",
                    "content": f"⚠ Visualisation skipped: {exc}",
                })

        # ── 5. Drill-down invite ─────────────────────────────────────────────
        state["messages"].append({
            "role": "assistant",
            "content": (
                "**Full analysis complete.** You can now ask follow-up "
                "questions like:\n"
                "- *Compare DMD Quad vs BL6 Quad in detail*\n"
                "- *Show the top biomarkers for the Heart tissue group*\n"
                "- *Why is pathway X enriched?*\n"
                "- *Show me the volcano plot*"
            ),
        })
        state["intent"]       = "run_full_pipeline"
        state["active_agent"] = "learning_agent"
        state["status"]       = "pipeline_complete"
        return state

    def _build_data_summary(self, state: BiomarkerState) -> str:
        """Render a compact markdown recap of the loaded dataset for the
        opening message of the full pipeline. Pulls everything from state
        — no hardcoded values."""
        n_proteins = state.get("n_proteins") or "?"
        n_samples  = state.get("n_samples")  or "?"
        lines = [f"## Dataset Summary\n",
                 f"- **{n_proteins}** proteins · **{n_samples}** samples"]

        details: List[str] = []
        if state.get("data_type"):       details.append(f"`{state['data_type']}`")
        if state.get("software"):        details.append(f"detected as **{state['software']}**")
        if state.get("organism"):        details.append(f"organism: **{state['organism']}**")
        if state.get("disease_program"): details.append(f"program: **{state['disease_program']}**")
        if details:
            lines.append("- " + " · ".join(details))

        # Groups detected by IngestionAgent
        sample_map = state.get("sample_map") or {}
        if sample_map:
            lines.append(f"- **{len(sample_map)}** pooled samples mapped: "
                         + ", ".join(f"`{k}`→{v.get('client_id') or v.get('strain') or '?'}"
                                      for k, v in list(sample_map.items())[:6]))
        all_groups = state.get("all_groups") or {}
        if all_groups:
            lines.append(f"- **{len(all_groups)}** inferred group(s): "
                         + ", ".join(f"`{k}` ({len(v)})"
                                      for k, v in list(all_groups.items())[:8]))

        # Files
        if state.get("file_id"):
            lines.append(f"- File: `{state.get('raw_data_path') or state.get('data_path') or ''}`")

        lines.append("\n_Running data summary → all comparisons → "
                     "pathway enrichment → plots…_")
        return "\n".join(lines)

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

        # Standard replicated proteomics — prefer the sanitised group map built
        # at ingestion time; only re-infer if it's missing.
        sample_cols = state.get("sample_columns") or []
        groups: Dict[str, List[str]] = state.get("all_groups") or {}
        if not groups:
            groups = self._infer_groups(sample_cols)

        # ── n=1 design fallback ───────────────────────────────────────────────
        # If every plausible group has just one sample (e.g. {strain × tissue}
        # with one mouse per cell), Welch / limma are undefined. Route to the
        # pooled-design skill which computes pairwise log₂FC + rescue scores
        # without requiring replicates.
        all_singleton = (
            bool(sample_cols)
            and (not groups or all(len(v) <= 1 for v in groups.values()))
        )
        if all_singleton:
            n = len(sample_cols)
            label_map = {c: c for c in sample_cols}  # each column is its own condition
            state["label_map"]        = label_map
            state["is_pooled_design"] = True
            state["omic_type"]        = "proteomics_pooled"
            state["messages"].append({
                "role": "assistant",
                "content": (
                    f"Each group in your data has only **n=1** sample — Welch t-tests "
                    f"need replicates and would return NaN for every protein. "
                    f"Switching to **pooled-design log₂ fold-change analysis** across "
                    f"all **{n} samples** ({n * (n-1) // 2} pairwise contrasts). "
                    f"Proteins are ranked by a rescue score that captures "
                    f"consistent up/down-regulation across contrasts."
                ),
            })
            return self._specialist("biomarker").run(state)

        if len(groups) < 2:
            state["messages"].append({
                "role": "assistant",
                "content": (
                    "I couldn't automatically detect biological groups from your column "
                    "names. Please tell me which groups to compare using the EXACT "
                    "column names, e.g. *'compare A_1, A_2, A_3 vs B_1, B_2, B_3'*."
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
        """
        Surface stored code on demand. Two kinds of code may be available:
          - state['analysis_code']    — full reproducible analysis script
          - state['last_query_code']  — the SQL or pandas snippet that
                                        answered the user's most recent
                                        data question.

        We pick which to show based on the recency: prefer the data-query
        code when it's the most recently set (typical case: user just got
        a data answer and asks "show me the code"). Otherwise show the
        full analysis script.
        """
        analysis_code = state.get("analysis_code")
        query_code    = state.get("last_query_code")
        query_engine  = (state.get("last_query_engine") or "sql").lower()
        user_query    = (state.get("user_query") or "").lower()

        # Heuristic: words like "query", "this", "that" → likely about the
        # last data answer, not the full analysis script.
        prefers_query = any(
            w in user_query
            for w in ("query", "that answer", "the answer", "how did you", "how do you", "this result")
        )

        if query_code and (prefers_query or not analysis_code):
            lang = "sql" if query_engine == "sql" else "python"
            state["messages"].append({
                "role": "assistant",
                "content": (
                    f"Here's the {query_engine.upper()} query I used to answer that:\n\n"
                    f"```{lang}\n" + query_code + "\n```"
                ),
            })
        elif analysis_code:
            state["messages"].append({
                "role": "assistant",
                "content": (
                    "Here is the reproducible Python script used for this analysis:\n\n"
                    "```python\n" + analysis_code + "\n```\n\n"
                    "_You can run this script directly or ask me to modify it._"
                ),
            })
        else:
            state["messages"].append({
                "role": "assistant",
                "content": (
                    "No code to show yet. Run an analysis (for the reproducible "
                    "Python script) or ask a data question (for the SQL/pandas "
                    "query I used)."
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
        # ── Generate → Review loop ────────────────────────────────────────────
        # Inspired by GenoMAS: the modified script gets reviewed by a separate
        # critic LLM before being returned to the user. If the reviewer flags
        # issues, we feed the critique back and ask for one revision.
        from agents.code_reviewer import CodeReviewerAgent
        if not hasattr(self, "_code_reviewer") or self._code_reviewer is None:
            self._code_reviewer = CodeReviewerAgent()

        def _generate_modification(extra_instruction: Optional[str]) -> str:
            full_prompt = prompt
            if extra_instruction:
                full_prompt += f"\n\nREVISION GUIDANCE FROM REVIEWER:\n{extra_instruction}\n"
            raw = self._call_llm(
                [{"role": "system", "content": "You are an expert Python bioinformatician."},
                 {"role": "user",   "content": full_prompt}],
                max_tokens=2000, temperature=0.0,
            ).strip()
            raw = re.sub(r"^```(?:python)?\s*", "", raw, flags=re.MULTILINE)
            return re.sub(r"\s*```$", "", raw, flags=re.MULTILINE).strip()

        last_review = None
        modified: str = ""
        try:
            for attempt in range(2):
                extra = None
                if last_review and not last_review.approved:
                    extra = (
                        f"Issues: {last_review.issues}\n"
                        f"Suggestion: {last_review.suggestion}"
                    )
                modified = _generate_modification(extra)

                last_review = self._code_reviewer.review(
                    user_question  = f"Modify the analysis: {user_request}",
                    schema_context = "(modified Python analysis script)",
                    candidate_code = modified,
                )
                if last_review.approved or last_review.severity == "minor":
                    break

            state["analysis_code"] = modified
            review_note = ""
            if last_review and last_review.issues and not last_review.approved:
                review_note = (
                    "\n\n_⚠ Code reviewer flagged: "
                    + "; ".join(last_review.issues[:3]) + "_"
                )
            state["messages"].append({
                "role": "assistant",
                "content": (
                    "Here is the modified analysis script:\n\n"
                    "```python\n" + modified + "\n```" + review_note
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

    # ── Raw-data query (LLM-generated pandas) ─────────────────────────────────

    @_traceable(run_type="chain", name="orchestrator.query_data",
                tags=["biomarker-discovery", "query_data"])
    def _query_data(self, state: BiomarkerState) -> BiomarkerState:
        """
        Answer cell-level / aggregation questions by executing LLM-generated
        pandas code against the user's uploaded sheets.
        """
        import pandas as pd
        import numpy as np
        from core.safe_exec import safe_exec, UnsafeCodeError, CodeTimeoutError

        user_query = state.get("user_query", "") or ""
        data_path  = state.get("data_path") or state.get("raw_data_path")

        if not data_path:
            state["messages"].append({
                "role": "assistant",
                "content": (
                    "I don't have a data file to query yet. Please upload your "
                    "proteomics CSV or Excel file first, then ask the question again."
                ),
            })
            state["status"] = "answered"
            return state

        sheets = self._load_sheets_for_query(state, data_path)
        if not sheets:
            state["messages"].append({
                "role": "assistant",
                "content": (
                    f"I couldn't read the data file at `{data_path}` to look up that "
                    "value. The file may be corrupted or in an unexpected format."
                ),
            })
            state["status"] = "answered"
            return state

        # Pick primary sheet (largest by row count, falling back to first)
        df_candidates = [(n, s) for n, s in sheets.items() if isinstance(s, pd.DataFrame)]
        if not df_candidates:
            state["messages"].append({
                "role": "assistant",
                "content": "The file is loaded but contains no readable sheets.",
            })
            state["status"] = "answered"
            return state
        primary_name, primary_df = max(df_candidates, key=lambda kv: kv[1].shape[0])

        # ── Ensure DuckDB tables are registered for this session ─────────────
        # IngestionAgent registers on upload, but after a server restart the
        # DuckDB connection is gone while the JSON checkpoint survives —
        # re-register here on first query so the SQL path always works.
        try:
            from core import data_store as _ds
            if _ds.is_available():
                if _ds.get_store(state.get("session_id", "")) is None:
                    _ds.register_sheets(state.get("session_id", ""), sheets)
        except Exception as exc:
            self.logger.debug("DuckDB lazy-register failed: %s", exc)

        # ── Tool-calling path (ExcelWorker pattern) ───────────────────────────
        # OpenAI function-call loop: the LLM picks tools and emits structured
        # arguments instead of raw SQL/pandas. We dispatch by tool name.
        # Falls through to the SQL-first path (and pandas review-revise) on
        # any failure so we never lose the previously working behavior.
        tools_record = self._query_data_via_tools(
            state, user_query, df_candidates, primary_df, primary_name,
        )
        if tools_record is not None and tools_record["ok"]:
            self._render_tool_result(state, tools_record, user_query)
            return state

        # ── SQL-first attempt via DuckDB ──────────────────────────────────────
        # SQL handles spaces / mixed-case column names cleanly, joins across
        # sheets naturally, and gives the LLM a stable schema to ground on.
        # We try SQL first; if it fails or produces an empty result the
        # downstream pandas fallback still runs.
        sql_record = self._try_sql_query(state, user_query, df_candidates)
        if sql_record is not None and sql_record.ok:
            self._render_sql_result(state, sql_record, user_query)
            return state

        # ── Build sheet-context block for code-generation prompt ──────────────
        sheet_blocks: List[str] = []
        for name, sheet in df_candidates:
            cols     = list(sheet.columns)
            preview  = sheet.head(3).to_csv(index=False)
            if len(preview) > 800:
                preview = preview[:800] + "\n…[truncated]"
            sheet_blocks.append(
                f"### Sheet: {name!r}\n"
                f"Shape: {sheet.shape[0]} rows × {sheet.shape[1]} cols\n"
                f"Columns: {cols[:40]}{' …(+more)' if len(cols) > 40 else ''}\n"
                f"First 3 rows:\n{preview}"
            )
        sheets_ctx = "\n\n".join(sheet_blocks)

        # ── Friendly-name → real-column mapping block ───────────────────────
        column_groups_pd = state.get("column_group_labels") or {}
        if column_groups_pd:
            reverse_pd = {}
            for col, label in column_groups_pd.items():
                reverse_pd.setdefault(label, []).append(col)
            rev_lines_pd = "\n".join(
                f"  '{label}'  →  use column(s) {cols}"
                for label, cols in reverse_pd.items()
            )
            cg_pd_block = (
                "\n══════════════════════════════════════════════════════════\n"
                "GROUP-LABEL → REAL-COLUMN MAPPING (AUTHORITATIVE — DO NOT GUESS)\n"
                "══════════════════════════════════════════════════════════\n"
                "The workbook stores 'friendly' group labels in a separate row\n"
                "from the real column names. When the user says 'DMD Soleus',\n"
                "use the EXACT column on the right of the mapping below.\n"
                "NEVER make up your own pairing.\n\n"
                f"{rev_lines_pd}\n"
                "══════════════════════════════════════════════════════════\n"
            )
        else:
            cg_pd_block = ""

        base_prompt = (
            "You write safe pandas code to answer a question about a proteomics dataset.\n"
            f"{cg_pd_block}\n"
            "VARIABLES ALREADY DEFINED IN YOUR EXECUTION NAMESPACE:\n"
            f"- `df`           — primary dataframe (sheet {primary_name!r}, all columns)\n"
            "- `df_spc`        — identifier columns + spectral-count columns ONLY\n"
            "- `df_intensity`  — identifier columns + intensity columns ONLY\n"
            "- `sheets`        — dict mapping sheet name → DataFrame for ALL sheets\n"
            "- `sample_map`    — dict mapping sample code → {client_id, strain,\n"
            "                    treatment, mouse_id, …} (built from the identifier sheet)\n"
            "- `pd`            — pandas\n"
            "- `np`            — numpy\n"
            "\n"
            "DETERMINISTIC HELPER FUNCTIONS YOU MUST USE WHEN APPROPRIATE:\n"
            "- `safe_fold_change(num, denom, sample_num='?', sample_den='?')`\n"
            "    Returns float OR a clear 'undefined — …' string for /0 cases.\n"
            "- `get_gene_symbol(protein_name)` — returns the value of `GN=…`\n"
            "    in a UniProt-style description, else 'Unknown'.\n"
            "- `get_short_name(protein_name)`  — bare description before ' OS='\n"
            "- `format_protein_row(name, accession, value, unit='')`\n"
            "    Returns a 'GeneSymbol (Accession) — Value Unit' string for any\n"
            "    protein row, regardless of which gene / accession it carries.\n"
            "- `get_nonstandard_protein(df, accession_or_name, metric='spc')`\n"
            "    Look up a protein robustly; returns dict of per-sample values.\n"
            "- `top_n_by_metric(df, metric_col, n=10)`\n"
            "    Sort by a metric column, keeping identifier columns.\n"
            "- `detect_metric_columns(df)` → {'identifier', 'spc', 'intensity', …}\n"
            "\n"
            "SHEETS DESCRIPTION:\n"
            f"{sheets_ctx}\n\n"
            f"USER QUESTION: {user_query!r}\n\n"
            "STRICT RULES — read carefully:\n"
            "\n"
            "[OUTPUT FORMAT]\n"
            "- Use ONLY pandas/numpy on these in-memory dataframes.\n"
            "- NO file I/O, NO network, NO os/sys/subprocess imports, NO open(), NO to_csv/to_excel.\n"
            "- Assign your final answer to a variable named `answer`.\n"
            "- `answer` may be: a scalar, list, dict, Series, or DataFrame.\n"
            "- Keep code under 20 lines. No markdown fences, no prose.\n"
            "\n"
            "[METRIC-TYPE DISCIPLINE — never mix metrics]\n"
            "Use `df_spc` for spectral-count questions and `df_intensity` for\n"
            "intensity questions. NEVER query `df` directly for these — the\n"
            "pre-split dataframes guarantee you can't accidentally sum SpC with\n"
            "Intensity columns.\n"
            "\n"
            "[IDENTIFIERS — always return them with values]\n"
            "When returning protein rows, use `format_protein_row(name, acc, val)`\n"
            "or include the protein name + accession columns directly. Never\n"
            "return a bare accession or a bare metric value.\n"
            "\n"
            "[SAMPLE-METADATA LOOKUPS — use sample_map first]\n"
            "When asked what a sample corresponds to (strain, treatment, client ID,\n"
            "group), look it up in `sample_map[code]` first — that dict was built\n"
            "from the identifier sheet, filtered to MaxQuant codes only. Return\n"
            "all fields of the entry, not just one.\n"
            "\n"
            "[PRESENCE / DETECTION — return the value, not a boolean]\n"
            "For 'is X detected in sample Y' style questions, return the actual\n"
            "SpC / Intensity value. The NL layer will phrase yes/no.\n"
            "\n"
            "[SAFE DIVISION — always use safe_fold_change for fold changes]\n"
            "Call `safe_fold_change(numer, denom, sample_num=..., sample_den=...)`.\n"
            "It returns a float when both samples have signal and a clear\n"
            "'undefined — protein absent in sample X' string when either is zero.\n"
            "NEVER do raw division on SpC / Intensity values.\n"
            "\n"
            "[NON-STANDARD ENTRIES — use the dedicated helper]\n"
            "For proteins with unusual identifiers (no GN= field, no UniProt-style\n"
            "accession, contaminants), call `get_nonstandard_protein(df_spc, name)`\n"
            "or `get_nonstandard_protein(df_intensity, name, metric='intensity')`.\n"
            "\n"
            "[CONTAMINANT FLAG — filter or count, never silently include]\n"
            "If the sheet has an `is_contaminant` column (added by the MaxQuant\n"
            "cleanup pipeline), filter `df = df[~df['is_contaminant']]` for\n"
            "biological analyses. For questions like 'how many CON__ proteins?'\n"
            "DO query `df['is_contaminant'].sum()` directly — the flag is there\n"
            "specifically to make contaminants countable without inspecting\n"
            "accession prefixes manually.\n"
            "\n"
            "[ROBUSTNESS]\n"
            "- Case-insensitive partial matching for protein/gene/accession lookups.\n"
            "- Try multiple candidate identifier columns if uncertain.\n"
            "- Wrap risky lookups in try/except and set the value to 'NOT FOUND' on error."
        )

        # ── Review-revise loop (GenoMAS-style) ────────────────────────────────
        # Generator writes code, Reviewer critiques, Executor runs in sandbox.
        # On reviewer rejection OR runtime error, the generator gets the
        # feedback and produces a corrected version. Up to 2 revision rounds.
        from agents.code_reviewer import (
            CodeReviewerAgent, ExecutionRecord, review_and_revise,
        )
        if not hasattr(self, "_code_reviewer") or self._code_reviewer is None:
            self._code_reviewer = CodeReviewerAgent()

        def _generate(extra_instruction: Optional[str]) -> str:
            prompt = base_prompt
            if extra_instruction:
                prompt += f"\n\nREVISION GUIDANCE:\n{extra_instruction}\n"
            raw = self._call_llm(
                [{"role": "system", "content": "You generate safe pandas code."},
                 {"role": "user",   "content": prompt}],
                max_tokens=600, temperature=0.0,
            )
            return re.sub(r"^```(?:python)?\s*|\s*```$", "", raw.strip(),
                          flags=re.MULTILINE).strip()

        # ── Pre-split the primary protein sheet into SpC vs Intensity (BUG-1) ──
        # The LLM never has to choose between mixed metric columns: df_spc only
        # contains identifier + SpC columns, df_intensity only intensity.
        from core import proteomics_tools as _pt
        df_spc, df_intensity = _pt.split_spc_intensity(primary_df)

        def _execute(code_str: str):
            namespace = {
                "df":           primary_df,
                "df_spc":       df_spc,
                "df_intensity": df_intensity,
                "sheets":       {n: s for n, s in df_candidates},
                "sample_map":   state.get("sample_map") or {},
                "pd":           pd,
                "np":           np,
                # Deterministic helpers (BUG 2, 4, 6 fixes)
                "safe_fold_change":       _pt.safe_fold_change,
                "get_gene_symbol":        _pt.get_gene_symbol,
                "get_short_name":         _pt.get_short_name,
                "format_protein_row":     _pt.format_protein_row,
                "get_nonstandard_protein": _pt.get_nonstandard_protein,
                "top_n_by_metric":        _pt.top_n_by_metric,
                "detect_metric_columns":  _pt.detect_metric_columns,
            }
            try:
                safe_exec(code_str, namespace, timeout=15)
                return namespace.get("answer", "(no `answer` variable set)"), None
            except UnsafeCodeError as exc:
                return None, f"Unsafe code rejected by sandbox: {exc}"
            except CodeTimeoutError as exc:
                return None, str(exc)
            except Exception as exc:
                return None, f"{type(exc).__name__}: {exc}"

        try:
            record: ExecutionRecord = review_and_revise(
                generator      = _generate,
                executor       = _execute,
                reviewer       = self._code_reviewer,
                user_question  = user_query,
                schema_context = sheets_ctx,
                max_rounds     = 2,
            )
        except Exception as exc:
            self.logger.warning("Review-revise loop failed: %s", exc)
            state["messages"].append({
                "role": "assistant",
                "content": "I couldn't generate a query for that — please rephrase.",
            })
            state["status"] = "answered"
            return state

        code       = record.code
        result     = record.result
        exec_error = record.error
        self.logger.info(
            "query_data review-revise: rounds=%d, ok=%s, final_error=%s",
            record.rounds_used, record.ok, exec_error,
        )

        # ── Format result for the user ────────────────────────────────────────
        if exec_error:
            formatted_result = f"(query failed: {exec_error})"
        elif isinstance(result, pd.DataFrame):
            # CSV avoids the tabulate dependency to_markdown requires
            formatted_result = result.head(30).to_csv(index=False)
        elif isinstance(result, pd.Series):
            formatted_result = result.head(50).to_string()
        elif isinstance(result, (list, tuple)):
            preview = list(result)[:60]
            formatted_result = "\n".join(f"- {x}" for x in preview)
            if len(result) > 60:
                formatted_result += f"\n…(+{len(result) - 60} more)"
        elif isinstance(result, dict):
            # Q&A dict from multi-question mode renders as a markdown list
            keys = list(result.keys())[:60]
            lines = []
            for k in keys:
                v = result[k]
                # Render lists / Series inline up to 8 items
                if isinstance(v, (list, tuple)):
                    v_str = ", ".join(str(x) for x in list(v)[:8])
                    if len(v) > 8:
                        v_str += f" …(+{len(v)-8} more)"
                elif isinstance(v, pd.Series):
                    v_str = ", ".join(str(x) for x in v.head(8).tolist())
                else:
                    v_str = str(v)
                # Trim very long single values
                if len(v_str) > 220:
                    v_str = v_str[:220] + "…"
                lines.append(f"- **{k}**: {v_str}")
            formatted_result = "\n".join(lines)
            if len(result) > 60:
                formatted_result += f"\n…(+{len(result) - 60} more keys)"
        else:
            formatted_result = str(result)

        # Cap the formatted-result size to avoid token blow-up.
        if len(formatted_result) > 6000:
            formatted_result = formatted_result[:6000] + "\n…[truncated]"

        # ── Ask LLM to write a clear natural-language answer ──────────────────
        nl_prompt = (
            f"The user asked: {user_query!r}\n\n"
            f"I ran a pandas query against their data and got this result:\n"
            f"```\n{formatted_result}\n```\n\n"
            "Write a clear, concise natural-language answer (2–5 sentences). "
            "Quote the exact numbers/values from the result. If the result is a "
            "table, summarise it. If it's empty or failed, say so plainly."
        )
        nl_max_tokens = 500
        try:
            nl_response = self._call_llm(
                [{"role": "system", "content":
                    "You translate pandas query results into clear answers. "
                    "Do not fabricate values that aren't in the result."},
                 {"role": "user", "content": nl_prompt}],
                max_tokens=nl_max_tokens,
            )
        except Exception as exc:
            self.logger.warning("Data-query NL response LLM failed: %s", exc)
            nl_response = f"Result:\n\n{formatted_result}"

        # Store the pandas query quietly — surfaced only when the user asks.
        state["last_query_code"]   = code
        state["last_query_engine"] = "pandas"
        state["messages"].append({"role": "assistant", "content": nl_response.strip()})
        state["intent"]       = "query_data"
        state["active_agent"] = "learning_agent"
        state["status"]       = "answered"
        return state

    # ── Multi-question split + per-question routing ───────────────────────────

    @_traceable(run_type="chain", name="orchestrator.multi_question",
                tags=["biomarker-discovery", "multi_question"])
    def _handle_multi_question(
        self, state: BiomarkerState, questions: List[str],
    ) -> BiomarkerState:
        """
        When the user pastes multiple questions in one message, split them and
        route each through the normal decision flow. Each question gets its own
        routing decision (query_data / answer / query_database / show_code etc.)
        and its own response. Responses are combined into a single numbered
        markdown block.
        """
        n = len(questions)
        parts: List[str] = [f"You asked **{n} questions** — answering each below.\n"]

        for i, q in enumerate(questions, 1):
            self.logger.info("Multi-q [%d/%d]: %s", i, n, q[:80])

            # Build a sub-state for this single question only. Critical:
            # `messages` is reset so per-question handlers don't append to the
            # main thread; we capture each sub-response separately.
            sub_state: BiomarkerState = {**state, "user_query": q, "messages": []}

            try:
                sub_decision = self._make_decision(sub_state)
                sub_action   = sub_decision.get("action", "answer")
                sub_state["intent"]       = sub_action
                sub_state["active_agent"] = "learning_agent"

                if sub_action == "query_data":
                    sub_state = self._query_data(sub_state)
                elif sub_action == "query_database":
                    sub_state = self._query_database(sub_state)
                elif sub_action == "show_code":
                    sub_state = self._show_code(sub_state)
                elif sub_action == "modify_code":
                    sub_state = self._modify_code(sub_state)
                else:
                    # answer, ask_clarification, run_*, load_data — all fall through
                    # to the conversational answer path for sub-questions, since we
                    # don't want side-effects like re-running analysis 30 times.
                    sub_state = self._answer(sub_state)

                last = next(
                    (m["content"] for m in reversed(sub_state.get("messages") or [])
                     if m.get("role") == "assistant"),
                    "_(no response generated)_",
                )
            except Exception as exc:
                self.logger.warning("Multi-q sub-question %d failed: %s", i, exc)
                last = f"_(internal error: {exc})_"

            parts.append(f"### {i}. {q}\n\n{last}\n")

        combined = "\n".join(parts)
        state["messages"].append({"role": "assistant", "content": combined})
        state["intent"]       = "multi_question"
        state["active_agent"] = "learning_agent"
        state["status"]       = "answered"
        return state

    # ── Tool-calling query path (ExcelWorker pattern) ─────────────────────────

    @_traceable(run_type="chain", name="orchestrator.query_data.tools",
                tags=["biomarker-discovery", "query_data", "tools"])
    def _query_data_via_tools(
        self,
        state:         BiomarkerState,
        user_query:    str,
        df_candidates: List[Tuple[str, Any]],
        primary_df:    Any,
        primary_name:  str,
        max_iterations: int = 7,
    ):
        """
        Drive the OpenAI tool-call loop. The LLM picks among:
          - load_preview_data
          - complex_duckdb_query
          - simple_dataframe_query
        and emits structured arguments. We dispatch, append a tool message,
        loop until the model returns a final text message (no more tool_calls)
        or hits `max_iterations`.

        Returns a dict ``{ok, final_text, tool_calls_history, error}`` or
        ``None`` when the path can't run (no tools available).
        """
        try:
            from core import data_store as _ds
            from core import llm_tools as _lt
            from core import proteomics_tools as _pt
        except ImportError:
            return None
        if not _ds.is_available():
            return None
        session_id = state.get("session_id", "")
        if _ds.get_store(session_id) is None:
            return None

        # ── Build the pandas execution namespace once and reuse it ────────────
        import pandas as pd
        import numpy as np
        df_spc, df_intensity = _pt.split_spc_intensity(primary_df)
        pandas_namespace: Dict[str, Any] = {
            "df":           primary_df,
            "df_spc":       df_spc,
            "df_intensity": df_intensity,
            "sheets":       {n: s for n, s in df_candidates},
            "sample_map":   state.get("sample_map") or {},
            "pd":           pd,
            "np":           np,
            "safe_fold_change":       _pt.safe_fold_change,
            "get_gene_symbol":        _pt.get_gene_symbol,
            "get_short_name":         _pt.get_short_name,
            "format_protein_row":     _pt.format_protein_row,
            "get_nonstandard_protein": _pt.get_nonstandard_protein,
            "top_n_by_metric":        _pt.top_n_by_metric,
            "detect_metric_columns":  _pt.detect_metric_columns,
        }

        # ── Column-friendly-label mapping (authoritative ground truth) ───────
        column_groups_tools = state.get("column_group_labels") or {}
        if column_groups_tools:
            reverse_tools: Dict[str, List[str]] = {}
            for col, label in column_groups_tools.items():
                reverse_tools.setdefault(label, []).append(col)
            cg_tools_block = (
                "\n══════════════════════════════════════════════════════════\n"
                "GROUP-LABEL → REAL-COLUMN MAPPING (AUTHORITATIVE — DO NOT GUESS)\n"
                "══════════════════════════════════════════════════════════\n"
                "The workbook stores friendly group labels (e.g. 'DMD Soleus')\n"
                "in a separate row from the real DuckDB column names (e.g.\n"
                "'SpC J'). When the user mentions a friendly name in their\n"
                "question, use the EXACT column on the right of this table.\n"
                "DO NOT make up your own pairing. DO NOT trust alphabetical\n"
                "order — the labels are not ordered alphabetically.\n\n"
                + "\n".join(f"  '{label}'  →  columns {cols}"
                            for label, cols in reverse_tools.items())
                + "\n══════════════════════════════════════════════════════════\n"
            )
        else:
            cg_tools_block = ""

        # ── System prompt for the tool-calling LLM ────────────────────────────
        system_prompt = (
            "You answer questions about a proteomics workbook by calling the "
            "provided tools. Workflow:\n"
            "  1. ALWAYS call `load_preview_data` first to see the exact table\n"
            "     and column names available — never guess column names.\n"
            "  2. For most data questions, call `complex_duckdb_query` with a\n"
            "     SQL query. ALWAYS double-quote identifiers containing spaces\n"
            "     or mixed case. Use ILIKE for partial text matches. Include\n"
            "     identifier columns (protein name, gene, accession) in every\n"
            "     SELECT that returns protein rows.\n"
            "  3. For operations awkward in SQL (custom regex, multi-step\n"
            "     transforms, calls to safe_fold_change / get_gene_symbol /\n"
            "     format_protein_row / get_nonstandard_protein), call\n"
            "     `simple_dataframe_query` with a pandas snippet that assigns\n"
            "     to `answer`. Use `df_spc` for SpC questions and\n"
            "     `df_intensity` for intensity questions — never mix.\n"
            "  4. If a sheet has an `is_contaminant` boolean column, filter\n"
            "     `WHERE NOT is_contaminant` for biology questions but use\n"
            "     it directly for 'how many contaminants?' counts.\n"
            "  5. For fold changes, use safe_fold_change (handles /0).\n"
            "  6. For protein-name / gene lookups, search the PROTEIN NAME\n"
            "     column (the long UniProt-style description), NOT the\n"
            "     Accession Number column. Gene symbols appear in the\n"
            "     description after 'GN='. Example: ILIKE '%dystrophin%' or\n"
            "     ILIKE '%GN=Dmd %'.\n"
            "  7. After you have the answer, return a clear natural-language\n"
            "     response WITHOUT calling another tool. Quote exact values\n"
            "     from the tool result; never fabricate.\n"
            f"{cg_tools_block}\n"
            f"Active session: {session_id[:8]}.\n"
            f"Primary table name in DuckDB: {_ds.get_store(session_id).table_names.get(primary_name, primary_name)}.\n"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_query},
        ]
        tool_specs = _lt.get_openai_tool_specs()
        history: List[Dict[str, Any]] = []
        final_text: Optional[str] = None
        error: Optional[str] = None

        for it in range(max_iterations):
            try:
                resp = self._call_llm_with_tools(
                    messages, tools=tool_specs,
                    max_tokens=900, temperature=0.0,
                )
            except Exception as exc:
                self.logger.warning("Tool-call LLM step %d failed: %s", it, exc)
                error = f"LLM tool-call error: {exc}"
                break

            tool_calls = resp.get("tool_calls")
            if not tool_calls:
                # Final text answer — exit the loop
                final_text = (resp.get("content") or "").strip()
                break

            # Append the assistant message with tool_calls so the model can
            # see its own previous tool requests in subsequent turns.
            messages.append({
                "role":       "assistant",
                "content":    resp.get("content"),
                "tool_calls": [
                    {
                        "id":   tc["id"],
                        "type": "function",
                        "function": {
                            "name":      tc["name"],
                            "arguments": tc["arguments"],
                        },
                    }
                    for tc in tool_calls
                ],
            })

            # Dispatch every tool call requested in this turn
            for tc in tool_calls:
                context = {"session_id": session_id}
                if tc["name"] == "simple_dataframe_query":
                    # Each pandas call gets a FRESH namespace clone so previous
                    # `answer` bindings don't bleed across iterations.
                    context["namespace"] = dict(pandas_namespace)
                tool_result_json = _lt.execute_tool_call(
                    tc["name"], tc["arguments"], context,
                )
                history.append({
                    "iteration":  it,
                    "tool":       tc["name"],
                    "arguments":  tc["arguments"],
                    "result_preview": tool_result_json[:300],
                })
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc["id"],
                    "content":      tool_result_json,
                })

        if final_text is None:
            error = error or f"tool-call loop hit {max_iterations}-iteration cap"

        ok = (final_text is not None) and bool(final_text.strip()) and (error is None)
        return {
            "ok":         ok,
            "final_text": final_text,
            "history":    history,
            "error":      error,
        }

    def _render_tool_result(
        self,
        state:        BiomarkerState,
        record:       dict,
        user_query:   str,
    ) -> None:
        """Append the tool-loop final answer to state and stash the last
        tool call's args as `last_query_code` so 'show me the code' still works."""
        state["messages"].append({
            "role":    "assistant",
            "content": record["final_text"],
        })
        # Save the last tool call code for the show_code action
        last_query = None
        last_engine = "sql"
        for h in reversed(record.get("history") or []):
            if h["tool"] in ("complex_duckdb_query", "simple_dataframe_query"):
                try:
                    args = json.loads(h.get("arguments") or "{}")
                    last_query = args.get("query")
                    last_engine = ("sql" if h["tool"] == "complex_duckdb_query"
                                   else "pandas")
                except Exception:
                    pass
                if last_query:
                    break
        if last_query:
            state["last_query_code"]   = last_query
            state["last_query_engine"] = last_engine
        state["intent"]       = "query_data"
        state["active_agent"] = "learning_agent"
        state["status"]       = "answered"

    # ── DuckDB / SQL query path ───────────────────────────────────────────────

    def _try_sql_query(
        self,
        state: BiomarkerState,
        user_query: str,
        df_candidates: List[Tuple[str, Any]],
    ):
        """
        SQL-first attempt: build a schema block from DuckDB DESCRIBE, ask the
        LLM to generate a SQL query, review it, execute via DuckDB. Returns
        an `ExecutionRecord`-compatible object (with .ok / .result / .error /
        .code) or None when DuckDB is unavailable.

        Falls through to the pandas path (in `_query_data`) when SQL fails.
        """
        try:
            from core import data_store as _ds
        except ImportError:
            return None
        if not _ds.is_available():
            return None

        session_id = state.get("session_id", "")
        store = _ds.get_store(session_id)
        if store is None or not store.table_names:
            return None

        schema_block = _ds.schema_text(session_id)
        if not schema_block.strip():
            return None

        # ── Build the column-friendly-name mapping block ────────────────────
        # When the workbook has group labels in a row above the headers (e.g.
        # "DMD Soleus" above "SpC J"), the LLM otherwise has no way to map
        # the user-friendly group name back to the real column name.
        column_groups = state.get("column_group_labels") or {}
        if column_groups:
            # Render BOTH directions so the LLM never has to "infer" a pairing:
            #   reverse: label  →  real column   (user types friendly name)
            #   forward: column →  label         (verification table)
            reverse = {}  # label → list of real columns
            for col, label in column_groups.items():
                reverse.setdefault(label, []).append(col)
            reverse_lines = "\n".join(
                f"  '{label}'  →  use column(s) {real_cols}"
                for label, real_cols in reverse.items()
            )
            forward_lines = "\n".join(
                f'  "{col}"  =  {label}'
                for col, label in column_groups.items()
            )
            cg_block = (
                "\n══════════════════════════════════════════════════════════\n"
                "GROUP-LABEL → REAL-COLUMN MAPPING (AUTHORITATIVE — DO NOT GUESS)\n"
                "══════════════════════════════════════════════════════════\n"
                "The user's workbook stores 'friendly' group labels (e.g. "
                "'DMD Soleus') in a separate row from the real column names "
                "(e.g. 'SpC J'). The mapping below is the ground truth — when "
                "the user mentions a friendly name, use the EXACT column on "
                "the right. NEVER invent your own mapping.\n\n"
                "Reverse lookup (friendly → real column):\n"
                f"{reverse_lines}\n\n"
                "Forward table (real column = friendly label):\n"
                f"{forward_lines}\n"
                "══════════════════════════════════════════════════════════\n"
            )
        else:
            cg_block = ""

        # ── Build SQL-generation prompt ───────────────────────────────────────
        sql_base_prompt = (
            "You write SQL queries (DuckDB dialect) to answer questions about a "
            "proteomics workbook. Every sheet is registered as a DuckDB table.\n"
            f"{cg_block}\n"
            "SCHEMA (DuckDB tables, exactly as available):\n"
            f"{schema_block}\n\n"
            f"USER QUESTION: {user_query!r}\n\n"
            "STRICT RULES — read carefully:\n"
            "\n"
            "[OUTPUT FORMAT]\n"
            "- Output a SINGLE SQL statement (no semicolons mid-query).\n"
            "- ALWAYS double-quote table names and column names that contain spaces,\n"
            "  punctuation, or mixed case: SELECT \"some col\" FROM \"sheet name\".\n"
            "- Return ONLY the SQL — no markdown fences, no prose, no explanation.\n"
            "- LIMIT results to 200 rows when returning a multi-row table.\n"
            "- If the question cannot be answered with SQL (e.g. concept question,\n"
            "  file structure not in tables), output the literal string 'NOT_SQL'.\n"
            "\n"
            "[METRIC-TYPE DISCIPLINE — never mix metrics]\n"
            "Proteomics workbooks usually have several PARALLEL column families that\n"
            "share the same per-sample prefix but encode different metrics — e.g.\n"
            "<S> SpC (spectral count), <S> Intensity, <S> Ratio H/L, <S> LFQ, MW, etc.\n"
            "- When the user asks for a specific metric (spectral count, intensity,\n"
            "  ratio, MW, concentration, log2FC), ONLY select columns whose name\n"
            "  matches that metric for the relevant sample(s).\n"
            "- Never sum, average, or filter across DIFFERENT metric types.\n"
            "- If the requested metric isn't in the schema, say so via NOT_SQL\n"
            "  rather than substituting a different metric.\n"
            "\n"
            "[IDENTIFIERS — always return them with values]\n"
            "When the question asks about proteins, ALWAYS include the protein\n"
            "identifier columns alongside the requested value:\n"
            "  - Protein name / description column (if present)\n"
            "  - Gene symbol column (if present)\n"
            "  - Accession / UniProt ID column (if present)\n"
            "This holds even if the user only asked for the value — the answer\n"
            "is incomplete without the row's identity.\n"
            "\n"
            "[SAMPLE-METADATA LOOKUPS — always join the metadata sheet]\n"
            "When the user asks what a SAMPLE corresponds to (its strain, treatment,\n"
            "client ID, group, etc.), look up the sample in the IDENTIFIER / METADATA\n"
            "sheet (typically the smallest sheet with one row per sample). Return\n"
            "ALL columns of that sheet for the matching sample row, not just one.\n"
            "Do NOT pick rows from the primary protein sheet for sample-level info.\n"
            "\n"
            "[PRESENCE / DETECTION QUESTIONS — return the value, not a boolean]\n"
            "When asked 'is X detected in sample Y' or 'does X appear in Y', return\n"
            "the actual quantitative value(s) — SpC, Intensity, whatever metric is\n"
            "appropriate. Do NOT return a bare TRUE/FALSE; the natural-language\n"
            "layer will phrase 'yes/no' from the value.\n"
            "\n"
            "[SAFE DIVISION — never divide by zero]\n"
            "Fold changes, ratios, and any division MUST use a safe-divide pattern:\n"
            "  CASE WHEN denom = 0 OR denom IS NULL THEN NULL ELSE numer / denom END\n"
            "Treat 0 as 'absent / not detected' — never silently produce 0 or inf.\n"
            "If a ratio is undefined for that pair, return NULL and let the NL\n"
            "layer explain 'undefined — protein absent in the denominator sample'.\n"
            "\n"
            "[TEXT MATCHING]\n"
            "- Use ILIKE '%pattern%' for case-insensitive partial matches on text\n"
            "  columns (protein name, gene, accession).\n"
            "- Search across multiple candidate identifier columns when uncertain\n"
            "  which column actually contains the user's reference.\n"
            "\n"
            "[AGGREGATIONS]\n"
            "- Use COUNT, SUM, AVG, MIN, MAX as appropriate.\n"
            "- For 'top N by X' use ORDER BY X DESC LIMIT N — include the\n"
            "  identifier columns in the SELECT (see [IDENTIFIERS] rule).\n"
            "\n"
            "[CONTAMINANT FLAG]\n"
            "If a table has an `is_contaminant` BOOLEAN column (added by the\n"
            "MaxQuant cleanup pipeline), add `WHERE is_contaminant = FALSE` to\n"
            "biological queries. For counting questions like 'how many CON__\n"
            "proteins?' DO use `WHERE is_contaminant = TRUE` — the flag exists\n"
            "specifically so counts work without scanning accession prefixes.\n"
        )

        from agents.code_reviewer import (
            CodeReviewerAgent, ExecutionRecord, review_and_revise,
        )
        if not hasattr(self, "_code_reviewer") or self._code_reviewer is None:
            self._code_reviewer = CodeReviewerAgent()

        def _generate_sql(extra: Optional[str]) -> str:
            prompt = sql_base_prompt
            if extra:
                prompt += f"\n\nREVISION GUIDANCE:\n{extra}\n"
            raw = self._call_llm(
                [{"role": "system", "content":
                    "You write DuckDB SQL queries — exact column names, no prose."},
                 {"role": "user", "content": prompt}],
                max_tokens=500, temperature=0.0,
            )
            # Strip code fences if the LLM added them
            cleaned = re.sub(r"^```(?:sql)?\s*|\s*```$", "", raw.strip(),
                             flags=re.MULTILINE).strip()
            return cleaned

        def _execute_sql(sql_str: str):
            # Sentinel: LLM punted because the question isn't SQL-shaped
            if sql_str.strip().upper().startswith("NOT_SQL"):
                return None, "LLM declined SQL — falling back to pandas."
            df_result, err = _ds.query(session_id, sql_str, max_rows=200)
            if err is not None:
                return None, err
            return df_result, None

        try:
            record: ExecutionRecord = review_and_revise(
                generator      = _generate_sql,
                executor       = _execute_sql,
                reviewer       = self._code_reviewer,
                user_question  = user_query,
                schema_context = schema_block,
                max_rounds     = 1,  # SQL is precise; one revision is usually enough
            )
        except Exception as exc:
            self.logger.debug("SQL review-revise crashed: %s", exc)
            return None

        return record

    def _render_sql_result(
        self,
        state:       BiomarkerState,
        record,
        user_query:  str,
    ) -> None:
        """Format a successful SQL ExecutionRecord into a chat message."""
        import pandas as pd
        result = record.result

        if isinstance(result, pd.DataFrame):
            if len(result) == 0:
                formatted = "_(query returned no rows)_"
            elif result.shape == (1, 1):
                # Scalar answer
                formatted = str(result.iloc[0, 0])
            else:
                # CSV avoids the tabulate dependency to_markdown requires
                formatted = result.head(30).to_csv(index=False)
        else:
            formatted = str(result)

        if len(formatted) > 4000:
            formatted = formatted[:4000] + "\n…[truncated]"

        nl_prompt = (
            f"The user asked: {user_query!r}\n\n"
            f"A DuckDB SQL query produced this result:\n```\n{formatted}\n```\n\n"
            "Write a clear, concise natural-language answer (2–5 sentences). "
            "Quote exact numbers/values. If the result is a table, summarise it. "
            "If it's empty, say so plainly."
        )
        try:
            nl_response = self._call_llm(
                [{"role": "system", "content":
                    "You translate SQL query results into clear answers. "
                    "Never fabricate values not in the result."},
                 {"role": "user", "content": nl_prompt}],
                max_tokens=500,
            )
        except Exception as exc:
            self.logger.warning("SQL NL response LLM failed: %s", exc)
            nl_response = f"Result:\n\n{formatted}"

        # Store the query quietly so the user can retrieve it via "show the code"
        # — by default we never include code in the chat message.
        state["last_query_code"]   = record.code
        state["last_query_engine"] = "sql"
        state["messages"].append({"role": "assistant", "content": nl_response.strip()})
        state["intent"]       = "query_data"
        state["active_agent"] = "learning_agent"
        state["status"]       = "answered"

    def _load_sheets_for_query(
        self, state: BiomarkerState, data_path: str,
    ) -> Dict[str, Any]:
        """Return {sheet_name: DataFrame}. Prefer cached state, fall back to disk."""
        import pandas as pd

        cached = state.get("all_sheets")
        if cached and isinstance(cached, dict):
            sheets = {}
            for k, v in cached.items():
                if isinstance(v, pd.DataFrame):
                    sheets[k] = v
            if sheets:
                return sheets

        try:
            if str(data_path).lower().endswith((".xlsx", ".xls")):
                return pd.read_excel(data_path, sheet_name=None)
            return {"data": pd.read_csv(data_path)}
        except Exception as exc:
            self.logger.warning("Sheet reload failed: %s", exc)
            return {}

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
            sample_cols  = state.get("sample_columns") or []
            meta_cols    = state.get("metadata_columns") or []
            label_map    = state.get("label_map") or {}
            g1           = state.get("group1_label")
            g2           = state.get("group2_label")
            g1_samps     = state.get("group1_samples") or []
            g2_samps     = state.get("group2_samples") or []
            all_sheets   = state.get("all_sheets") or {}
            data_path    = state.get("data_path") or state.get("raw_data_path") or "?"

            ctx += [
                f"- Data loaded: YES",
                f"- File path: {data_path}",
                f"- Proteins: {state.get('n_proteins','?')}",
                f"- Samples: {state.get('n_samples','?')}",
                f"- Data type: {state.get('data_type','?')}",
                f"- Omic type: {state.get('omic_type','proteomics')}",
                f"- Pooled design: {state.get('is_pooled_design', False)}",
                f"- Organism: {state.get('organism', 'not set')}",
                f"- Disease program: {state.get('disease_program', 'General')}",
                f"- Sample columns ({len(sample_cols)}): {sample_cols[:30]}",
                f"- Metadata columns ({len(meta_cols)}): {meta_cols[:20]}",
            ]
            if all_sheets:
                sheet_names = list(all_sheets.keys())
                ctx.append(f"- Number of sheets: {len(sheet_names)}")
                ctx.append(f"- Sheet names: {sheet_names}")
                # Show shape + a few columns from each sheet so factual questions
                # about file structure can be answered without query_data.
                try:
                    import pandas as _pd  # local import to keep top-level lean
                    for sname, sdf in list(all_sheets.items())[:8]:
                        if isinstance(sdf, _pd.DataFrame):
                            scols = list(sdf.columns)[:15]
                            ctx.append(
                                f"  · {sname!r}: shape={sdf.shape}, "
                                f"cols={scols}{' …' if len(sdf.columns) > 15 else ''}"
                            )
                except Exception:
                    pass
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

        # ── Multi-question split ──────────────────────────────────────────────
        # If the user pasted ≥2 questions, split them and answer each one
        # individually — each question gets its own routing decision and
        # its own response. Better-quality answers than batching everything
        # into a single LLM call.
        qs = _extract_questions(user_query)
        if len(qs) >= 2:
            self.logger.info("Multi-question detected (%d) — splitting and routing each.", len(qs))
            return self._handle_multi_question(state, qs)

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

        # ── Full auto-pipeline: summary + analyses + enrichment + viz ──────────
        if action == "run_full_pipeline":
            return self._run_full_pipeline(state)

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

        # ── Raw-data query (pandas on uploaded sheets) ─────────────────────────
        if action == "query_data":
            return self._query_data(state)

        # ── Answer (default) ──────────────────────────────────────────────────
        return self._answer(state)
