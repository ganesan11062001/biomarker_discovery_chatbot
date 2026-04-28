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
from typing import Any, Dict, List, Literal, Optional

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
    "show_code", "modify_code", "query_database", "answer",
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
  "answer"              — answer a question, explain something, or have a conversation

Decision rules (in priority order):
1.  Questions ("what is X", "explain X", "how does Y work", "what did the analysis find") → "answer"
2.  Off-topic messages (random forest, Python, general statistics, etc.) → "answer"
3.  No data loaded yet → "answer" (tell user to upload a file first)
4.  "show code" / "give me the code" / "what code was used" → "show_code"
5.  "modify the code" / "change threshold" / "alter the script" / "add volcano" → "modify_code"
6.  "look up proteins" / "get gene names" / "annotate proteins" / "UniProt" / "convert IDs" → "query_database"
7.  "run analysis" with NO group names AND non-pooled design → "run_all_comparisons"
8.  "run analysis" / "analyze" with specific group names mentioned → "run_analysis"
    - Set group1_label, group1_samples, group2_label, group2_samples
    - Match names to available_columns; leave lists empty if uncertain
9.  Pathway / enrichment / KEGG / GO → "run_enrichment"
10. Plot / visualize / chart / heatmap / volcano / report → "run_visualization"
11. Pooled design "run analysis" → "run_all_comparisons"

For "run_analysis" populate groups only when you can confidently match column names.
Leave group sample lists empty if the user hasn't given enough information.

OUTPUT: valid JSON only — no markdown fences, no prose, no trailing text.
{
  "action": "<action>",
  "group1_label": "<label or null>",
  "group1_samples": [],
  "group2_label": "<label or null>",
  "group2_samples": [],
  "requested_plots": [],
  "confidence": 0.95,
  "reason": "<one sentence explaining the decision>"
}

"confidence" is a float 0.0–1.0 representing how certain you are of this decision.
The system will automatically demote decisions with confidence < 0.7 to "answer"
as a safety measure against misrouted pipeline actions.

For "run_visualization": populate "requested_plots" with the canonical names of plots the user
specifically asked for (e.g. ["volcano", "pca", "heatmap"]).
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
1. When referencing this session's results, ONLY cite proteins, fold-change values,
   p-values, and pathways that are explicitly listed in the session context below.
2. If you are asked for a specific value (e.g. "what is the fold-change of PROTEIN_X")
   and it is NOT in the session data, say "that value is not in the current results"
   rather than estimating or fabricating it.
3. For off-topic / general science questions (not about this session's data),
   answer freely and accurately from your training knowledge — the grounding
   rule applies only to session-specific claims.
4. Do NOT invent protein names, gene symbols, accession IDs, or pathway names
   that are not grounded in the session context or your training knowledge.
5. Use markdown formatting. Be concise and precise.
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

        ctx  = "SESSION STATE:\n"
        ctx += f"  data_loaded: {bool(state.get('data_type'))}\n"
        ctx += f"  data_type: {state.get('data_type', 'none')}\n"
        ctx += f"  n_proteins: {state.get('n_proteins', 0)}\n"
        ctx += f"  n_samples: {state.get('n_samples', 0)}\n"
        ctx += f"  is_pooled_design: {state.get('is_pooled_design', False)}\n"
        ctx += f"  omic_type: {state.get('omic_type', 'none')}\n"
        ctx += f"  analysis_complete: {state.get('n_significant') is not None}\n"
        ctx += f"  n_significant: {state.get('n_significant', 'none')}\n"
        ctx += f"  analysis_mode: {state.get('analysis_mode', 'none')}\n"
        ctx += f"  group1_label: {state.get('group1_label', 'none')}\n"
        ctx += f"  group2_label: {state.get('group2_label', 'none')}\n"
        ctx += f"  has_analysis_code: {bool(state.get('analysis_code'))}\n"
        ctx += f"  has_plots: {bool(state.get('plot_paths'))}\n"
        ctx += f"  enrichment_done: {bool(state.get('pathways'))}\n"
        ctx += f"  status: {state.get('status', 'ready')}\n"
        ctx += f"  available_columns (first 20): {sample_cols[:20]}\n"
        ctx += f"  label_map (pooled groups): {label_map}\n"
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
        Run BiomarkerAgent for every pair of detected groups.
        Pooled designs delegate to PooledFoldChangeSkill (covers all contrasts).
        """
        is_pooled = state.get("is_pooled_design") or state.get("omic_type") == "proteomics_pooled"

        if is_pooled:
            label_map = state.get("label_map") or {}
            groups = list(label_map.values()) if label_map else ["all groups"]
            contrast_preview = " · ".join(groups)
            state["messages"].append({
                "role": "assistant",
                "content": (
                    f"Pooled design detected — running log₂ fold-change analysis "
                    f"across all contrasts for groups: **{contrast_preview}** …"
                ),
            })
            return self._specialist("biomarker").run(state)

        # Non-pooled: infer groups from column names
        sample_cols = state.get("sample_columns") or []
        groups = self._infer_groups(sample_cols)

        if len(groups) < 2:
            state["messages"].append({
                "role": "assistant",
                "content": (
                    "I couldn't automatically detect groups from your column names. "
                    "Please assign **Group 1** and **Group 2** in the sidebar and click "
                    "**▶ Run Analysis**, or type something like: "
                    "*'compare WT_1, WT_2, WT_3 vs KO_1, KO_2, KO_3'*."
                ),
            })
            return state

        group_names = list(groups.keys())
        pairs = list(combinations(group_names, 2))
        summary_lines = [
            f"Running **{len(pairs)} pairwise comparisons** across "
            f"{len(group_names)} groups: {', '.join(group_names)}\n"
        ]

        biomarker = self._specialist("biomarker")
        last_n_sig = 0

        for g1_name, g2_name in pairs:
            state["group1_label"]   = g1_name
            state["group1_samples"] = groups[g1_name]
            state["group2_label"]   = g2_name
            state["group2_samples"] = groups[g2_name]
            state["analysis_mode"]  = "supervised"

            state = biomarker.run(state)
            n_sig = state.get("n_significant", 0)
            last_n_sig = n_sig
            top3 = [b.get("protein", "") for b in (state.get("top_biomarkers") or [])[:3]]
            summary_lines.append(
                f"- **{g1_name} vs {g2_name}**: {n_sig} significant | "
                f"top: {', '.join(top3)}"
            )

        state["messages"].append({
            "role": "assistant",
            "content": "\n".join(summary_lines),
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
            # Strip markdown fences if the LLM added them
            modified = modified.removeprefix("```python").removeprefix("```").removesuffix("```").strip()
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
        ctx = ["## Session context"]
        if state.get("data_type"):
            ctx += [
                f"- Data loaded: YES — {state.get('n_proteins','?')} proteins, "
                f"{state.get('n_samples','?')} samples, type={state.get('data_type','?')}",
                f"- Omic type: {state.get('omic_type','proteomics')}",
                f"- Pooled design: {state.get('is_pooled_design', False)}",
                f"- Groups: {state.get('label_map') or {state.get('group1_label'): 'g1', state.get('group2_label'): 'g2'}}",
                f"- Organism: {state.get('organism', 'unknown')}",
                f"- Disease program: {state.get('disease_program', 'General')}",
            ]
            if state.get("n_significant") is not None:
                top5 = [b.get("protein","") for b in (state.get("top_biomarkers") or [])[:5]]
                ctx += [
                    "- Analysis complete: YES",
                    f"- Significant biomarkers: {state.get('n_significant')}",
                    f"- Top 5: {top5}",
                    f"- Method: {state.get('analysis_mode','?')} mode",
                    f"- Comparison: {state.get('group1_label','?')} vs {state.get('group2_label','?')}",
                ]
            else:
                ctx.append("- Analysis complete: NO")
            if state.get("pathways"):
                top3pw = [p.get("pathway","") for p in state["pathways"][:3]]
                ctx.append(f"- Enrichment done: YES — top pathways: {top3pw}")
            if state.get("plot_paths"):
                ctx.append(f"- Plots generated: {len(state['plot_paths'])}")
        else:
            ctx.append("- Data loaded: NO")

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

        # ── Load data ─────────────────────────────────────────────────────────
        if action == "load_data":
            return self._specialist("ingestion").run(state)

        # ── Specific comparison (groups named by user) ─────────────────────────
        if action == "run_analysis":
            g1_label   = decision.get("group1_label")
            g1_samples = decision.get("group1_samples") or []
            g2_label   = decision.get("group2_label")
            g2_samples = decision.get("group2_samples") or []
            if g1_samples and g2_samples:
                state["group1_label"]   = g1_label or "Group1"
                state["group1_samples"] = g1_samples
                state["group2_label"]   = g2_label or "Group2"
                state["group2_samples"] = g2_samples
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
