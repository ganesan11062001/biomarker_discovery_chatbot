"""
agents/domain_expert.py
Domain Expert Agent — inspired by GenoMAS (Liu-Hy/GenoMAS).

After BiomarkerAgent computes statistical results (significant proteins, fold
changes, p-values), the Domain Expert produces a biological interpretation
pass: pathway implications, known disease associations, mechanistic clues.

This is a *separate* LLM call from the BiomarkerAgent summary — by isolating
biological reasoning into its own agent with a focused prompt, the
interpretation is more rigorous and cites verifiable knowledge rather than
generic boilerplate.

The agent is intentionally read-only with respect to state — it appends one
interpretation message and never mutates analysis results.
"""
from __future__ import annotations

import logging
from typing import Optional

from agents.base_agent import BaseAgent
from config.settings import get_settings
from core.state import BiomarkerState

settings = get_settings()
logger   = logging.getLogger(__name__)

try:
    from langsmith import traceable as _traceable
except ImportError:
    def _traceable(**_kw):           # type: ignore[misc]
        def _wrap(fn): return fn
        return _wrap


_DOMAIN_EXPERT_SYSTEM_PROMPT = """\
You are a domain expert in proteomics, cell biology, and disease mechanisms.

You have been given the FINISHED results of a differential-expression or
pooled-fold-change analysis — including a small list of top biomarkers with
their fold-changes and (where applicable) adjusted p-values. Your task is to
write a SHORT biological interpretation (≤250 words, markdown).

STRICT RULES:

1. Only discuss proteins, genes, and pathways that appear in the provided
   biomarker list. Do NOT invent additional proteins, GO terms, or pathways.

2. If you reference a pathway or biological process, ground it in well-known
   established biology — do not speculate beyond verified knowledge.

3. Avoid generic statements like "this may have implications for disease."
   Be specific: cite proteins BY NAME from the provided list and tie their
   direction of change to a concrete biological role (e.g. fast-twitch fiber
   contractility, mitochondrial respiration, inflammation).

4. Call out conflicting or counterintuitive results explicitly. If two markers
   move in opposite directions on the same pathway, say so.

5. If the comparison labels suggest a well-known disease model (the user
   tells you the labels — never assume), anchor your interpretation in that
   disease context. Otherwise stay neutral.

6. End with one concrete follow-up suggestion — typically either a
   validation experiment (Western blot, qPCR, IHC) of the top hit, or a
   downstream analysis (pathway enrichment, GO term over-representation).

OUTPUT FORMAT (markdown):

**Biological interpretation**

  • Observation 1 — concrete, cites a specific protein from the list.
  • Observation 2 — same.
  • Observation 3 — same (optional).

**Possible mechanism:** one paragraph.

**Suggested next step:** one sentence.
"""


class DomainExpertAgent(BaseAgent):
    """
    Post-analysis biological interpretation agent.

    Usage: `DomainExpertAgent().interpret(state)` returns a markdown message
    (does not mutate state — caller decides where to append it).
    """

    def __init__(self) -> None:
        super().__init__(
            deployment_name=settings.azure_deployment_domain_expert,
            system_prompt_path="prompts/domain_expert.txt",
        )

    @_traceable(run_type="chain", name="agent.domain_expert",
                tags=["biomarker-discovery", "domain_expert"])
    def interpret(self, state: BiomarkerState) -> Optional[str]:
        """Return a biological interpretation paragraph, or None if there's
        nothing to interpret yet."""
        top = state.get("top_biomarkers") or state.get("top_proteins") or []
        if not top:
            return None

        # Build a compact grounding block. The reviewer prompt explicitly
        # forbids referencing anything outside this list.
        lines = []
        for b in top[:20]:
            protein = b.get("protein", "?")
            fc      = b.get("log2_fold_change",
                            b.get("max_pairwise_log2fc",
                                  b.get("rescue_score", "?")))
            adj_p   = b.get("adj_p_value", "?")
            lines.append(f"- {protein}  log2FC={fc}  adj_p={adj_p}")
        biomarkers_block = "\n".join(lines)

        g1 = state.get("group1_label") or "Group1"
        g2 = state.get("group2_label") or "Group2"
        organism = state.get("organism") or "human"
        program  = state.get("disease_program") or "General"
        omic     = state.get("omic_type") or "proteomics"
        mode     = state.get("analysis_mode") or "supervised"
        n_sig    = state.get("n_significant", "?")

        ctx = (
            f"## Analysis context\n"
            f"- Comparison: **{g1}** vs **{g2}**\n"
            f"- Organism: {organism}\n"
            f"- Disease program: {program}\n"
            f"- Omic type: {omic}\n"
            f"- Mode: {mode}\n"
            f"- Significant biomarkers: {n_sig}\n\n"
            f"## Top biomarkers (cite ONLY from this list)\n"
            f"{biomarkers_block}\n"
        )

        messages = [
            {"role": "system", "content": _DOMAIN_EXPERT_SYSTEM_PROMPT},
            {"role": "user",   "content": ctx},
        ]
        try:
            response = self._call_llm(messages, max_tokens=550, temperature=0.3)
        except Exception as exc:
            logger.warning("Domain expert LLM failed: %s", exc)
            return None
        return response.strip()

    # BaseAgent demands run() — pass through. Use interpret() for the real API.
    def run(self, state: BiomarkerState) -> BiomarkerState:
        msg = self.interpret(state)
        if msg:
            state["messages"].append({"role": "assistant", "content": msg})
        return state
