"""
agents/code_reviewer.py
Code Reviewer Agent — inspired by GenoMAS (Liu-Hy/GenoMAS).

Reviews LLM-generated pandas code before it is executed, and after a runtime
error, suggests targeted revisions. Wrapping each LLM-driven code-generation
step with a review-revise loop dramatically improves answer accuracy:

  1. Generator LLM writes pandas code to answer a question.
  2. Reviewer LLM critiques it:
       - Does it correctly address the question?
       - Are column/sheet names matched to the actual schema?
       - Are edge cases handled (missing values, case sensitivity)?
       - Are forbidden operations present?
  3. If APPROVED → execute.
     If REJECTED → return the critique to the generator, which produces a
     revised version. Up to N revision rounds.
  4. If execution still raises after revisions → return the final error.

This module is deliberately stateless: callers (LearningAgent._query_data,
BiomarkerAgent) hold the generator LLM and the executor; CodeReviewer only
provides the review verdict.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import List, Optional

from agents.base_agent import BaseAgent
from config.settings import get_settings

settings = get_settings()
logger   = logging.getLogger(__name__)

# ── LangSmith @traceable (graceful no-op if not installed) ───────────────────
try:
    from langsmith import traceable as _traceable
except ImportError:
    def _traceable(**_kw):           # type: ignore[misc]
        def _wrap(fn): return fn
        return _wrap


_REVIEWER_SYSTEM_PROMPT = """\
You are a senior bioinformatics code reviewer.

Your job is to critique LLM-generated pandas/numpy code that answers a specific
question about a proteomics dataset. You see the question, the available data
schema (sheet names + columns), and the candidate code.

Return a JSON object with exactly these fields:

{
  "approved":    boolean,            // true if the code is correct and safe to run
  "severity":    "ok"|"minor"|"major"|"fatal",
  "issues":      [string, ...],      // concrete problems, empty list if none
  "suggestion":  string              // one-sentence fix suggestion if not approved
}

REVIEW CRITERIA (in priority order):

1. CORRECTNESS — does the code actually answer the user's question?
   - For a "lookup X for sample Y" question, the code must filter rows AND
     select the right sample column.
   - For an aggregation ("how many proteins with X"), the code must filter
     correctly and use the right aggregation (sum/count/nunique).
   - For sheet questions, the code must use the `sheets` dict, not `df`.

2. SCHEMA MATCH — do the column names referenced exist in the dataset?
   - If the code references "Spectral Count" but the column is "A SpC",
     that's a major issue.
   - Case mismatches (Protein vs protein) are minor — the code should
     case-fold or use `.str.contains(..., case=False)`.

3. ROBUSTNESS — does it handle missing data and absent rows?
   - A lookup must not crash when the protein isn't found; it should
     return 'NOT FOUND' or similar.
   - Numeric ops should handle NaN.

4. ANSWER VARIABLE — is `answer` assigned? Does its type make sense for
   the question (scalar for "what value", list for "which proteins",
   dict for "summarise these counts")?

5. SAFETY — does the code use any forbidden operations? (You don't need to
   re-check the safe-exec patterns; the executor blocks them. But flag obvious
   issues like file I/O, network calls, eval, getattr-tricks.)

Be concise — a few specific issues, never philosophical commentary. Output ONLY
the JSON object, no markdown fences.
"""


@dataclass
class ReviewResult:
    approved:   bool
    severity:   str
    issues:     List[str]
    suggestion: str

    @property
    def needs_revision(self) -> bool:
        return not self.approved


class CodeReviewerAgent(BaseAgent):
    """
    Standalone code-review agent. Stateless — one instance can review many
    candidate code blocks for many user sessions.
    """

    def __init__(self) -> None:
        super().__init__(
            deployment_name=settings.azure_deployment_code_reviewer,
            system_prompt_path="prompts/code_reviewer.txt",
        )

    @_traceable(run_type="chain", name="agent.code_reviewer",
                tags=["biomarker-discovery", "code_review"])
    def review(
        self,
        user_question:  str,
        schema_context: str,
        candidate_code: str,
        last_error:     Optional[str] = None,
    ) -> ReviewResult:
        """
        Critique candidate pandas code. Returns a structured verdict.

        Parameters
        ----------
        user_question  Original natural-language question.
        schema_context Sheet names + column lists + small previews.
        candidate_code The code the generator produced.
        last_error     If the code was already executed and failed, paste the
                       Python error message here so the reviewer can target it.
        """
        user_msg = (
            f"USER QUESTION:\n{user_question!r}\n\n"
            f"AVAILABLE DATA SCHEMA:\n{schema_context}\n\n"
            f"CANDIDATE CODE:\n```python\n{candidate_code}\n```\n"
        )
        if last_error:
            user_msg += f"\nLAST RUNTIME ERROR:\n```\n{last_error}\n```\n"

        messages = [
            {"role": "system", "content": _REVIEWER_SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ]
        try:
            raw = self._call_llm(messages, max_tokens=400,
                                 temperature=0.0, json_mode=True)
            obj = json.loads(raw)
            return ReviewResult(
                approved   = bool(obj.get("approved", False)),
                severity   = str(obj.get("severity", "minor")),
                issues     = [str(i) for i in (obj.get("issues") or [])],
                suggestion = str(obj.get("suggestion", "")),
            )
        except Exception as exc:
            logger.warning("Code review LLM failed: %s — auto-approving.", exc)
            # Fail open: if the reviewer LLM itself errors, don't block
            # legitimate code. The downstream sandbox is still the hard gate.
            return ReviewResult(
                approved=True, severity="ok",
                issues=[], suggestion="(reviewer LLM failed; auto-approved)",
            )

    # BaseAgent demands run() — provide a thin wrapper so LangGraph can call
    # the reviewer as a node in the future if needed.
    def run(self, state):  # pragma: no cover - reviewer is invoked directly
        return state


def review_and_revise(
    *,
    generator,                 # callable: (extra_instruction: str|None) -> code:str
    executor,                  # callable: (code: str) -> (result, error: str|None)
    reviewer:        CodeReviewerAgent,
    user_question:   str,
    schema_context:  str,
    max_rounds:      int = 2,
) -> "ExecutionRecord":
    """
    Coordinator for the write → review → execute → revise loop.

    The generator and executor are passed as callables so this helper stays
    decoupled from any specific call site (LearningAgent._query_data,
    BiomarkerAgent.analysis_code generation, etc.).

    Returns an ExecutionRecord with the final code, result, error, and a
    structured history of every round (for tracing / debugging).
    """
    history: List[dict] = []
    last_code  = ""
    last_error: Optional[str] = None
    last_result = None
    last_review: Optional[ReviewResult] = None

    for round_idx in range(max_rounds + 1):
        # ── 1. Generate / regenerate ──────────────────────────────────────────
        extra_instruction: Optional[str] = None
        if last_review and last_review.needs_revision:
            extra_instruction = (
                f"Your previous code was reviewed and needs revision.\n"
                f"Issues: {last_review.issues}\n"
                f"Suggestion: {last_review.suggestion}\n"
                f"Produce a corrected version that addresses each issue."
            )
        elif last_error:
            extra_instruction = (
                f"Your previous code raised this error when executed:\n"
                f"  {last_error}\n"
                f"Produce a fixed version that handles the failure mode "
                f"(wrong column name, missing value, partial match, etc.)."
            )

        try:
            code = generator(extra_instruction)
        except Exception as exc:
            logger.warning("Generator round %d failed: %s", round_idx, exc)
            history.append({"round": round_idx, "stage": "generate",
                            "error": str(exc)})
            break
        last_code = code

        # ── 2. Review ─────────────────────────────────────────────────────────
        review = reviewer.review(
            user_question  = user_question,
            schema_context = schema_context,
            candidate_code = code,
            last_error     = last_error,
        )
        history.append({
            "round":   round_idx,
            "stage":   "review",
            "code":    code,
            "review":  {"approved": review.approved,
                        "severity": review.severity,
                        "issues":   review.issues,
                        "suggestion": review.suggestion},
        })
        last_review = review

        # If reviewer rejects with major/fatal severity, regenerate without
        # executing (saves a sandbox round-trip on hopeless code).
        if not review.approved and review.severity in ("major", "fatal") \
                and round_idx < max_rounds:
            continue

        # ── 3. Execute ────────────────────────────────────────────────────────
        result, error = executor(code)
        history[-1]["execution"] = {"error": error, "ok": error is None}

        if error is None:
            last_result = result
            last_error  = None
            return ExecutionRecord(
                code=code, result=result, error=None,
                review=review, history=history, rounds_used=round_idx + 1,
            )

        last_error  = error
        last_result = result

    return ExecutionRecord(
        code=last_code, result=last_result, error=last_error,
        review=last_review, history=history, rounds_used=max_rounds + 1,
    )


@dataclass
class ExecutionRecord:
    code:        str
    result:      object
    error:       Optional[str]
    review:      Optional[ReviewResult]
    history:     List[dict]
    rounds_used: int

    @property
    def ok(self) -> bool:
        return self.error is None
