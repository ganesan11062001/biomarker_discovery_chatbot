"""
agents/enrichment_agent.py
Knowledge Layer — pathway enrichment with LLM-generated biological interpretation.

Design:
  - Only significant proteins (significance != "NS") are submitted for enrichment
  - Up- and down-regulated proteins are separated so enrichment is directional
  - All measured proteins are used as background (avoids genome-wide inflation)
  - LLM summary receives full biological context: groups, direction, fold changes
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from agents.base_agent import BaseAgent
from config.settings import get_settings
from core.state import BiomarkerState
from core.tracing import get_enrichment_metadata
from skills.run_enrichment import PathwaySkill

settings = get_settings()
logger   = logging.getLogger(__name__)

# ── LangSmith @traceable (graceful no-op if not installed) ───────────────────
try:
    from langsmith import traceable as _traceable
    from langsmith.run_helpers import get_current_run_tree as _get_run_tree
except ImportError:
    def _traceable(**_kw):          # type: ignore[misc]
        def _wrap(fn): return fn
        return _wrap
    def _get_run_tree():            # type: ignore[misc]
        return None


def _get_log2fc(protein_dict: dict) -> float:
    """Retrieve log2 fold-change from any of the field names used across skills."""
    for field in (
        "log2_fold_change",         # canonical (set by every skill + dual_engine)
        "mean_log2fc", "log2fc_python", "log2fc_r",   # dual-engine aliases
        "log2fc", "log2_ratio", "max_log2fc", "max_pairwise_log2fc",
    ):
        val = protein_dict.get(field)
        if val is not None:
            try:
                fv = float(val)
                if fv == fv:        # NaN check
                    return fv
            except (ValueError, TypeError):
                pass
    return 0.0


class EnrichmentAgent(BaseAgent):
    """
    Runs KEGG and GO pathway enrichment, then uses the LLM to produce
    a biologically grounded interpretation of the top pathways.
    """

    def __init__(self) -> None:
        super().__init__(
            deployment_name=settings.azure_deployment_enrichment,
            system_prompt_path="prompts/enrichment_agent.txt",
        )
        self.pathway_skill = PathwaySkill()

    # ── Main entry point ──────────────────────────────────────────────────────

    @_traceable(run_type="chain", name="agent.enrichment",
                tags=["biomarker-discovery", "enrichment"])
    def run(self, state: BiomarkerState) -> BiomarkerState:
        rt = _get_run_tree()
        if rt is not None:
            try:
                rt.extra.setdefault("metadata", {}).update(get_enrichment_metadata(state))
            except Exception:
                pass

        protein_source = state.get("top_biomarkers") or state.get("top_proteins")

        # If user chose "all", read the full results from the Excel file
        if state.get("enrichment_scope") == "all" and state.get("excel_path"):
            protein_source = self._load_all_significant(state) or protein_source

        # If user specified an explicit top-N, slice protein_source to that count
        enrichment_top_n = state.get("enrichment_top_n")
        if (
            enrichment_top_n
            and state.get("enrichment_scope") == "top_n"
            and protein_source
            and len(protein_source) > enrichment_top_n
        ):
            protein_source = protein_source[:enrichment_top_n]
            logger.info(
                "Enrichment: sliced to top %d proteins per user request (was %d)",
                enrichment_top_n, len(state.get("top_biomarkers") or []),
            )
        if not protein_source:
            msg = self._llm_no_data()
            state["status"]        = "error"
            state["error_message"] = "No biomarker results found."
            state["messages"].append({"role": "assistant", "content": msg})
            return state

        # Filter to significant proteins only (NS excluded from gene set)
        sig_proteins = [
            p for p in protein_source
            if p.get("significance") not in (None, "NS", "")
            and p.get("protein")
        ]
        if not sig_proteins:
            # Fallback: no significance field means all results are valid candidates
            sig_proteins = [p for p in protein_source if p.get("protein")]
            logger.info("No significance field found; using all %d top proteins.", len(sig_proteins))

        if not sig_proteins:
            state["messages"].append({
                "role": "assistant",
                "content": "No significant proteins found for pathway enrichment.",
            })
            return state

        protein_list = [p["protein"] for p in sig_proteins]

        # Separate by direction of regulation
        up_proteins   = [p["protein"] for p in sig_proteins if _get_log2fc(p) > 0]
        down_proteins = [p["protein"] for p in sig_proteins if _get_log2fc(p) < 0]

        # Background: all measured proteins from the experiment
        background_proteins = self._get_background_proteins(state)

        results_path = state.get("excel_path") or state.get("dea_result_path") or ""

        try:
            result = self.pathway_skill.execute(
                protein_list=protein_list,
                background_proteins=background_proteins,
                up_proteins=up_proteins   or None,
                down_proteins=down_proteins or None,
                dea_result_path=results_path,
                organism=state.get("organism", "human"),
                omic_type=("phosphoproteomics" if state.get("ptm_analysis")
                           else state.get("omic_type") or "proteomics"),
            )

            state["enrichment_result_path"] = result["enrichment_result_path"]
            state["pathways"]               = result["top_pathways"]

            # Every Enrichr library call errored — this is an infrastructure
            # failure, not a genuine "zero pathways enriched" result. Report
            # it honestly instead of letting the LLM narrate a biological
            # explanation for a service outage.
            if result.get("all_libraries_failed"):
                state["status"] = "enrichment_failed"
                msg = (
                    "⚠ **Pathway enrichment could not run** — the Enrichr "
                    f"service failed on all {result.get('libraries_attempted', 0)} "
                    "library queries (network error or an unexpected response). "
                    "This is a service issue, not a biological result — please "
                    "don't interpret it as 'no pathways enriched'. Try again in "
                    "a moment, or ask me to retry."
                )
                state["messages"].append({"role": "assistant", "content": msg})
                return state

            state["status"] = "enrichment_complete"

            # Persist pathways into comparison_history so cross-comparison
            # answer queries can access every prior enrichment, not just the last.
            _cmp_key = (
                f"{state.get('group1_label') or 'G1'}"
                f"_vs_"
                f"{state.get('group2_label') or 'G2'}"
            )
            _hist = dict(state.get("comparison_history") or {})
            _hist.setdefault(_cmp_key, {}).update({
                "pathways":              result["top_pathways"],
                "enrichment_result_path": result["enrichment_result_path"],
            })
            state["comparison_history"] = _hist

            # Store reproducible enrichment code for "show code"
            state["analysis_code"] = self._build_enrichment_code(
                protein_list, up_proteins, down_proteins,
                background_proteins, state,
            )

            # Generate pathway dotplot
            self._generate_pathway_plot(state, result)

            msg = self._llm_enrichment_summary(result, state, sig_proteins,
                                               up_proteins, down_proteins,
                                               background_proteins)
            state["messages"].append({
                "role": "assistant",
                "content": msg,
                "has_plots": bool(state.get("plot_paths")),
            })

            logger.info(
                "Enrichment complete | session=%s sig=%d up=%d down=%d kegg=%d go=%d bg=%s",
                state.get("session_id"),
                len(protein_list), len(up_proteins), len(down_proteins),
                result.get("n_kegg_significant", 0),
                result.get("n_go_significant", 0),
                result.get("background_size"),
            )

            rt = _get_run_tree()
            if rt is not None:
                try:
                    rt.extra.setdefault("metadata", {}).update(
                        get_enrichment_metadata(state, result)
                    )
                except Exception:
                    pass

        except Exception as exc:
            logger.exception("Pathway enrichment failed: %s", exc)
            state["status"]        = "error"
            state["error_message"] = str(exc)
            msg = self._llm_error(str(exc))
            state["messages"].append({"role": "assistant", "content": msg})

        return state

    # ── Load all significant proteins from Excel ─────────────────────────────

    @staticmethod
    def _load_all_significant(state: BiomarkerState) -> Optional[List[Dict]]:
        """Read the 'All Results' sheet from the analysis Excel and return
        all proteins that passed significance filters."""
        excel_path = state.get("excel_path")
        if not excel_path:
            return None
        try:
            xl = pd.ExcelFile(excel_path)
            if "All Results" not in xl.sheet_names:
                logger.warning("'All Results' sheet not found in %s (available: %s)",
                               excel_path, xl.sheet_names)
                return None
            df = xl.parse("All Results")
            sig = df[df["significance"].notna() & (df["significance"] != "NS")]
            if sig.empty:
                sig = df
            records = sig.to_dict("records")
            logger.info(
                "Enrichment scope=all: loaded %d significant proteins from Excel "
                "(vs %d in top_biomarkers)",
                len(records),
                len(state.get("top_biomarkers") or []),
            )
            return records
        except Exception as exc:
            logger.warning("Could not read All Results from Excel: %s", exc)
            return None

    # ── Background extraction ─────────────────────────────────────────────────

    def _get_background_proteins(self, state: BiomarkerState) -> Optional[List[str]]:
        """Read the first column of the processed data file as background gene set."""
        data_path = state.get("data_path")
        if not data_path:
            return None
        try:
            col = pd.read_csv(data_path, usecols=[0], header=0).iloc[:, 0]
            proteins = [str(p) for p in col.tolist() if pd.notna(p) and str(p).strip()]
            logger.info("Background: %d measured proteins from data file", len(proteins))
            return proteins or None
        except Exception as exc:
            logger.warning("Could not read background proteins from data file: %s", exc)
            return None

    # ── LLM summary ───────────────────────────────────────────────────────────

    @_traceable(run_type="chain", name="enrichment.summary",
                tags=["biomarker-discovery", "enrichment"])
    def _llm_enrichment_summary(
        self,
        result: dict,
        state: BiomarkerState,
        sig_proteins: list,
        up_proteins: List[str],
        down_proteins: List[str],
        background_proteins: Optional[List[str]],
    ) -> str:
        g1      = state.get("group1_label", "Group1")
        g2      = state.get("group2_label", "Group2")
        omic    = state.get("omic_type", "proteomics")
        organism = state.get("organism", "human")
        bg_size = result.get("background_size") or (len(background_proteins) if background_proteins else "genome-wide")

        # Top 3 up and down pathways
        up_lines   = _format_pathway_lines(result.get("up_pathways",   [])[:3], "↑")
        down_lines = _format_pathway_lines(result.get("down_pathways", [])[:3], "↓")
        all_top    = _format_pathway_lines(result.get("top_pathways",  [])[:5], "")

        # Top proteins with fold-changes
        top_fc = sorted(sig_proteins, key=lambda p: abs(_get_log2fc(p)), reverse=True)[:5]
        fc_lines = "\n".join(
            f"  {p['protein']} (log2FC={_get_log2fc(p):+.2f})" for p in top_fc
        )

        # log2FC convention: log2(group2 / group1) — positive = up in group2
        contam = result.get("contaminants_excluded") or []
        ctx = (
            f"PATHWAY ENRICHMENT ANALYSIS COMPLETE\n\n"
            f"Comparison: {g1} vs {g2}\n"
            f"Omic type: {omic} | Organism: {organism}\n\n"
            f"Proteins submitted for enrichment: {len(sig_proteins)} significant proteins\n"
            f"  - Up-regulated (higher in {g2}):   {len(up_proteins)}\n"
            f"  - Down-regulated (higher in {g1}): {len(down_proteins)}\n"
            f"Background gene set: {bg_size} measured proteins\n"
            + (f"Blood / prep contaminants excluded before submission: "
               f"{', '.join(contam[:8])}{'…' if len(contam) > 8 else ''}\n"
               if contam else "")
            + f"\nResults:\n"
            f"  KEGG pathways significant: {result.get('n_kegg_significant', 0)}\n"
            f"  GO terms significant:      {result.get('n_go_significant', 0)}\n\n"
        )

        if up_lines:
            ctx += f"Top UP-regulated pathways (higher in {g2} than {g1}):\n{up_lines}\n\n"
        if down_lines:
            ctx += f"Top DOWN-regulated pathways (lower in {g2} than {g1}):\n{down_lines}\n\n"
        if not up_lines and not down_lines and all_top:
            ctx += f"Top enriched pathways:\n{all_top}\n\n"

        ctx += f"Proteins with largest fold-changes:\n{fc_lines}\n\n"
        ctx += (
            "Task: provide a concise, biologically meaningful interpretation. "
            "Explain what biological processes are activated and suppressed in the comparison. "
            "Connect the pathway findings to the experimental context. "
            "Mention if the directionality reveals a coherent biological story. "
            "Be specific — cite actual pathway names and gene names from above."
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": ctx},
        ]
        try:
            return self._call_llm(messages, max_tokens=500)
        except Exception as exc:
            logger.warning("Enrichment LLM summary failed: %s — using fallback.", exc)
            return self._fallback_summary(result, g1, g2)

    # ── LLM error / no-data helpers ───────────────────────────────────────────

    def _llm_no_data(self) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": (
                "The user requested pathway enrichment but no biomarker results exist yet. "
                "Tell them they need to run biomarker analysis first."
            )},
        ]
        try:
            return self._call_llm(messages, max_tokens=150)
        except Exception:
            return "No biomarker results found. Please run the analysis first, then request enrichment."

    def _llm_error(self, error_text: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": f"Enrichment failed with error: {error_text}. Tell the user what went wrong."},
        ]
        try:
            return self._call_llm(messages, max_tokens=150)
        except Exception:
            return f"Pathway enrichment failed: {error_text}"

    # ── Pathway dotplot ──────────────────────────────────────────────────────

    def _generate_pathway_plot(self, state: BiomarkerState, result: Dict) -> None:
        """Generate a pathway dotplot and store paths in state."""
        try:
            from skills.run_visualization import plot_pathway_dot
            stem = state.get("session_id") or "session"
            out_dir = str(Path(settings.output_dir) / "plots" / stem)
            path = plot_pathway_dot(
                pathways=result.get("top_pathways", []),
                stem=stem,
                output_dir=out_dir,
            )
            if path:
                plot_paths = list(state.get("plot_paths") or [])
                plot_paths.append(path)
                state["plot_paths"] = plot_paths
                logger.info("Pathway dotplot generated: %s", path)
        except Exception as exc:
            logger.warning("Pathway dotplot generation failed: %s", exc)

    # ── Reproducible code ────────────────────────────────────────────────────

    @staticmethod
    def _build_enrichment_code(
        protein_list: List[str],
        up_proteins: List[str],
        down_proteins: List[str],
        background: Optional[List[str]],
        state: BiomarkerState,
    ) -> str:
        from skills.run_enrichment import _normalize_organism, _LIBRARIES

        organism = state.get("organism", "human")
        g1 = state.get("group1_label", "Group1")
        g2 = state.get("group2_label", "Group2")

        organism_key = _normalize_organism(organism)
        libs = _LIBRARIES.get(organism_key, _LIBRARIES["human"])
        enr_organism = "human" if organism_key == "human" else "mouse"

        # Build gene-list + optional background block separately to avoid
        # Python's ternary-inside-implicit-string-concat precedence bug.
        data_block = f"gene_list = {protein_list[:10]!r}  # ... ({len(protein_list)} total)\n"
        if background:
            data_block += (
                f"background = {background[:5]!r}  # ... ({len(background)} total)\n"
            )
        data_block += "\n"

        return (
            f"import gseapy as gp\n\n"
            f"# Pathway enrichment: {g1} vs {g2}\n"
            f"# Organism: {organism} (key: {organism_key})\n"
            f"# Significant proteins submitted: {len(protein_list)}\n"
            f"#   Up-regulated (higher in {g2}):   {len(up_proteins)}\n"
            f"#   Down-regulated (higher in {g1}): {len(down_proteins)}\n"
            f"# Background: {len(background) if background else 'genome-wide'} proteins\n\n"
            + data_block
            + f"libraries = {libs!r}\n\n"
            f"for lib in libraries:\n"
            f"    enr = gp.enrichr(\n"
            f"        gene_list=gene_list,\n"
            f"        gene_sets=lib,\n"
            f"        organism='{enr_organism}',\n"
            f"        background=background if background else 20000,\n"
            f"        cutoff=0.05,\n"
            f"    )\n"
            f"    sig = enr.results[enr.results['Adjusted P-value'] <= 0.05]\n"
            f"    print(f'{{lib}}: {{len(sig)}} significant terms')\n"
        )

    # ── Fallback ──────────────────────────────────────────────────────────────

    @staticmethod
    def _fallback_summary(result: dict, g1: str = "Group1", g2: str = "Group2") -> str:
        top5  = result.get("top_pathways", [])[:5]
        lines = [
            f"**Pathway Enrichment Complete** ({g1} vs {g2})\n",
            f"- KEGG pathways: **{result.get('n_kegg_significant', 0)}**",
            f"- GO terms: **{result.get('n_go_significant', 0)}**",
        ]
        bg = result.get("background_size")
        if bg:
            lines.append(f"- Background: **{bg} measured proteins**")
        lines.append("\n**Top pathways:**")
        for i, p in enumerate(top5):
            direction = f" [{p.get('direction', '')}]" if p.get("direction") not in (None, "all", "") else ""
            lines.append(
                f"  {i+1}. {p['pathway']}{direction} "
                f"(adj.p={float(p.get('p_adjust') or 1.0):.3e}, genes={p.get('gene_count','?')})"
            )
        return "\n".join(lines)


# ── Module-level helper ───────────────────────────────────────────────────────

def _format_pathway_lines(pathways: list, prefix: str) -> str:
    lines = []
    for p in pathways:
        tag = f" {prefix}" if prefix else ""
        lines.append(
            f"  •{tag} {p['pathway']} | {p.get('library','')} | "
            f"adj_p={float(p.get('p_adjust') or 1.0):.3e} | "
            f"genes={p.get('gene_count','?')} ({p.get('overlap','')})"
        )
    return "\n".join(lines)
