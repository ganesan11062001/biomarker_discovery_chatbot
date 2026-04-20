"""
scripts/verify_pipeline.py
End-to-end pipeline verification — no API / Streamlit required.

Runs:
  1. DataLoadingSkill   — parse Excel / CSV
  2. IngestionAgent     — enrich state with metadata
  3. BiomarkerAgent     — dispatch omic skill, produce top biomarkers

Usage:
    python3 scripts/verify_pipeline.py <path_to_xlsx_or_csv>
"""
import logging
import sys
from pathlib import Path
from pprint import pformat

# ── Make project root importable ──────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("verify_pipeline")


def _sep(title: str) -> None:
    print(f"\n{'='*64}")
    print(f"  {title}")
    print(f"{'='*64}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/verify_pipeline.py <path_to_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    if not Path(file_path).exists():
        print(f"[ERROR] File not found: {file_path}")
        sys.exit(1)

    suffix      = Path(file_path).suffix.lower()
    data_format = "excel" if suffix in (".xlsx", ".xls") else "csv"

    # ── Stage 0: imports ──────────────────────────────────────────────────────
    _sep("STAGE 0 — Imports")
    from agents.ingestion_agent import IngestionAgent
    from agents.biomarker_agent  import BiomarkerAgent
    print("  ✅  Agents imported successfully")

    # ── Stage 1: build initial state ──────────────────────────────────────────
    _sep("STAGE 1 — Initial State")
    state = {
        "session_id":    "test-session",
        "messages":      [],
        "data_path":     file_path,
        "data_format":   data_format,
        "disease_program": "DMD",
        "organism":      "mouse",
    }
    print(f"  data_path   : {file_path}")
    print(f"  data_format : {data_format}")

    # ── Stage 2: IngestionAgent ───────────────────────────────────────────────
    _sep("STAGE 2 — IngestionAgent")
    ingestion_agent = IngestionAgent()
    state = ingestion_agent.run(state)

    print(f"\n  status           : {state.get('status')}")
    if state.get("status") == "error":
        print(f"  [ERROR] {state.get('error_message')}")
        print("\n  Last message:")
        for m in (state.get("messages") or []):
            print(f"    [{m['role']}] {m['content'][:120]}")
        sys.exit(1)

    print(f"  data_type        : {state.get('data_type')}")
    print(f"  n_proteins       : {state.get('n_proteins')}")
    print(f"  n_samples        : {state.get('n_samples')}")
    print(f"  is_pooled_design : {state.get('is_pooled_design')}")
    print(f"  omic_type        : {state.get('omic_type')}")
    print(f"  label_map        : {state.get('label_map')}")

    sc = state.get("sample_columns") or []
    mc = state.get("metadata_columns") or []
    print(f"  sample_columns   : {sc[:6]}{'…' if len(sc) > 6 else ''}")
    print(f"  metadata_columns : {mc[:4]}{'…' if len(mc) > 4 else ''}")
    print(f"  raw_data_path    : {state.get('raw_data_path')}")
    print(f"  data_path (proc) : {state.get('data_path')}")
    print(f"  all_sheets       : {list((state.get('all_sheets') or {}).keys())}")

    print("\n  ── Ingestion messages ──────────────────────────────────")
    for m in (state.get("messages") or []):
        print(f"  [{m['role']}]\n{m['content'][:400]}\n")

    # ── Stage 3: BiomarkerAgent ───────────────────────────────────────────────
    _sep("STAGE 3 — BiomarkerAgent")

    # For supervised designs (non-pooled) we need group assignments.
    # Auto-assign first half / second half of sample columns if no groups set.
    if not state.get("is_pooled_design"):
        cols = state.get("sample_columns") or []
        if cols and not state.get("group1_samples"):
            mid = max(1, len(cols) // 2)
            state["group1_samples"] = cols[:mid]
            state["group2_samples"] = cols[mid:]
            state["group1_label"]   = "Group1"
            state["group2_label"]   = "Group2"
            print(f"  [auto-assigned]  Group1={cols[:mid]}  Group2={cols[mid:]}")

    biomarker_agent = BiomarkerAgent()
    state = biomarker_agent.run(state)

    print(f"\n  status           : {state.get('status')}")
    if state.get("status") == "error":
        print(f"  [ERROR] {state.get('error_message')}")
        sys.exit(1)

    print(f"  omic_type        : {state.get('omic_type')}")
    print(f"  analysis_mode    : {state.get('analysis_mode')}")
    print(f"  n_significant    : {state.get('n_significant')}")
    print(f"  excel_path       : {state.get('excel_path')}")

    qc = state.get("qc_summary") or {}
    print(f"  proteins_after_qc: {qc.get('proteins_after_qc', '?')}")
    print(f"  contrasts_computed:{qc.get('contrasts_computed', '?')}")
    print(f"  groups_detected  : {qc.get('groups_detected', '?')}")
    plots = qc.get("plot_paths") or []
    print(f"  plots generated  : {len(plots)}")

    # ── Top biomarkers ────────────────────────────────────────────────────────
    _sep("STAGE 3 — Top 10 Biomarkers")
    top = state.get("top_biomarkers") or []
    if not top:
        print("  [WARN] No top_biomarkers in state!")
    else:
        print(f"  Total biomarkers returned: {len(top)}\n")
        for b in top[:10]:
            rank    = b.get("rank", "?")
            protein = b.get("protein", "?")
            rescue  = b.get("rescue_score")
            fc_parts = "  ".join(
                f"{k}={v:+.2f}" for k, v in b.items()
                if k not in ("rank", "protein", "rescue_score") and isinstance(v, float)
            )
            if rescue is not None:
                print(f"  #{rank:>2}  {protein:<30}  rescue={rescue:.3f}  {fc_parts}")
            else:
                lfc = b.get("log2_fold_change", "?")
                adj = b.get("adj_p_value", "?")
                print(f"  #{rank:>2}  {protein:<30}  log2FC={lfc}  adj_p={adj}")

    # ── LLM summary ───────────────────────────────────────────────────────────
    _sep("STAGE 3 — LLM Analysis Summary")
    summary = state.get("analysis_summary") or "(no summary generated)"
    print(summary[:800])

    print(f"\n{'='*64}")
    print("  ✅  Pipeline verification complete")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
