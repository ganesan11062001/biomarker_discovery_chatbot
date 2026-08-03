"""
End-to-end smoke test of the biomarker discovery pipeline.

Exercises:
  1. Data ingestion        (IngestionAgent)
  2. Differential analysis (BiomarkerAgent)
  3. Pathway enrichment    (EnrichmentAgent — requires gseapy + internet)
  4. Visualization         (VisualizationAgent)

Run:  python scripts/smoke_test_pipeline.py <path/to/file.xlsx>
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def main(data_path: str) -> int:
    from core.state import BiomarkerState
    from core.session_manager import SessionManager
    from agents.ingestion_agent import IngestionAgent
    from agents.biomarker_agent import BiomarkerAgent
    from agents.enrichment_agent import EnrichmentAgent
    from agents.visualization_agent import VisualizationAgent

    failures: list[str] = []
    src = Path(data_path).resolve()
    assert src.exists(), f"Data file not found: {src}"

    # ── 0. Seed state ────────────────────────────────────────────────────────
    session_id = SessionManager.create_session()
    state: BiomarkerState = SessionManager.get(session_id)  # type: ignore[assignment]
    state["file_path"] = str(src)
    state["messages"] = []

    # ── 1. Ingestion ─────────────────────────────────────────────────────────
    _banner("1. INGESTION")
    try:
        state = IngestionAgent().run(state)
        print(f"  proteins:        {len(state.get('proteins') or [])}")
        print(f"  sample_columns:  {len(state.get('sample_columns') or [])}")
        print(f"  data_type:       {state.get('data_type')}")
        print(f"  all_groups:      {list((state.get('all_groups') or {}).keys())}")
        if not state.get("sample_columns"):
            failures.append("Ingestion: sample_columns empty")
    except Exception as e:
        traceback.print_exc()
        failures.append(f"Ingestion crashed: {e!r}")
        return _report(failures)

    # ── 2. Differential analysis ─────────────────────────────────────────────
    _banner("2. DIFFERENTIAL ANALYSIS")
    groups = state.get("all_groups") or {}
    group_names = list(groups.keys())
    if len(group_names) < 2:
        failures.append("Analysis: <2 groups detected — cannot run differential")
        return _report(failures)

    g1_label, g2_label = group_names[0], group_names[1]
    state["group1_label"]   = g1_label
    state["group1_samples"] = groups[g1_label]
    state["group2_label"]   = g2_label
    state["group2_samples"] = groups[g2_label]
    print(f"  comparing: {g1_label} ({len(groups[g1_label])} samples) "
          f"vs {g2_label} ({len(groups[g2_label])} samples)")
    try:
        state = BiomarkerAgent().run(state)
        top = state.get("top_biomarkers") or []
        print(f"  top_biomarkers:  {len(top)} returned")
        if top:
            head = top[0]
            print(f"  top hit:         {head.get('protein','?')} "
                  f"log2FC={head.get('log2_fold_change','?')} "
                  f"adj_p={head.get('adj_p_value','?')}")
        if not top:
            failures.append("Analysis: top_biomarkers empty")
    except Exception as e:
        traceback.print_exc()
        failures.append(f"Analysis crashed: {e!r}")

    # ── 3. Pathway enrichment ────────────────────────────────────────────────
    _banner("3. PATHWAY ENRICHMENT")
    try:
        state = EnrichmentAgent().run(state)
        pw = state.get("pathways") or []
        print(f"  pathways:        {len(pw)} returned")
        if pw:
            for p in pw[:3]:
                print(f"    - {p.get('pathway','?')[:80]}  "
                      f"adj_p={p.get('p_adjust', p.get('adj_p','?'))}")
        else:
            print("  (no enriched pathways — could be a small input or Enrichr returned nothing)")
    except Exception as e:
        traceback.print_exc()
        failures.append(f"Enrichment crashed: {e!r}")

    # ── 4. Visualization ─────────────────────────────────────────────────────
    _banner("4. VISUALIZATION")
    try:
        state = VisualizationAgent().run(state)
        plots = state.get("plot_paths") or []
        png   = [p for p in plots if str(p).endswith(".png")]
        html  = [p for p in plots if str(p).endswith(".html")]
        print(f"  plot_paths:      {len(plots)} ({len(png)} PNG, {len(html)} HTML)")
        for p in plots[:6]:
            exists = Path(p).exists()
            size_kb = (Path(p).stat().st_size / 1024) if exists else 0
            print(f"    - [{'OK' if exists else 'MISSING'}] {p}  ({size_kb:.1f} KB)")
        if not plots:
            failures.append("Visualization: no plot_paths produced")
        elif not any(Path(p).exists() for p in plots):
            failures.append("Visualization: plot_paths reference non-existent files")
    except Exception as e:
        traceback.print_exc()
        failures.append(f"Visualization crashed: {e!r}")

    return _report(failures)


def _report(failures: list[str]) -> int:
    _banner("RESULT")
    if not failures:
        print("  ALL PIPELINE STAGES OK")
        return 0
    print(f"  FAILED ({len(failures)} issue{'s' if len(failures)!=1 else ''}):")
    for f in failures:
        print(f"    - {f}")
    return 1


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else \
        r"data/raw/01d82a2f-132e-45e7-8636-5ce247c0369e/8a5d4dc521014daa80a2e7e2ee30140f.xlsx"
    raise SystemExit(main(path))
