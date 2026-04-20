"""
scripts/debug_ingest.py
Run this directly to diagnose Excel parsing without starting the full API.

Usage:
    python3 scripts/debug_ingest.py path/to/your_file.xlsx
"""
import logging
import sys
from pathlib import Path

# Make project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)-8s %(name)s — %(message)s",
)

from skills.load_data import DataLoadingSkill


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/debug_ingest.py <path_to_excel_or_csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    print(f"\n{'='*60}")
    print(f"Parsing: {file_path}")
    print(f"{'='*60}\n")

    skill = DataLoadingSkill()
    try:
        result = skill.execute(
            data_path=file_path,
            data_format="excel" if file_path.endswith((".xlsx", ".xls")) else "csv",
            output_dir="data/processed",
        )
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        raise

    print(f"\n{'='*60}")
    print("RESULT")
    print(f"{'='*60}")
    for k, v in result.items():
        if k in ("sample_columns", "metadata_columns") and isinstance(v, list):
            print(f"  {k} ({len(v)}): {v[:8]}")
        elif k == "label_map":
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v}")

    print(f"\n  is_pooled_design : {result.get('is_pooled_design')}")
    print(f"  label_map        : {result.get('label_map')}")
    print(f"  n_proteins       : {result.get('n_proteins')}")
    print(f"  n_samples        : {result.get('n_samples')}")
    print(f"  data_type        : {result.get('data_type')}")
    print()


if __name__ == "__main__":
    main()
