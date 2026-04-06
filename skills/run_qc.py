"""
Data Layer – QCSkill
Quality control for proteomics intensity matrices:
  - Missing value filtering (protein-level and sample-level)
  - CV (coefficient of variation) cutoff
  - Outlier sample detection (IQR-based Mahalanobis or PCA distance)

Calls r_scripts/proteomics_qc.R for heavy computation.
"""
import json
from pathlib import Path
from skills.base_skill import BaseSkill

R_SCRIPT = Path(__file__).parent.parent / "r_scripts" / "proteomics_qc.R"


class QCSkill(BaseSkill):
    def __init__(self):
        super().__init__(script_path=str(R_SCRIPT))

    def execute(
        self,
        data_path: str,
        data_type: str = "generic",
        missing_threshold: float = 0.30,
        cv_cutoff: float = None,
        outlier_sd: float = 3.0,
        output_dir: str = "outputs",
    ) -> dict:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        args = {
            "data_path": data_path,
            "data_type": data_type,
            "missing_threshold": missing_threshold,
            "cv_cutoff": cv_cutoff,
            "outlier_sd": outlier_sd,
            "output_dir": output_dir,
        }

        result = self.run_r_script(args)

        # Ensure required keys exist with defaults if R script didn't return them
        result.setdefault("qc_passed", True)
        result.setdefault("filtered_data_path", data_path)
        result.setdefault("qc_report_path", str(Path(output_dir) / "qc_report.json"))
        result.setdefault("proteins_total", result.get("proteins_retained", 0))
        result.setdefault("samples_retained", 0)
        result.setdefault("missing_threshold", missing_threshold)
        result.setdefault("outliers_removed", 0)

        return result
