"""
Analysis Layer – ProteomicsSkill
Differential expression analysis for proteomics data.

Supported methods (auto-selected by data_type):
  - olink_npx   → OlinkAnalyze (lmer mixed model)
  - ms_intensity → MSstats or DEP
  - generic      → limma (linear models for microarray/proteomics)

Calls r_scripts/limma_dea.R for all methods.
"""
from pathlib import Path
from skills.base_skill import BaseSkill

R_SCRIPT = Path(__file__).parent.parent / "r_scripts" / "limma_dea.R"


class ProteomicsSkill(BaseSkill):
    def __init__(self):
        super().__init__(script_path=str(R_SCRIPT))

    def execute(
        self,
        data_path: str,
        sample_group_col: str,
        contrast_groups: list,
        data_type: str = "generic",
        adj_pval_cutoff: float = 0.05,
        logfc_cutoff: float = 0.5,
        output_dir: str = "outputs",
    ) -> dict:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Choose method based on data type
        method_map = {
            "olink_npx": "olink",
            "ms_intensity": "msstats",
            "generic": "limma",
        }
        method = method_map.get(data_type, "limma")

        args = {
            "data_path": data_path,
            "sample_group_col": sample_group_col,
            "contrast_groups": contrast_groups,
            "method": method,
            "adj_pval_cutoff": adj_pval_cutoff,
            "logfc_cutoff": logfc_cutoff,
            "output_dir": output_dir,
        }

        result = self.run_r_script(args)

        result.setdefault("method", method)
        result.setdefault("dea_result_path", str(Path(output_dir) / "dea_results.csv"))
        result.setdefault("top_proteins", [])
        result.setdefault("n_significant", len(result.get("top_proteins", [])))
        result.setdefault("n_up", 0)
        result.setdefault("n_down", 0)

        return result
