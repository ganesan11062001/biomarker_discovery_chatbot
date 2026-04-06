"""
Output Layer – ReportingSkill
Generates publication-quality visualizations and a summary report.

Plots produced:
  - Volcano plot      (DEA results)
  - Heatmap           (top proteins × samples)
  - Dot plot          (pathway enrichment, if available)
  - Biomarker ranking table (CSV)

Calls r_scripts/plot_volcano.R.
"""
from pathlib import Path
from skills.base_skill import BaseSkill

R_SCRIPT = Path(__file__).parent.parent / "r_scripts" / "plot_volcano.R"


class ReportingSkill(BaseSkill):
    def __init__(self):
        super().__init__(script_path=str(R_SCRIPT))

    def execute(
        self,
        dea_result_path: str,
        enrichment_result_path: str = None,
        top_proteins: list = None,
        top_pathways: list = None,
        contrast_groups: list = None,
        disease_program: str = "",
        output_dir: str = "outputs",
    ) -> dict:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        args = {
            "dea_result_path": dea_result_path,
            "enrichment_result_path": enrichment_result_path or "",
            "top_proteins": [p["protein"] for p in (top_proteins or [])],
            "contrast_groups": contrast_groups or [],
            "disease_program": disease_program,
            "output_dir": output_dir,
        }

        result = self.run_r_script(args)

        result.setdefault("plot_paths", [
            str(Path(output_dir) / "volcano_plot.png"),
            str(Path(output_dir) / "heatmap.png"),
        ])
        if enrichment_result_path:
            result["plot_paths"].append(str(Path(output_dir) / "enrichment_dotplot.png"))
        result.setdefault("report_path", str(Path(output_dir) / "biomarker_ranking.csv"))
        result.setdefault("plain_language_summary", "")

        return result
