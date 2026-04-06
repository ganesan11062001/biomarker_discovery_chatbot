"""
Knowledge Layer – PathwaySkill
Pathway enrichment on a list of significant proteins.

Backends:
  - clusterProfiler (R): KEGG and GO enrichment
  - GSEApy (Python): KEGG and GO enrichment (fallback if R unavailable)

Calls r_scripts/pathway_enrichment.R.
"""
from pathlib import Path
from skills.base_skill import BaseSkill

R_SCRIPT = Path(__file__).parent.parent / "r_scripts" / "pathway_enrichment.R"


class PathwaySkill(BaseSkill):
    def __init__(self):
        super().__init__(script_path=str(R_SCRIPT))

    def execute(
        self,
        protein_list: list,
        dea_result_path: str,
        organism: str = "human",
        pval_cutoff: float = 0.05,
        output_dir: str = "outputs",
    ) -> dict:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Map organism name to KEGG/OrgDb identifiers
        organism_map = {
            "human": {"kegg_org": "hsa", "orgdb": "org.Hs.eg.db"},
            "mouse": {"kegg_org": "mmu", "orgdb": "org.Mm.eg.db"},
            "rat":   {"kegg_org": "rno", "orgdb": "org.Rn.eg.db"},
        }
        org_ids = organism_map.get(organism.lower(), organism_map["human"])

        args = {
            "protein_list": protein_list,
            "dea_result_path": dea_result_path,
            "kegg_org": org_ids["kegg_org"],
            "orgdb": org_ids["orgdb"],
            "pval_cutoff": pval_cutoff,
            "output_dir": output_dir,
        }

        result = self.run_r_script(args)

        result.setdefault("enrichment_result_path", str(Path(output_dir) / "enrichment_results.csv"))
        result.setdefault("top_pathways", [])
        result.setdefault("n_kegg_significant", 0)
        result.setdefault("n_go_significant", 0)

        return result
