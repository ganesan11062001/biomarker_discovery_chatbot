from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages

class BiomarkerState(TypedDict):
    messages: Annotated[list, add_messages]
    session_id: str
    user_query: str
    intent: Optional[str]
    active_agent: Optional[str]

    # Data ingestion
    file_id: Optional[str]
    data_path: Optional[str]           # Path to loaded CSV or Excel file
    data_type: Optional[str]           # "olink_npx" | "ms_intensity" | "generic"
    data_format: Optional[str]         # "csv" | "excel"
    n_proteins: Optional[int]          # Protein count after loading
    n_samples: Optional[int]           # Sample count after loading

    # Analysis configuration
    sample_group_col: Optional[str]
    contrast_groups: Optional[list]
    disease_program: Optional[str]

    # QC results (Data Layer)
    qc_report_path: Optional[str]
    qc_passed: Optional[bool]

    # Proteomics analysis results (Analysis Layer)
    dea_result_path: Optional[str]
    top_proteins: Optional[list]

    # Pathway enrichment results (Knowledge Layer)
    enrichment_result_path: Optional[str]
    pathways: Optional[list]

    # Output (Output Layer)
    plot_paths: Optional[list]
    report_path: Optional[str]

    status: Optional[str]
    error_message: Optional[str]
