from pathlib import Path
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # ── Azure OpenAI ──────────────────────────────────────────────────────────
    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_api_version: str = "2024-08-01-preview"

    # ── Model deployments ─────────────────────────────────────────────────────
    azure_deployment_chat: str = "gpt-4o"
    azure_deployment_ingestion: str = "gpt-4o"
    azure_deployment_biomarker: str = "gpt-4o"
    azure_deployment_enrichment: str = "gpt-4o"
    azure_deployment_visualization: str = "gpt-4o"

    # ── Directories ───────────────────────────────────────────────────────────
    data_raw_dir: str = "data/raw"
    data_processed_dir: str = "data/processed"
    output_dir: str = "outputs"

    # ── API / UI ──────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_base_url: str = "http://localhost:8000"

    # ── Runtime ───────────────────────────────────────────────────────────────
    app_env: str = "development"
    log_level: str = "INFO"
    max_file_size_mb: int = 200

    # ── Analysis defaults ─────────────────────────────────────────────────────
    top_n_biomarkers: int = 50
    adj_pval_cutoff: float = 0.05
    log2fc_cutoff: float = 1.0
    missing_value_threshold: float = 0.5   # max fraction of NaN per protein

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def ensure_dirs(self) -> None:
        for d in (self.data_raw_dir, self.data_processed_dir, self.output_dir):
            Path(d).mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    return Settings()
