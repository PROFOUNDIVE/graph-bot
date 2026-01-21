from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GRAPH_BOT_", env_file=".env", extra="ignore"
    )

    # LLM / embedding provider configs
    llm_provider: str = Field(default="mock") # Default changed to mock for B3 verification
    llm_model: str = Field(default="llama3-8b-instruct")
    llm_base_url: str = Field(default="http://127.0.0.1:2427/v1")
    llm_api_key: str = Field(default="EMPTY")
    llm_temperature: float = Field(default=0.0)
    llm_token_cost_usd_per_1k: float = Field(default=0.0)

    retry_max_attempts: int = Field(default=3)
    retry_temperature_1: float = Field(default=0.0)
    retry_temperature_2: float = Field(default=0.6)
    retry_temperature_3: float = Field(default=0.9)
    embedding_model: str = Field(default="all-MiniLM-L6-v2")

    # GraphRAG DB connection (placeholder)
    graphrag_uri: str = Field(default="sqlite:///graphrag.db")
    metagraph_path: Path = Field(default=Path("outputs/metagraph.json"))

    # Pipeline hyperparameters
    max_tree_depth: int = Field(default=4)
    beam_width: int = Field(default=3)
    top_k_paths: int = Field(default=3)
    rerank_top_n: int = Field(default=10)

    # Update rule defaults (v0.1)
    ema_alpha: float = Field(default=0.1)
    ema_tau_days: int = Field(default=7)
    ema_min_seen: int = Field(default=5)
    ema_min_success: float = Field(default=0.2)

    # Continual stream config (v0.2)
    mode: str = Field(
        default="graph_bot",
        description="Execution mode: graph_bot or flat_template_rag",
    )
    use_edges: bool = Field(
        default=True, description="Use graph edges for path construction"
    )
    policy_id: str = Field(
        default="semantic_topK_stats_rerank",
        description="Selection policy: semantic_only or semantic_topK_stats_rerank",
    )
    validator_mode: str = Field(
        default="oracle",
        description="Validator mode: oracle, exec_repair, or weak_llm_judge",
    )


settings = AppSettings()
