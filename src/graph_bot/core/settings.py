from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GRAPH_BOT_", env_file=".env", extra="ignore"
    )

    # LLM / embedding provider configs
    llm_provider: str = Field(default="openai")
    llm_model: str = Field(default="gpt-4o-mini")
    embedding_model: str = Field(default="all-MiniLM-L6-v2")

    # GraphRAG DB connection (placeholder)
    graphrag_uri: str = Field(default="sqlite:///graphrag.db")

    # Pipeline hyperparameters
    max_tree_depth: int = Field(default=4)
    beam_width: int = Field(default=3)
    top_k_paths: int = Field(default=3)
    rerank_top_n: int = Field(default=10)


settings = AppSettings()
