"""
utils/settings.py
-----------------
Pydantic-settings central configuration.
All values can be overridden by environment variables or .env file.
Every module should import `get_settings()` instead of reading env directly.
"""

from functools import lru_cache
from typing import List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Security ──────────────────────────────────────────────────────────────
    orchestrator_api_key: str = Field(
        default="changeme-in-production",
        description="Secret key agents and clients must supply via X-API-Key header.",
    )
    allowed_origins: str = Field(
        default="*",
        description="Comma-separated list of CORS origins, or * for all.",
    )

    # ── Orchestrator ──────────────────────────────────────────────────────────
    orchestrator_host: str = Field(default="0.0.0.0")
    orchestrator_port: int = Field(default=8000)
    scheduler_type: str = Field(
        default="adaptive",
        description="round_robin | least_loaded | adaptive",
    )
    heartbeat_timeout_s: float = Field(default=15.0)
    heartbeat_sweep_s: float = Field(default=5.0)
    registration_grace_s: float = Field(
        default=120.0,
        description="Seconds after registration before heartbeat timeout applies. "
                    "Gives manually registered nodes time to start their agent.",
    )

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = Field(
        default="sqlite:///./gpu_orchestration.db",
        description="SQLAlchemy database URL. Use postgresql:// for production.",
    )
    db_pool_size: int = Field(default=10)
    db_max_overflow: int = Field(default=20)

    # ── Agent ─────────────────────────────────────────────────────────────────
    node_id: str = Field(default="node-1")
    agent_host: Optional[str] = Field(
        default=None,
        description="Advertised host for this agent. Auto-detected if not set.",
    )
    agent_port: int = Field(default=8001)
    orchestrator_url: str = Field(default="http://127.0.0.1:8000")
    heartbeat_interval_s: int = Field(default=5)

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")
    log_format: str = Field(
        default="text",
        description="text | json  — use json in production for structured logs.",
    )

    # ── Training ──────────────────────────────────────────────────────────────
    checkpoint_dir: str = Field(default="./checkpoints")
    data_dir: str = Field(default="./data")

    # ── Adaptive scheduler weights ────────────────────────────────────────────
    adaptive_alpha: float = Field(default=0.5)
    adaptive_beta: float = Field(default=0.4)
    adaptive_gamma: float = Field(default=0.1)

    @field_validator("allowed_origins")
    @classmethod
    def parse_origins(cls, v: str) -> str:
        return v  # stored as raw string; consumed via .get_cors_origins()

    def get_cors_origins(self) -> List[str]:
        """Return CORS origins as a list."""
        if self.allowed_origins == "*":
            return ["*"]
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings singleton."""
    return Settings()
