"""
Hermes Miner Configuration Settings
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
import os


@dataclass
class ModelConfig:
    """Model configuration for inference"""
    # LLM Provider (gemini, openai, vllm, mock)
    provider: str = "gemini"
    model: str = "gemini-2.5-flash"
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    # Generation parameters
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout: float = 60.0

    @classmethod
    def from_env(cls) -> "ModelConfig":
        """Load from environment variables."""
        return cls(
            provider=os.getenv("LLM_PROVIDER", "gemini"),
            model=os.getenv("GEMINI_MODEL", os.getenv("OPENAI_MODEL", "gemini-2.5-flash")),
            api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048")),
        )


@dataclass
class GraphQLConfig:
    """GraphQL configuration"""
    # Endpoint settings
    endpoint: str = ""
    api_key: Optional[str] = None
    auth_type: Literal["bearer", "apikey", "none"] = "bearer"

    # Supported protocols
    supported_protocols: list = field(default_factory=lambda: ["subql", "thegraph"])

    # Block height parameter names by protocol
    block_height_params: dict = field(default_factory=lambda: {
        "subql": "blockHeight",
        "thegraph": "block"
    })

    # Query execution settings
    query_timeout: float = 30.0
    max_query_depth: int = 5

    @classmethod
    def from_env(cls) -> "GraphQLConfig":
        """Load from environment variables."""
        return cls(
            endpoint=os.getenv("GRAPHQL_ENDPOINT", ""),
            api_key=os.getenv("GRAPHQL_API_KEY"),
            auth_type=os.getenv("GRAPHQL_AUTH_TYPE", "bearer"),
        )


@dataclass
class AgentConfig:
    """Agent configuration"""
    # ReAct agent settings
    max_retries: int = 3
    recursion_limit: int = 12

    # Caching
    enable_cache: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Load from environment variables."""
        return cls(
            max_retries=int(os.getenv("AGENT_MAX_RETRIES", "3")),
            recursion_limit=int(os.getenv("AGENT_RECURSION_LIMIT", "12")),
            enable_cache=os.getenv("ENABLE_CACHE", "true").lower() == "true",
        )


@dataclass
class RAGConfig:
    """RAG configuration for schema retrieval (optional)"""
    # Vector DB settings
    weaviate_url: str = "http://localhost:8080"
    collection_name: str = "graphql_schema"

    # Embedding model
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Retrieval settings
    top_k_types: int = 5
    top_k_examples: int = 3

    @classmethod
    def from_env(cls) -> "RAGConfig":
        """Load from environment variables."""
        return cls(
            weaviate_url=os.getenv("WEAVIATE_URL", "http://localhost:8080"),
            collection_name=os.getenv("WEAVIATE_COLLECTION", "graphql_schema"),
        )


@dataclass
class Config:
    """Main configuration class"""
    model: ModelConfig = field(default_factory=ModelConfig.from_env)
    graphql: GraphQLConfig = field(default_factory=GraphQLConfig.from_env)
    agent: AgentConfig = field(default_factory=AgentConfig.from_env)
    rag: RAGConfig = field(default_factory=RAGConfig.from_env)

    # Environment
    debug: bool = False
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables"""
        config = cls()

        # Override with environment variables if set
        if os.getenv("DEBUG"):
            config.debug = os.getenv("DEBUG").lower() == "true"
        if os.getenv("LOG_LEVEL"):
            config.log_level = os.getenv("LOG_LEVEL")

        return config

    def validate(self):
        """Validate required configuration."""
        errors = []

        if not self.graphql.endpoint:
            errors.append("GRAPHQL_ENDPOINT is required")

        if self.model.provider == "gemini" and not self.model.api_key:
            if not os.getenv("GOOGLE_API_KEY"):
                errors.append("GOOGLE_API_KEY is required for Gemini provider")

        if self.model.provider == "openai" and not self.model.api_key:
            if not os.getenv("OPENAI_API_KEY"):
                errors.append("OPENAI_API_KEY is required for OpenAI provider")

        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")

        return True


# Global config instance
config = Config.from_env()
