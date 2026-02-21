"""
Hermes Miner Configuration Settings
"""

from dataclasses import dataclass, field
from typing import Optional
import os

@dataclass
class ModelConfig:
    """Model configuration for fine-tuning and inference"""
    # Base model
    base_model: str = "unsloth/Qwen2.5-Coder-7B-Instruct"
    output_dir: str = "models/hermes_miner"

    # Training parameters
    max_seq_length: int = 4096
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    # Training hyperparameters
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 10
    learning_rate: float = 2e-4
    max_steps: int = 500
    logging_steps: int = 1

    # Inference parameters
    temperature: float = 0.1
    top_p: float = 0.95
    max_tokens: int = 512
    gpu_memory_utilization: float = 0.85


@dataclass
class RAGConfig:
    """RAG configuration for schema retrieval"""
    # Vector DB settings
    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "graphql_schema"

    # Embedding model
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Retrieval settings
    top_k_types: int = 5
    top_k_examples: int = 3

    # Schema paths
    schema_dir: str = "data/schemas"


@dataclass
class MinerConfig:
    """Miner configuration for Bittensor integration"""
    # Network settings
    network: str = "finney"
    subnet_uid: int = 82
    wallet_name: str = "miner"
    hotkey_name: str = "default"

    # Performance settings
    max_concurrent_requests: int = 10
    request_timeout: float = 30.0
    max_retries: int = 3

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000


@dataclass
class GraphQLConfig:
    """GraphQL agent configuration"""
    # Supported protocols
    supported_protocols: list = field(default_factory=lambda: ["subql", "the_graph"])

    # Block height parameter names by protocol
    block_height_params: dict = field(default_factory=lambda: {
        "subql": "blockHeight",
        "the_graph": "block"
    })

    # Query execution settings
    query_timeout: float = 10.0
    max_query_depth: int = 5


@dataclass
class Config:
    """Main configuration class"""
    model: ModelConfig = field(default_factory=ModelConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    miner: MinerConfig = field(default_factory=MinerConfig)
    graphql: GraphQLConfig = field(default_factory=GraphQLConfig)

    # Environment
    debug: bool = False
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables"""
        config = cls()

        # Override with environment variables if set
        if os.getenv("QDRANT_URL"):
            config.rag.qdrant_url = os.getenv("QDRANT_URL")
        if os.getenv("MODEL_PATH"):
            config.model.output_dir = os.getenv("MODEL_PATH")
        if os.getenv("API_PORT"):
            config.miner.api_port = int(os.getenv("API_PORT"))
        if os.getenv("DEBUG"):
            config.debug = os.getenv("DEBUG").lower() == "true"

        return config


# Global config instance
config = Config.from_env()
