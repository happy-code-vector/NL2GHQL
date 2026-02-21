# NL2GHHQL - Hermes Miner for Bittensor Subnet 82

A high-performance **Natural Language to GraphQL** system for the Hermes Subnet (Bittensor Subnet 82). This project implements a competitive miner that converts natural language questions into valid GraphQL queries and answers.

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Bittensor      │     │   FastAPI    │     │     vLLM        │
│  Validator      │────▶│   Miner      │────▶│   Qwen2.5-Coder │
│  (Challenge)    │     │   Service    │     │   7B Instruct   │
└─────────────────┘     └──────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌──────────────┐
                        │   Qdrant     │
                        │   (RAG)      │
                        └──────────────┘
                               │
                               ▼
                        ┌──────────────┐
                        │  GraphQL     │
                        │  Endpoint    │
                        └──────────────┘
```

## Components

| Component | Description |
|-----------|-------------|
| `schema_indexer.py` | Parses GraphQL schemas and indexes into Qdrant |
| `graphql_agent.py` | RAG-powered agent for query generation |
| `dataset_generator.py` | Creates synthetic training data |
| `train_miner.py` | Unsloth fine-tuning script |
| `miner_service.py` | FastAPI production service with vLLM |
| `bittensor_miner.py` | Bittensor subnet integration |

## Quick Start

### 1. Prerequisites

- Python 3.10+
- CUDA-capable GPU (RTX 3090/4090 recommended, 24GB VRAM)
- Docker (for Qdrant)

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/nl2ghhql.git
cd nl2ghhql

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install Unsloth (for training)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### 3. Start Qdrant

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 4. Index Schema

```bash
python -m src.rag.schema_indexer \
    --schema path/to/blockchain_schema.graphql \
    --project my-blockchain
```

### 5. Generate Training Data

```bash
python -m src.training.dataset_generator \
    --schema path/to/blockchain_schema.graphql \
    --output data/datasets/train_dataset.jsonl \
    --count 2000
```

### 6. Fine-tune Model

```bash
python -m src.training.train_miner \
    --dataset data/datasets/train_dataset.jsonl \
    --output models/hermes_miner \
    --max-steps 500
```

### 7. Run Miner Service

```bash
python -m src.hermes_miner.miner_service \
    --host 0.0.0.0 \
    --port 8000 \
    --model-path models/hermes_miner
```

### 8. Run Bittensor Miner

```bash
python -m src.hermes_miner.bittensor_miner \
    --network finney \
    --wallet-name miner \
    --agent-url http://localhost:8000
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/generate` | POST | Generate GraphQL query from NL |
| `/answer` | POST | Full answer with query execution |
| `/index-schema` | POST | Index a new schema |

### Example Request

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is the total stake of all indexers?",
    "protocol": "subql"
  }'
```

### Example Response

```json
{
  "query": "query { indexers(first: 10) { nodes { id totalStake } totalCount } }",
  "valid": true,
  "schema_context": "type Indexer { id: ID!, totalStake: BigInt...",
  "error": null
}
```

## Configuration

Environment variables can be used to override defaults:

| Variable | Description | Default |
|----------|-------------|---------|
| `QDRANT_URL` | Qdrant server URL | `http://localhost:6333` |
| `MODEL_PATH` | Fine-tuned model path | `models/hermes_miner` |
| `API_PORT` | Miner API port | `8000` |
| `DEBUG` | Enable debug mode | `false` |

## Hermes Subnet Specifics

The Hermes subnet evaluates miners on:

1. **Accuracy** (Fact Score 0-10): How correct is the answer?
2. **Latency** (Time Weight): Quadratic penalty for slow responses
3. **Schema Compliance**: Queries must follow the schema

### Supported Query Types

- **Count queries**: "How many indexers are active?"
- **Superlative queries**: "Which indexer has the highest stake?"
- **Entity queries**: "What is the commission rate of indexer 0xABC...?"
- **Time-travel queries**: "What was the stake at block 5000000?"

### Block Height Parameters

```graphql
# SubQL format
indexers(blockHeight: "5000000") { nodes { id } }

# The Graph format
swaps(block: {number: 5000000}) { id }
```

## Performance Optimization

### For RTX 3090 (24GB VRAM)

- Use 4-bit quantization for training
- Use 16-bit for inference
- Enable Flash Attention 2
- Expected latency: 300-800ms per query

### Caching Strategies

1. **Semantic Caching**: Cache RAG results for similar queries
2. **Schema Caching**: Cache retrieved schema contexts
3. **Query Caching**: Cache frequent query patterns

## Project Structure

```
nl2ghhql/
├── config/
│   ├── __init__.py
│   └── settings.py          # Configuration classes
├── src/
│   ├── rag/
│   │   ├── __init__.py
│   │   └── schema_indexer.py # Schema parsing & Qdrant indexing
│   ├── agent/
│   │   ├── __init__.py
│   │   └── graphql_agent.py  # RAG-powered GraphQL agent
│   ├── training/
│   │   ├── __init__.py
│   │   ├── dataset_generator.py # Training data generation
│   │   └── train_miner.py    # Unsloth fine-tuning
│   └── hermes_miner/
│       ├── __init__.py
│       ├── miner_service.py  # FastAPI + vLLM service
│       └── bittensor_miner.py # Bittensor integration
├── data/
│   ├── schemas/              # GraphQL schema files
│   └── datasets/             # Training datasets
├── models/                   # Fine-tuned models
├── requirements.txt
└── README.md
```

## Datasets

### Recommended Training Datasets

| Dataset | Size | Description |
|---------|------|-------------|
| StepZen NL2GQL | ~1,700 pairs | Manually validated NL-GQL pairs |
| IBM NL2GQL | ~10,000 triples | Cross-domain benchmark |
| StockGQL | Variable | Financial graph queries |

### Custom Dataset Format

```json
{
  "instruction": "What is the total stake of all indexers?",
  "input": "type Indexer { id: ID!, totalStake: BigInt, selfStake: BigInt }",
  "output": "query { indexers(first: 10) { nodes { id totalStake } totalCount } }"
}
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use 4-bit quantization
   - Reduce max_seq_length

2. **Qdrant Connection Error**
   - Ensure Qdrant is running: `docker ps`
   - Check URL configuration

3. **Slow Inference**
   - Enable Flash Attention 2
   - Use vLLM instead of raw Transformers
   - Consider model quantization (AWQ/GPTQ)

## License

MIT License

## Contributing

Contributions welcome! Please read the contributing guidelines first.

## Resources

- [Hermes Subnet GitHub](https://github.com/SN-Hermes/hermes-subnet)
- [Bittensor Documentation](https://docs.bittensor.com)
- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [vLLM Documentation](https://docs.vllm.ai)
