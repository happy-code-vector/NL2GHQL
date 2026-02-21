"""
Hermes Miner Production Service

FastAPI service that combines:
- vLLM for high-speed inference
- Qdrant for schema RAG
- GraphQL query validation
- Answer generation
"""

import os
import sys
import json
import time
import asyncio
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
from contextlib import asynccontextmanager

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from config.settings import config
from src.rag.schema_indexer import SchemaIndexer


# --- Pydantic Models ---

class GenerateRequest(BaseModel):
    """Request for GraphQL query generation"""
    prompt: str = Field(..., description="Natural language question")
    schema_context: Optional[str] = Field(None, description="Pre-fetched schema context")
    project: Optional[str] = Field(None, description="Project/scope for schema retrieval")
    block_height: Optional[int] = Field(None, description="Block height for time-travel queries")
    protocol: str = Field("subql", description="Protocol type: subql or the_graph")


class GenerateResponse(BaseModel):
    """Response from GraphQL query generation"""
    query: str
    valid: bool
    schema_context: str
    error: Optional[str] = None


class AnswerRequest(BaseModel):
    """Request for full answer generation"""
    question: str = Field(..., description="Natural language question")
    endpoint: str = Field(..., description="GraphQL API endpoint URL")
    project: Optional[str] = Field(None, description="Project/scope identifier")
    block_height: Optional[int] = Field(None, description="Block height for time-travel")
    protocol: str = Field("subql", description="Protocol type")


class AnswerResponse(BaseModel):
    """Full answer response"""
    answer: str
    query: Optional[str] = None
    query_result: Optional[Dict] = None
    valid: bool = True
    error: Optional[str] = None
    elapsed_time: float = 0.0


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    rag_ready: bool
    version: str = "1.0.0"


# --- Global State ---

class MinerState:
    """Global miner state"""
    llm = None
    sampling_params = None
    indexer: Optional[SchemaIndexer] = None
    schema_cache: Dict[str, str] = {}


state = MinerState()


# --- LLM Client ---

class VLLMClient:
    """vLLM client wrapper"""

    def __init__(self, model_path: str, gpu_memory_utilization: float = 0.85):
        from vllm import LLM, SamplingParams

        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype="float16",
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.95,
            max_tokens=512,
            stop=["###", "<|im_end|>", "\n\n\n"]
        )

    def generate(self, prompt: str) -> str:
        """Generate text from prompt"""
        outputs = self.llm.generate([prompt], self.sampling_params)
        return outputs[0].outputs[0].text.strip()

    async def generate_async(self, prompt: str) -> str:
        """Async generate (runs in thread pool)"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, prompt)


class MockLLMClient:
    """Mock LLM client for testing without GPU"""

    def __init__(self):
        print("Warning: Using MockLLMClient - no actual inference")

    def generate(self, prompt: str) -> str:
        # Return a sample GraphQL query for testing
        return """query {
  indexers(first: 10) {
    nodes {
      id
      totalStake
      selfStake
    }
    totalCount
  }
}"""

    async def generate_async(self, prompt: str) -> str:
        return self.generate(prompt)


# --- Prompt Templates ---

QUERY_GEN_PROMPT = """You are an expert GraphQL query generator for blockchain data.
Given a natural language question and relevant schema context, generate a valid GraphQL query.

IMPORTANT RULES:
1. Only use types and fields that exist in the provided schema
2. For SubQL endpoints, use blockHeight parameter: `fieldName(blockHeight: "NUMBER")`
3. For The Graph endpoints, use block parameter: `fieldName(block: {{number: NUMBER}})`
4. Always include the required parameters
5. For count questions, use totalCount or count patterns
6. For "highest/lowest" questions, use orderBy and limit: 1
7. Return ONLY the GraphQL query, no explanations or markdown

Schema Context:
{schema_context}

Protocol: {protocol}
{block_instruction}

Question: {question}

GraphQL Query:"""

ANSWER_GEN_PROMPT = """You are a helpful assistant that answers questions about blockchain data.

Question: {question}

GraphQL Query Used:
{query}

Query Result:
{result}

Provide a concise, direct answer to the question based on the query result.
Focus on the specific numerical value or entity requested.
If the result is empty, say "No data found".
Do not include explanations about the query or data source.

Answer:"""


# --- Application Lifespan ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    print("=" * 60)
    print("Hermes Miner Service Starting...")
    print("=" * 60)

    # Initialize RAG
    print("\n[1/2] Initializing RAG system...")
    try:
        state.indexer = SchemaIndexer(
            qdrant_url=config.rag.qdrant_url,
            collection_name=config.rag.collection_name,
            embedding_model=config.rag.embedding_model,
            embedding_dim=config.rag.embedding_dim
        )
        print(f"RAG initialized. Qdrant URL: {config.rag.qdrant_url}")
    except Exception as e:
        print(f"Warning: RAG initialization failed: {e}")
        print("Service will run without RAG (schema context must be provided)")

    # Initialize LLM
    print("\n[2/2] Loading LLM...")
    model_path = config.model.output_dir
    if os.path.exists(model_path):
        try:
            state.llm = VLLMClient(
                model_path=model_path,
                gpu_memory_utilization=config.model.gpu_memory_utilization
            )
            print(f"vLLM loaded from: {model_path}")
        except Exception as e:
            print(f"Warning: vLLM loading failed: {e}")
            print("Falling back to mock LLM client")
            state.llm = MockLLMClient()
    else:
        print(f"Model not found at: {model_path}")
        print("Using mock LLM client for testing")
        state.llm = MockLLMClient()

    print("\n" + "=" * 60)
    print("Hermes Miner Service Ready!")
    print(f"API: http://{config.miner.api_host}:{config.miner.api_port}")
    print("=" * 60)

    yield

    # Shutdown
    print("\nShutting down...")


# --- FastAPI App ---

app = FastAPI(
    title="Hermes Miner Service",
    description="NL2GraphQL service for Bittensor Subnet 82",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Helper Functions ---

def get_schema_context(prompt: str, project: Optional[str] = None) -> str:
    """Get relevant schema context for a prompt"""
    if state.indexer is None:
        return ""

    # Check cache first
    cache_key = f"{project}:{prompt[:50]}"
    if cache_key in state.schema_cache:
        return state.schema_cache[cache_key]

    try:
        context = state.indexer.get_schema_context(
            query=prompt,
            top_k=config.rag.top_k_types,
            project=project
        )
        # Cache for future use
        state.schema_cache[cache_key] = context
        return context
    except Exception as e:
        print(f"RAG retrieval error: {e}")
        return ""


def validate_graphql_query(query: str) -> tuple[bool, Optional[str]]:
    """Validate a GraphQL query syntax"""
    from graphql import parse, GraphQLSyntaxError

    try:
        parse(query)
        return True, None
    except GraphQLSyntaxError as e:
        return False, f"Syntax error: {e.message}"
    except Exception as e:
        return False, str(e)


def extract_query(response: str) -> str:
    """Extract GraphQL query from LLM response"""
    import re

    # Try code blocks first
    code_block_pattern = r'```(?:graphql|gql)?\s*([\s\S]*?)```'
    matches = re.findall(code_block_pattern, response)
    if matches:
        return matches[0].strip()

    # Try to find query pattern
    query_pattern = r'(query\s+\w*\s*\{[\s\S]*\}|\{[\s\S]*\})'
    match = re.search(query_pattern, response)
    if match:
        return match.group(1).strip()

    return response.strip()


# --- API Endpoints ---

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if state.llm else "degraded",
        model_loaded=state.llm is not None,
        rag_ready=state.indexer is not None
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate_query(request: GenerateRequest):
    """Generate a GraphQL query from natural language"""
    start_time = time.time()

    try:
        # Get schema context
        if request.schema_context:
            schema_context = request.schema_context
        else:
            schema_context = get_schema_context(request.prompt, request.project)

        if not schema_context:
            return GenerateResponse(
                query="",
                valid=False,
                schema_context="",
                error="No schema context available. Please provide schema_context or ensure RAG is initialized."
            )

        # Build block height instruction
        block_instruction = ""
        if request.block_height:
            if request.protocol == "subql":
                block_instruction = f"Use blockHeight: \"{request.block_height}\" for time-travel query."
            else:
                block_instruction = f"Use block: {{number: {request.block_height}}} for time-travel query."

        # Build prompt
        prompt = QUERY_GEN_PROMPT.format(
            schema_context=schema_context,
            protocol=request.protocol,
            block_instruction=block_instruction,
            question=request.prompt
        )

        # Generate query
        response = await state.llm.generate_async(prompt)
        query = extract_query(response)

        # Validate query
        is_valid, error = validate_graphql_query(query)

        return GenerateResponse(
            query=query,
            valid=is_valid,
            schema_context=schema_context[:200] + "..." if len(schema_context) > 200 else schema_context,
            error=error
        )

    except Exception as e:
        return GenerateResponse(
            query="",
            valid=False,
            schema_context="",
            error=str(e)
        )


@app.post("/answer", response_model=AnswerResponse)
async def answer_question(request: AnswerRequest):
    """Answer a natural language question with full execution"""
    import httpx

    start_time = time.time()

    try:
        # Step 1: Generate query
        gen_request = GenerateRequest(
            prompt=request.question,
            project=request.project,
            block_height=request.block_height,
            protocol=request.protocol
        )
        gen_response = await generate_query(gen_request)

        if not gen_response.valid or not gen_response.query:
            return AnswerResponse(
                answer="",
                query=gen_response.query,
                valid=False,
                error=gen_response.error,
                elapsed_time=time.time() - start_time
            )

        query = gen_response.query

        # Step 2: Execute query
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.post(
                    request.endpoint,
                    json={"query": query},
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                result = response.json()

                if "errors" in result:
                    return AnswerResponse(
                        answer="",
                        query=query,
                        valid=False,
                        error=f"GraphQL error: {result['errors']}",
                        elapsed_time=time.time() - start_time
                    )

                query_result = result.get("data")

            except httpx.TimeoutException:
                return AnswerResponse(
                    answer="",
                    query=query,
                    valid=False,
                    error="Query execution timed out",
                    elapsed_time=time.time() - start_time
                )
            except Exception as e:
                return AnswerResponse(
                    answer="",
                    query=query,
                    valid=False,
                    error=f"Query execution failed: {e}",
                    elapsed_time=time.time() - start_time
                )

        # Step 3: Generate answer
        answer_prompt = ANSWER_GEN_PROMPT.format(
            question=request.question,
            query=query,
            result=json.dumps(query_result, indent=2)
        )
        answer = await state.llm.generate_async(answer_prompt)

        return AnswerResponse(
            answer=answer,
            query=query,
            query_result=query_result,
            valid=True,
            elapsed_time=time.time() - start_time
        )

    except Exception as e:
        return AnswerResponse(
            answer="",
            valid=False,
            error=str(e),
            elapsed_time=time.time() - start_time
        )


@app.post("/index-schema")
async def index_schema(schema_file: str, project: Optional[str] = None):
    """Index a GraphQL schema file into Qdrant"""
    if state.indexer is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    if not os.path.exists(schema_file):
        raise HTTPException(status_code=404, detail=f"Schema file not found: {schema_file}")

    try:
        state.indexer.create_collection(recreate=False)
        count = state.indexer.index_schema_file(schema_file, project)
        return {"status": "success", "indexed_count": count, "project": project}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Main Entry Point ---

def main():
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Hermes Miner Service")
    parser.add_argument("--host", default=config.miner.api_host, help="API host")
    parser.add_argument("--port", type=int, default=config.miner.api_port, help="API port")
    parser.add_argument("--model-path", default=config.model.output_dir, help="Model path")
    parser.add_argument("--qdrant-url", default=config.rag.qdrant_url, help="Qdrant URL")

    args = parser.parse_args()

    # Update config
    config.miner.api_host = args.host
    config.miner.api_port = args.port
    config.model.output_dir = args.model_path
    config.rag.qdrant_url = args.qdrant_url

    # Run server
    uvicorn.run(
        "miner_service:app",
        host=args.host,
        port=args.port,
        reload=False,
        workers=1
    )


if __name__ == "__main__":
    main()
