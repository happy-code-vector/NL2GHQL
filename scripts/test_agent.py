"""
Test Script for Enhanced GraphQL Agent with RAG

Uses real test questions from the Hermes subnet dataset.

Requires:
1. Qdrant running: docker run -p 6333:6333 qdrant/qdrant
2. Schema indexed: python -m src.rag.schema_indexer --schema data/schemas/schema_subnet.graphql

Usage:
    # Test with OpenAI (default)
    python scripts/test_agent.py --provider openai

    # Test with limited questions
    python scripts/test_agent.py --provider openai --limit 5

    # Test with custom dataset
    python scripts/test_agent.py --provider openai --dataset my_dataset.json

    # Test with vLLM server (local Qwen)
    python scripts/test_agent.py --provider vllm --base-url http://localhost:8000/v1
"""

import os
import sys
import asyncio
import argparse
import json
import time
from pathlib import Path
from typing import List, Dict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env file
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from src.llm.llm_client import get_llm
from src.rag.schema_indexer import SchemaIndexer
from src.agent.enhanced_graphql_agent import EnhancedGraphQLAgent


def load_dataset(dataset_path: str) -> List[Dict]:
    """Load test questions from dataset file"""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle different dataset formats
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'questions' in data:
        return data['questions']
    else:
        raise ValueError(f"Unknown dataset format in {dataset_path}")


async def test_rag_retrieval(indexer: SchemaIndexer):
    """Test RAG schema retrieval"""
    print("\n" + "=" * 60)
    print("Testing RAG Schema Retrieval")
    print("=" * 60)

    test_queries = [
        "indexer total stake rewards",
        "delegation amount era",
        "project deployment service agreement",
        "consumer query spending"
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = indexer.search(query, top_k=3)

        if results:
            print(f"  Found {len(results)} results:")
            for i, r in enumerate(results[:2], 1):
                print(f"    {i}. {r['name']} (score: {r['score']:.3f})")
        else:
            print("  [FAIL] No results found")


async def test_with_dataset(agent: EnhancedGraphQLAgent, dataset: List[Dict], limit: int = None):
    """Test agent with real dataset questions"""
    print("\n" + "=" * 60)
    print("Testing with Hermes Subnet Questions")
    print("=" * 60)

    questions = dataset[:limit] if limit else dataset
    total = len(questions)

    print(f"\nTesting {total} questions...")

    results = []
    total_time = 0

    for i, item in enumerate(questions, 1):
        question = item.get('question', '')
        original_score = item.get('score', 0)
        original_time = item.get('responseTime', '0s')

        print(f"\n[{i}/{total}] Q: {question[:80]}...")
        print(f"  Original score: {original_score}, Time: {original_time}")

        start_time = time.time()

        try:
            result = await agent.answer_question(
                question=question,
                endpoint=None,  # No execution, just generation
                protocol="subql"
            )

            elapsed = time.time() - start_time
            total_time += elapsed

            is_valid = result.get('is_valid', False)
            query = result.get('query', '')
            error = result.get('error', '')

            status = "[OK]" if is_valid and query else "[FAIL]"
            print(f"  {status} Generated in {elapsed:.2f}s")

            if query:
                # Show first few lines of query
                query_lines = query.strip().split('\n')[:3]
                print(f"  Query preview:")
                for line in query_lines:
                    print(f"    {line}")
                if len(query.strip().split('\n')) > 3:
                    print(f"    ...")

            if error:
                print(f"  Error: {error[:100]}")

            results.append({
                'question': question,
                'original_score': original_score,
                'valid': is_valid,
                'elapsed': elapsed,
                'query': query,
                'error': error
            })

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"  [FAIL] Exception: {str(e)[:100]}")
            results.append({
                'question': question,
                'original_score': original_score,
                'valid': False,
                'elapsed': elapsed,
                'query': '',
                'error': str(e)
            })

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    valid_count = sum(1 for r in results if r['valid'])
    avg_time = total_time / total if total > 0 else 0

    print(f"Total questions: {total}")
    print(f"Valid queries generated: {valid_count}/{total} ({100*valid_count/total:.1f}%)")
    print(f"Average time per question: {avg_time:.2f}s")
    print(f"Total time: {total_time:.2f}s")

    return results


def test_llm_connection(llm, provider: str) -> bool:
    """Test LLM connection"""
    print("\n" + "=" * 60)
    print(f"Testing LLM Connection ({provider})")
    print("=" * 60)

    test_prompt = "Say 'Hello, I am working!' in exactly those words."

    try:
        response = llm.generate(test_prompt)
        print(f"Response: {response}")
        print("[OK] LLM connection successful")
        return True
    except Exception as e:
        print(f"[FAIL] LLM connection failed: {e}")
        return False


async def run_tests(args):
    """Run all tests"""
    print("=" * 60)
    print("Hermes GraphQL Agent Test Suite")
    print("=" * 60)

    # Step 1: Initialize RAG
    print(f"\n[1/3] Initializing RAG with Qdrant: {args.qdrant_url}")
    try:
        indexer = SchemaIndexer(
            qdrant_url=args.qdrant_url,
            collection_name=args.collection,
            embedding_model=args.embedding_model
        )

        type_names = indexer.get_all_type_names()
        print(f"[OK] RAG initialized with {len(type_names)} indexed types")

    except Exception as e:
        print(f"[FAIL] RAG initialization failed: {e}")
        print("\nMake sure:")
        print("  1. Qdrant is running: docker run -p 6333:6333 qdrant/qdrant")
        print("  2. Schema is indexed: python -m src.rag.schema_indexer --schema data/schemas/schema_subnet.graphql")
        return

    # Step 2: Initialize LLM
    print(f"\n[2/3] Initializing LLM: {args.provider}")
    llm = get_llm(
        provider=args.provider,
        api_key=args.api_key,
        model=args.model,
        base_url=args.base_url
    )

    if args.provider != "mock":
        if not test_llm_connection(llm, args.provider):
            print("[FAIL] LLM connection failed. Exiting.")
            return
    else:
        print("Using mock LLM (no API calls)")

    # Step 3: Create agent and run tests
    print(f"\n[3/3] Creating agent and running tests...")
    agent = EnhancedGraphQLAgent(
        schema_indexer=indexer,
        llm_client=llm
    )

    # Load dataset
    print(f"\nLoading dataset from: {args.dataset}")
    try:
        dataset = load_dataset(args.dataset)
        print(f"[OK] Loaded {len(dataset)} questions")
    except Exception as e:
        print(f"[FAIL] Failed to load dataset: {e}")
        return

    # Run tests
    if args.test == "rag":
        await test_rag_retrieval(indexer)
    elif args.test == "dataset":
        await test_with_dataset(agent, dataset, args.limit)
    elif args.test == "all":
        await test_rag_retrieval(indexer)
        await test_with_dataset(agent, dataset, args.limit)
    else:
        # Default: test with dataset
        await test_with_dataset(agent, dataset, args.limit)

    print("\n" + "=" * 60)
    print("Tests Complete!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Test Enhanced GraphQL Agent with RAG")

    # LLM Provider options
    parser.add_argument(
        "--provider",
        choices=["openai", "vllm", "mock"],
        default="openai",
        help="LLM provider to use (default: openai)"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for the LLM provider"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (e.g., gpt-4o for OpenAI)"
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Base URL for vLLM API"
    )

    # RAG options
    parser.add_argument(
        "--qdrant-url",
        default="http://localhost:6333",
        help="Qdrant server URL (default: http://localhost:6333)"
    )
    parser.add_argument(
        "--collection",
        default="graphql_schema",
        help="Qdrant collection name (default: graphql_schema)"
    )
    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="Embedding model for RAG (default: all-MiniLM-L6-v2)"
    )

    # Dataset options
    parser.add_argument(
        "--dataset",
        default="full_dataset.json",
        help="Path to dataset JSON file (default: full_dataset.json)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions to test (default: all)"
    )

    # Test options
    parser.add_argument(
        "--test",
        choices=["all", "rag", "dataset"],
        default="dataset",
        help="Which tests to run (default: dataset)"
    )

    args = parser.parse_args()

    # Set environment variables from args
    if args.api_key:
        if args.provider == "openai":
            os.environ["OPENAI_API_KEY"] = args.api_key

    if args.model:
        os.environ["LLM_MODEL"] = args.model

    if args.base_url:
        os.environ["LLM_BASE_URL"] = args.base_url

    # Run tests
    asyncio.run(run_tests(args))


if __name__ == "__main__":
    main()
