"""
Test Script for Enhanced GraphQL Agent

Usage:
    # Test with mock (no API calls)
    python scripts/test_agent.py

    # Test with OpenAI
    python scripts/test_agent.py --provider openai --api-key sk-...

    # Test with vLLM server (local Qwen)
    python scripts/test_agent.py --provider vllm --base-url http://localhost:8000/v1
"""

import os
import sys
import asyncio
import argparse
import json
from pathlib import Path
from typing import List, Dict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env file
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from src.llm.llm_client import get_llm, LLMProvider, LLMConfig
from src.agent.enhanced_graphql_agent import EnhancedGraphQLAgent, FEW_SHOT_EXAMPLES


# Test questions covering different query patterns
TEST_QUESTIONS = [
    # Count queries
    {
        "question": "How many indexers are currently active?",
        "expected_entity": "indexers",
        "expected_aggregation": "count",
        "category": "count"
    },
    {
        "question": "What is the total number of delegations?",
        "expected_entity": "delegations",
        "expected_aggregation": "count",
        "category": "count"
    },

    # Superlative queries
    {
        "question": "Which indexer has the highest total stake?",
        "expected_entity": "indexers",
        "expected_aggregation": None,
        "category": "superlative"
    },
    {
        "question": "What is the maximum commission rate among all indexers?",
        "expected_entity": "indexers",
        "expected_aggregation": None,
        "category": "superlative"
    },

    # Entity-specific queries
    {
        "question": "What is the total stake of indexer 0xe60554D90AF0e84A9C3d1A8643e41e49403945a6?",
        "expected_entity": "indexers",
        "expected_aggregation": None,
        "category": "entity_specific"
    },

    # Time-travel queries
    {
        "question": "What was the total stake at block 5000000?",
        "expected_entity": "indexers",
        "expected_aggregation": None,
        "category": "time_travel"
    },

    # Aggregation queries
    {
        "question": "What is the total delegated stake across all delegations?",
        "expected_entity": "delegations",
        "expected_aggregation": "sum",
        "category": "aggregation"
    },

    # Complex queries
    {
        "question": "How many delegators have delegated to indexer 0x1234?",
        "expected_entity": "delegations",
        "expected_aggregation": "count",
        "category": "complex"
    }
]


class MockSchemaIndexer:
    """Mock schema indexer for testing without Qdrant"""

    def __init__(self):
        self.entities = [
            "indexers", "delegations", "delegators", "eras",
            "deployments", "projects", "rewards", "transfers"
        ]

    def get_schema_context(self, query: str, top_k: int = 5) -> str:
        """Return mock schema context"""
        return """
type Indexer {
  id: ID!
  totalStake: BigInt!
  selfStake: BigInt!
  capacity: BigInt!
  commission: Int!
  active: Boolean!
  controller: String
  delegations(first: Int): DelegationsConnection!
}

type Delegation {
  id: ID!
  indexerId: String!
  delegatorId: String!
  amount: BigInt!
  createdBlock: Int!
}

type Delegator {
  id: ID!
  totalDelegated: BigInt!
  delegations(first: Int): DelegationsConnection!
}

type Query {
  indexers(
    first: Int
    filter: IndexerFilter
    orderBy: [IndexersOrderBy!]
    blockHeight: String
  ): IndexersConnection!

  indexerById(id: ID!): Indexer

  delegations(
    first: Int
    filter: DelegationFilter
    orderBy: [DelegationsOrderBy!]
    blockHeight: String
  ): DelegationsConnection!

  delegators(
    first: Int
    filter: DelegatorFilter
    blockHeight: String
  ): DelegatorsConnection!
}
"""

    def get_all_type_names(self) -> List[str]:
        return self.entities


async def test_intent_extraction(agent: EnhancedGraphQLAgent):
    """Test intent extraction from questions"""
    print("\n" + "=" * 60)
    print("Testing Intent Extraction")
    print("=" * 60)

    for test in TEST_QUESTIONS:
        question = test["question"]
        print(f"\nQ: {question}")

        intent = await agent.extract_intent(question)

        print(f"  Entity: {intent.entity}")
        print(f"  Fields: {intent.fields}")
        print(f"  Filters: {intent.filters}")
        print(f"  Aggregation: {intent.aggregation}")
        print(f"  Valid: {intent.is_valid}")

        # Check if expected values match
        if test.get("expected_entity"):
            match = "[OK]" if intent.entity.lower() == test["expected_entity"].lower() else "[FAIL]"
            print(f"  Entity Match: {match}")


async def test_query_generation(agent: EnhancedGraphQLAgent):
    """Test query generation from intents"""
    print("\n" + "=" * 60)
    print("Testing Query Generation")
    print("=" * 60)

    for test in TEST_QUESTIONS:
        question = test["question"]
        print(f"\nQ: {question}")
        print("-" * 40)

        # Get intent
        intent = await agent.extract_intent(question)

        # Get schema context
        schema_context = agent.indexer.get_schema_context(
            query=f"{intent.entity} {' '.join(intent.fields or [])}",
            top_k=3
        )

        # Generate query
        query = await agent.generate_query(
            question=question,
            schema_context=schema_context,
            intent=intent
        )

        print(f"Generated Query:")
        print(query)
        print()


async def test_full_pipeline(agent: EnhancedGraphQLAgent):
    """Test the full pipeline"""
    print("\n" + "=" * 60)
    print("Testing Full Pipeline")
    print("=" * 60)

    # Test without endpoint (just query generation)
    for test in TEST_QUESTIONS[:3]:  # Test first 3 only
        question = test["question"]
        print(f"\nQ: {question}")
        print("-" * 40)

        result = await agent.answer_question(
            question=question,
            endpoint=None,  # No execution
            protocol="subql"
        )

        print(f"Valid: {result['is_valid']}")
        print(f"Intent: {json.dumps(result.get('intent', {}), indent=2)}")
        print(f"Query: {result.get('query', 'N/A')}")

        if result.get('error'):
            print(f"Error: {result['error']}")


def test_llm_connection(llm, provider: str):
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
    print("Enhanced GraphQL Agent Test Suite")
    print("=" * 60)

    # Create LLM client
    print(f"\nInitializing LLM: {args.provider}")
    llm = get_llm(
        provider=args.provider,
        api_key=args.api_key,
        model=args.model,
        base_url=args.base_url
    )

    # Test LLM connection (skip for mock)
    if args.provider != "mock":
        if not test_llm_connection(llm, args.provider):
            print("LLM connection failed. Falling back to mock.")
            llm = get_llm(provider="mock")
    else:
        print("Using mock LLM (no API calls)")

    # Create agent with mock indexer
    indexer = MockSchemaIndexer()
    agent = EnhancedGraphQLAgent(
        schema_indexer=indexer,
        llm_client=llm
    )

    # Run tests
    if args.test == "all" or args.test == "intent":
        await test_intent_extraction(agent)

    if args.test == "all" or args.test == "query":
        await test_query_generation(agent)

    if args.test == "all" or args.test == "pipeline":
        await test_full_pipeline(agent)

    print("\n" + "=" * 60)
    print("Tests Complete!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Test Enhanced GraphQL Agent")

    # LLM Provider options
    parser.add_argument(
        "--provider",
        choices=["openai", "vllm", "mock"],
        default="mock",
        help="LLM provider to use"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for the LLM provider (required for OpenAI)"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (e.g., gpt-4o for OpenAI, models/hermes_miner for vLLM)"
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Base URL for API (for vLLM or custom endpoints)"
    )

    # Test options
    parser.add_argument(
        "--test",
        choices=["all", "intent", "query", "pipeline"],
        default="all",
        help="Which tests to run"
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
