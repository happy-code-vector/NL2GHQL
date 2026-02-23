"""
Compare Qdrant vs Weaviate for GraphQL Schema RAG

This script tests both vector databases with the same queries and compares:
- Retrieval accuracy
- Search latency
- Memory usage
- Ease of use

Usage:
    python scripts/compare_vector_dbs.py --schema data/schemas/schema_subnet.graphql
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")


# Test queries
TEST_QUERIES = [
    "indexer total stake rewards",
    "delegation amount era",
    "project deployment service agreement",
    "consumer query spending",
    "era reward claims",
    "totalRevenue field",
    "network operator commission",
    "delegator unclaimed rewards"
]


def test_qdrant(schema_path: str, project: str = "subquery") -> Dict:
    """Test Qdrant with hybrid search"""
    print("\n" + "=" * 60)
    print("Testing Qdrant with Hybrid Search")
    print("=" * 60)

    try:
        from src.rag.hybrid_indexer import HybridSchemaIndexer, parse_schema_file

        indexer = HybridSchemaIndexer(
            qdrant_url="http://localhost:6333",
            collection_name="graphql_schema_compare"
        )

        # Create and index
        start = time.time()
        indexer.create_collection(recreate=True)

        chunks = parse_schema_file(schema_path, project)
        indexer.index_chunks(chunks)
        index_time = time.time() - start
        print(f"Indexed {len(chunks)} chunks in {index_time:.2f}s")

        # Test searches
        results = []
        total_search_time = 0

        for query in TEST_QUERIES:
            start = time.time()
            search_results = indexer.hybrid_search(query, top_k=3, alpha=0.5)
            search_time = time.time() - start
            total_search_time += search_time

            results.append({
                "query": query,
                "time": search_time,
                "results": [(r["name"], r["score"]) for r in search_results]
            })
            print(f"  '{query[:30]}...' -> {search_time*1000:.1f}ms")

        return {
            "success": True,
            "index_time": index_time,
            "avg_search_time": total_search_time / len(TEST_QUERIES),
            "results": results
        }

    except Exception as e:
        print(f"[FAIL] Qdrant test failed: {e}")
        return {"success": False, "error": str(e)}


def test_weaviate(schema_path: str, project: str = "subquery") -> Dict:
    """Test Weaviate with native hybrid search"""
    print("\n" + "=" * 60)
    print("Testing Weaviate with Native Hybrid Search")
    print("=" * 60)

    try:
        from src.rag.weaviate_indexer import WeaviateSchemaIndexer, parse_schema_file

        indexer = WeaviateSchemaIndexer(
            weaviate_url="http://localhost:8080",
            use_openai_vectorizer=False  # Use local embeddings
        )

        # Create and index
        start = time.time()
        indexer.create_collection()

        chunks = parse_schema_file(schema_path, project)
        indexer.index_chunks(chunks)
        index_time = time.time() - start
        print(f"Indexed {len(chunks)} chunks in {index_time:.2f}s")

        # Test searches
        results = []
        total_search_time = 0

        for query in TEST_QUERIES:
            start = time.time()
            search_results = indexer.hybrid_search(query, top_k=3, alpha=0.5)
            search_time = time.time() - start
            total_search_time += search_time

            results.append({
                "query": query,
                "time": search_time,
                "results": [(r["name"], r["score"]) for r in search_results]
            })
            print(f"  '{query[:30]}...' -> {search_time*1000:.1f}ms")

        indexer.close()

        return {
            "success": True,
            "index_time": index_time,
            "avg_search_time": total_search_time / len(TEST_QUERIES),
            "results": results
        }

    except Exception as e:
        print(f"[FAIL] Weaviate test failed: {e}")
        return {"success": False, "error": str(e)}


def compare_results(qdrant_result: Dict, weaviate_result: Dict):
    """Compare and display results"""
    print("\n" + "=" * 60)
    print("Comparison Results")
    print("=" * 60)

    if not qdrant_result.get("success"):
        print(f"Qdrant: FAILED - {qdrant_result.get('error')}")
    else:
        print(f"Qdrant: Avg search time = {qdrant_result['avg_search_time']*1000:.1f}ms")

    if not weaviate_result.get("success"):
        print(f"Weaviate: FAILED - {weaviate_result.get('error')}")
    else:
        print(f"Weaviate: Avg search time = {weaviate_result['avg_search_time']*1000:.1f}ms")

    # Compare result quality
    if qdrant_result.get("success") and weaviate_result.get("success"):
        print("\n--- Search Result Comparison ---")

        for i, query in enumerate(TEST_QUERIES):
            print(f"\nQuery: {query}")
            print(f"  Qdrant:   {qdrant_result['results'][i]['results'][:3]}")
            print(f"  Weaviate: {weaviate_result['results'][i]['results'][:3]}")


def main():
    parser = argparse.ArgumentParser(description="Compare Qdrant vs Weaviate")
    parser.add_argument("--schema", required=True, help="Path to GraphQL schema")
    parser.add_argument("--project", default="subquery", help="Project name")
    parser.add_argument("--skip-qdrant", action="store_true", help="Skip Qdrant test")
    parser.add_argument("--skip-weaviate", action="store_true", help="Skip Weaviate test")

    args = parser.parse_args()

    qdrant_result = {}
    weaviate_result = {}

    if not args.skip_qdrant:
        qdrant_result = test_qdrant(args.schema, args.project)

    if not args.skip_weaviate:
        weaviate_result = test_weaviate(args.schema, args.project)

    compare_results(qdrant_result, weaviate_result)

    print("\n" + "=" * 60)
    print("Recommendation")
    print("=" * 60)

    if qdrant_result.get("success") and weaviate_result.get("success"):
        q_time = qdrant_result["avg_search_time"]
        w_time = weaviate_result["avg_search_time"]

        if q_time < w_time:
            print(f"Qdrant is {w_time/q_time:.1f}x faster")
        else:
            print(f"Weaviate is {q_time/w_time:.1f}x faster")

        print("\nBoth are working! Choose based on:")
        print("  - Qdrant: Lower memory, more control, slightly more setup")
        print("  - Weaviate: Native GraphQL, built-in RAG, easier setup")
    else:
        print("One or both failed. Check the errors above.")


if __name__ == "__main__":
    main()
