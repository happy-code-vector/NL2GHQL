#!/usr/bin/env python3
"""
Batch test script for RAG Agent
Runs all questions from full_dataset.json with concurrent execution

Usage:
    python batch_test.py                    # Default 5 workers
    python batch_test.py --workers 10       # 10 parallel workers
    python batch_test.py -w 3               # 3 parallel workers
"""

import argparse
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any

load_dotenv()


async def process_question(
    agent,
    question: str,
    item_id: str,
    original_score: int,
    semaphore: asyncio.Semaphore,
    index: int,
    total: int,
) -> Dict[str, Any]:
    """Process a single question with semaphore for concurrency control."""
    async with semaphore:
        print(f"[{index}/{total}] Starting: {question[:60]}...")

        start_time = time.perf_counter()

        try:
            result = await agent.query(question)
            latency = time.perf_counter() - start_time

            # Build clean result object
            output = {
                "id": item_id,
                "question": question,
                "graphql_query": result.get("graphql_query"),
                "query_result": result.get("query_result"),
                "answer": result.get("answer"),
                "success": result.get("success", False),
                "error": result.get("error"),
                "latency_ms": int(latency * 1000),
                "original_score": original_score,
            }

            if result.get("success"):
                print(f"[{index}/{total}] ✓ Success in {latency:.2f}s")
            else:
                print(f"[{index}/{total}] ✗ Failed: {result.get('error', 'Unknown')[:50]}")

            return output

        except Exception as e:
            latency = time.perf_counter() - start_time
            print(f"[{index}/{total}] ✗ Exception: {str(e)[:50]}")

            return {
                "id": item_id,
                "question": question,
                "graphql_query": None,
                "query_result": None,
                "answer": None,
                "success": False,
                "error": str(e),
                "latency_ms": int(latency * 1000),
                "original_score": original_score,
            }


async def run_batch_test(workers: int):
    """Run batch test on all questions in full_dataset.json with concurrency."""
    from src.agent import ReActGraphQLAgent
    from src.llm.llm_client import LLMConfig
    from config.settings import Config

    # Load config
    config = Config.from_env()

    # Check endpoint
    endpoint = config.graphql.endpoint
    if not endpoint:
        print("Error: GRAPHQL_ENDPOINT not set in .env")
        return

    print(f"Endpoint: {endpoint}")
    print(f"Workers: {workers}")

    # Setup headers
    headers = {"Content-Type": "application/json"}
    if config.graphql.api_key:
        if config.graphql.auth_type == "bearer":
            headers["Authorization"] = f"Bearer {config.graphql.api_key}"
        elif config.graphql.auth_type == "apikey":
            headers["X-API-Key"] = config.graphql.api_key

    # Create LLM config
    llm_config = LLMConfig(
        provider=config.model.provider,
        model=config.model.model,
        api_key=config.model.api_key,
        temperature=config.model.temperature,
        max_tokens=config.model.max_tokens,
    )

    # Create agent
    print("Initializing agent...")
    agent = ReActGraphQLAgent(
        endpoint=endpoint,
        headers=headers,
        llm_config=llm_config,
        max_retries=config.agent.max_retries,
        enable_cache=config.agent.enable_cache,
    )

    # Load dataset
    dataset_path = Path("full_dataset.json")
    if not dataset_path.exists():
        print("Error: full_dataset.json not found")
        return

    with open(dataset_path) as f:
        dataset = json.load(f)

    total_questions = len(dataset)
    print(f"\nLoaded {total_questions} questions")
    print("=" * 70)

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(workers)

    # Create all tasks
    tasks = []
    for i, item in enumerate(dataset, 1):
        question = item.get("question", "")
        if not question:
            continue

        task = process_question(
            agent=agent,
            question=question,
            item_id=item.get("id"),
            original_score=item.get("score"),
            semaphore=semaphore,
            index=i,
            total=total_questions,
        )
        tasks.append(task)

    # Run all tasks concurrently
    start_time = time.perf_counter()
    print(f"\nStarting batch processing with {workers} workers...")
    print("-" * 70)

    results = await asyncio.gather(*tasks)

    total_time = time.perf_counter() - start_time

    # Calculate stats
    successful = sum(1 for r in results if r.get("success"))
    failed = len(results) - successful
    avg_latency = sum(r.get("latency_ms", 0) for r in results) / len(results) / 1000 if results else 0

    # Save results
    output_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_data = {
        "summary": {
            "total": len(results),
            "successful": successful,
            "failed": failed,
            "success_rate": round(successful / len(results) * 100, 1) if results else 0,
            "avg_latency_ms": round(avg_latency * 1000, 0),
            "total_time_ms": round(total_time * 1000, 0),
            "workers": workers,
            "timestamp": datetime.now().isoformat(),
        },
        "results": results,
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(json.dumps(output_data["summary"], indent=2))
    print(f"\nResults saved to: {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Batch test RAG Agent with concurrent execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python batch_test.py              # Default 5 workers
    python batch_test.py -w 10        # 10 parallel workers
    python batch_test.py --workers 3  # 3 parallel workers
        """,
    )

    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=5,
        help="Number of parallel workers (default: 5)"
    )

    args = parser.parse_args()

    asyncio.run(run_batch_test(args.workers))


if __name__ == "__main__":
    main()
