#!/usr/bin/env python3
"""
RAG Agent - NL2GraphQL with ReAct Pattern

Usage:
    python main.py                                          # Interactive mode
    python main.py --question "What is the total stake?"    # Single question
    python main.py --file questions.txt                     # Batch from file
    python main.py --block-height 5000000 --question "..."  # Time-travel query
"""

import asyncio
import argparse
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def print_result(result: dict, show_details: bool = True):
    """Pretty print a result."""
    print("\n" + "=" * 70)
    print(f"Question: {result.get('question', 'N/A')}")
    if result.get("retries", 0) > 0:
        print(f"Retries: {result.get('retries')}")
    print("=" * 70)

    if result.get("success"):
        if show_details:
            print(f"\nGraphQL Query:")
            print("-" * 40)
            query = result.get("graphql_query")
            if query:
                print(query)
            else:
                print("(No GraphQL query generated)")

            print(f"\nQuery Result:")
            print("-" * 40)
            query_result = result.get("query_result")
            if query_result:
                print(json.dumps(query_result, indent=2, default=str)[:500])
            else:
                print("(No result)")

        print(f"\nAnswer:")
        print("-" * 40)
        answer = result.get("answer")
        if answer:
            print(answer)
        else:
            print("(No answer generated)")

        tools = result.get("tool_calls", [])
        if tools:
            print(f"\nTools Used: {tools}")

        print(f"\nLatency: {result.get('latency_ms', 0)}ms")

    else:
        print(f"\nError: {result.get('error', 'Unknown error')}")

    print("=" * 70)


async def interactive_mode(agent, args):
    """Run in interactive mode."""
    print("\nRAG Agent - NL2GraphQL with ReAct Pattern")
    print("Commands: 'quit' | 'cache' | 'clear'")
    print("-" * 50)

    while True:
        try:
            question = input("\nQuestion: ").strip()

            if not question:
                continue

            if question.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if question.lower() == "cache":
                print(f"Schema cache: {'enabled' if agent.client._cache.enabled else 'disabled'}")
                continue

            if question.lower() == "clear":
                agent.clear_cache()
                print("Cache cleared!")
                continue

            # Optional block height
            block_height = args.block_height
            if not block_height:
                bh_input = input("Block height (press Enter to skip): ").strip()
                if bh_input:
                    try:
                        block_height = int(bh_input)
                    except ValueError:
                        print("Invalid block height, skipping...")

            result = await agent.query(question, block_height)
            print_result(result, show_details=not args.quiet)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


async def single_question(agent, args):
    """Process a single question."""
    result = await agent.query(args.question, args.block_height)
    print_result(result, show_details=not args.quiet)
    return result


async def batch_from_file(agent, args):
    """Process questions from a file."""
    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {args.file}")
        return []

    questions = [line.strip() for line in path.read_text().split("\n") if line.strip()]

    print(f"\nProcessing {len(questions)} questions from {args.file}")
    print("=" * 70)

    results = []
    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] Processing...")
        result = await agent.query(question, args.block_height)
        results.append(result)
        print_result(result, show_details=not args.quiet)

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="RAG Agent - NL2GraphQL with ReAct Pattern",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py -q "What is the total stake?"
  python main.py -f questions.txt -b 5000000
  python main.py --endpoint https://api.example.com/graphql -q "..."
        """,
    )

    # Query options
    parser.add_argument("--question", "-q", type=str, help="Single question to process")
    parser.add_argument("--file", "-f", type=str, help="File containing questions (one per line)")
    parser.add_argument("--block-height", "-b", type=int, default=None, help="Block height for time-travel queries")

    # Endpoint options
    parser.add_argument("--endpoint", "-e", type=str, help="GraphQL endpoint URL (overrides env var)")
    parser.add_argument("--headers", type=str, help="Headers JSON for GraphQL endpoint")

    # Model options
    parser.add_argument("--provider", "-p", type=str, default="gemini", help="LLM provider (gemini, openai, mock)")
    parser.add_argument("--model", "-m", type=str, default="gemini-2.5-flash", help="Model name")
    parser.add_argument("--api-key", type=str, help="API key for LLM provider")

    # Agent options
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum retry attempts")
    parser.add_argument("--no-cache", action="store_true", help="Disable schema caching")

    # Output
    parser.add_argument("--quiet", action="store_true", help="Minimal output (answer only)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    # Import and initialize
    try:
        from src.agent import ReActGraphQLAgent
        from src.llm.llm_client import LLMConfig
        from config.settings import Config

        # Load config
        config = Config.from_env()

        # Override endpoint if provided
        endpoint = args.endpoint or config.graphql.endpoint
        if not endpoint:
            print("Error: GraphQL endpoint is required. Set GRAPHQL_ENDPOINT or use --endpoint")
            sys.exit(1)

        # Parse headers
        headers = None
        if args.headers:
            headers = json.loads(args.headers)
        elif config.graphql.api_key:
            auth_type = config.graphql.auth_type
            if auth_type == "bearer":
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {config.graphql.api_key}"
                }
            elif auth_type == "apikey":
                headers = {
                    "Content-Type": "application/json",
                    "X-API-Key": config.graphql.api_key
                }

        # Create LLM config
        llm_config = LLMConfig(
            provider=args.provider if args.provider else config.model.provider,
            model=args.model if args.model else config.model.model,
            api_key=args.api_key or config.model.api_key,
            temperature=config.model.temperature,
            max_tokens=config.model.max_tokens,
        )

        # Create agent
        agent = ReActGraphQLAgent(
            endpoint=endpoint,
            headers=headers,
            llm_config=llm_config,
            max_retries=args.max_retries,
            enable_cache=not args.no_cache,
        )

    except Exception as e:
        print(f"Initialization error: {e}")
        print("\nMake sure you have:")
        print("1. Created a .env file with required variables (copy .env.example)")
        print("2. Installed dependencies: pip install -r requirements.txt")
        print("3. Set GOOGLE_API_KEY for Gemini or OPENAI_API_KEY for OpenAI")
        sys.exit(1)

    # Run appropriate mode
    if args.question:
        result = asyncio.run(single_question(agent, args))
        if args.json:
            print(json.dumps(result, indent=2, default=str))
    elif args.file:
        results = asyncio.run(batch_from_file(agent, args))
        if args.json:
            print(json.dumps(results, indent=2, default=str))
    else:
        asyncio.run(interactive_mode(agent, args))


if __name__ == "__main__":
    main()
