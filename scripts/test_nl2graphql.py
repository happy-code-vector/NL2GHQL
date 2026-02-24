"""
Test NL2GraphQL Pipeline: RAG + Query Generation

Usage:
    # Single question (in-memory, no Docker needed)
    python scripts/test_nl2graphql.py --question "What is the total stake?"

    # Use Weaviate (requires Docker, but faster after first index)
    python scripts/test_nl2graphql.py --question "What is the total stake?" --use-weaviate

    # Test with dataset
    python scripts/test_nl2graphql.py --dataset full_dataset.json --limit 5
"""

import sys
import json
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import numpy as np
from sentence_transformers import SentenceTransformer


# --- Simple In-Memory Schema Indexer ---

class SimpleSchemaIndexer:
    """In-memory schema indexer using sentence-transformers"""

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(embedding_model)
        self.chunks: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None

    def load_schema(self, schema_path: str) -> int:
        """Load and index schema from JSON file"""
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)

        types_data = schema.get('types', {})
        if isinstance(types_data, dict):
            types_list = list(types_data.values())
        else:
            types_list = types_data

        self.chunks = []
        for type_info in types_list:
            if not isinstance(type_info, dict):
                continue

            name = type_info.get('name', '')
            kind = type_info.get('kind', '')

            # Skip introspection types and scalars
            if name.startswith('__') or name in ['Boolean', 'String', 'Int', 'Float', 'ID']:
                continue

            fields = type_info.get('fields', []) or type_info.get('inputFields', []) or []
            enum_values = type_info.get('enumValues', [])
            description = type_info.get('description', '') or ''

            # Build definition
            if kind == 'ENUM' and enum_values:
                field_names = [e.get('name', '') if isinstance(e, dict) else str(e) for e in enum_values]
                definition = f"enum {name} {{ " + " ".join(field_names) + " }"
            elif fields:
                field_strs = []
                field_names = []
                for field in fields:
                    if not isinstance(field, dict):
                        continue
                    fname = field.get('name', '')
                    ftype = field.get('type', '')
                    if isinstance(ftype, dict):
                        ftype = self._type_to_str(ftype)
                    field_names.append(fname)

                    args = field.get('args', [])
                    if args:
                        arg_strs = []
                        for arg in args:
                            if not isinstance(arg, dict):
                                continue
                            arg_name = arg.get('name', '')
                            arg_type = arg.get('type', '')
                            if isinstance(arg_type, dict):
                                arg_type = self._type_to_str(arg_type)
                            arg_strs.append(f"{arg_name}: {arg_type}")
                        field_strs.append(f"  {fname}({', '.join(arg_strs)}): {ftype}")
                    else:
                        field_strs.append(f"  {fname}: {ftype}")

                if not field_strs:
                    continue

                type_kw = "input" if kind == "INPUT_OBJECT" else "type"
                definition = f"{type_kw} {name} {{\n" + "\n".join(field_strs) + "\n}"
                fields = field_names
            else:
                continue

            # Create searchable text
            text_parts = []
            if description:
                text_parts.append(f"Description: {description}")
            text_parts.append(f"Type: {kind} named {name}")
            if fields:
                text_parts.append(f"Fields: {', '.join(fields[:20])}")
            text_parts.append(definition)

            self.chunks.append({
                'name': name,
                'kind': kind,
                'definition': definition,
                'fields': fields,
                'description': description,
                'text': "\n".join(text_parts)
            })

        # Create embeddings
        if self.chunks:
            texts = [c['text'] for c in self.chunks]
            self.embeddings = self.encoder.encode(texts, show_progress_bar=True)

        return len(self.chunks)

    def _type_to_str(self, type_info: Dict) -> str:
        """Convert type dict to string"""
        if not type_info:
            return "Unknown"
        kind = type_info.get('kind', '')
        name = type_info.get('name', '')
        of_type = type_info.get('ofType', {})
        if kind == 'NON_NULL':
            return self._type_to_str(of_type) + '!'
        elif kind == 'LIST':
            return f"[{self._type_to_str(of_type)}]"
        return name

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant schema chunks"""
        if not self.chunks or self.embeddings is None:
            return []

        query_embedding = self.encoder.encode(query)
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            results.append({
                'score': float(similarities[idx]),
                'name': chunk['name'],
                'kind': chunk['kind'],
                'definition': chunk['definition'],
                'fields': chunk['fields'],
                'description': chunk['description'],
            })
        return results

    def get_context(self, query: str, top_k: int = 5) -> str:
        """Get schema context for query"""
        results = self.search(query, top_k)
        parts = []
        for r in results:
            if r['description']:
                parts.append(f"# {r['description']}")
            parts.append(r['definition'])
            parts.append("")
        return "\n".join(parts)


# --- Query Generation ---

FEW_SHOT_EXAMPLES = """
## Example 1: Count Query
Question: "How many indexers are currently active?"
GraphQL:
```graphql
query {
  indexers(first: 1000, filter: { active: { equalTo: true } }) {
    totalCount
  }
}
```

## Example 2: Superlative Query (highest/lowest)
Question: "Which indexer has the highest total stake?"
GraphQL:
```graphql
query {
  indexers(first: 1, orderBy: [TOTAL_STAKE_DESC]) {
    nodes {
      id
      totalStake
    }
  }
}
```

## Example 3: Aggregation Query
Question: "What is the total rewards earned in era 42200?"
GraphQL:
```graphql
query {
  eraRewards(filter: { eraIdx: { equalTo: 42200 } }) {
    nodes {
      totalRewards
    }
  }
}
```
"""


async def generate_query(llm, question: str, schema_context: str) -> str:
    """Generate GraphQL query using LLM"""

    prompt = f"""You are a GraphQL query generator for a blockchain indexing service.

Generate a valid GraphQL query based on the question and schema context.

{FEW_SHOT_EXAMPLES}

## Schema Context
{schema_context}

## Question
{question}

## GraphQL Query
Generate only the GraphQL query (no explanation):

```graphql
"""

    response = await llm.generate_async(prompt)
    return response


async def test_pipeline(
    schema_path: str,
    question: str,
    top_k: int = 5,
    use_weaviate: bool = False
):
    """Test the full pipeline for a single question"""

    print("=" * 70)
    print("NL2GraphQL Pipeline Test")
    print("=" * 70)

    # Choose indexer
    if use_weaviate:
        print("\n[1] Using Weaviate (persistent index)...")
        try:
            from src.rag.weaviate_indexer import WeaviateSchemaIndexer, parse_schema_file
            indexer = WeaviateSchemaIndexer()

            # Check if collection exists and has data
            if indexer.client.collections.exists(indexer.COLLECTION_NAME):
                collection = indexer.client.collections.get(indexer.COLLECTION_NAME)
                result = collection.aggregate.over_all(total_count=True)
                if result.total_count > 0:
                    print(f"    Using existing index with {result.total_count} types")
                else:
                    print("    Indexing schema into Weaviate...")
                    indexer.create_collection()
                    chunks = parse_schema_file(schema_path, "subquery")
                    indexer.index_chunks(chunks)
            else:
                print("    Creating new Weaviate index...")
                indexer.create_collection()
                chunks = parse_schema_file(schema_path, "subquery")
                indexer.index_chunks(chunks)

            # Use Weaviate's hybrid search
            results = indexer.hybrid_search(question, top_k=top_k)
            context = indexer.get_schema_context(question, top_k)

        except Exception as e:
            print(f"    Weaviate error: {e}")
            print("    Falling back to in-memory indexer...")
            use_weaviate = False

    if not use_weaviate:
        print("\n[1] Loading schema (in-memory)...")
        indexer = SimpleSchemaIndexer()
        count = indexer.load_schema(schema_path)
        print(f"    Indexed {count} schema types")

        context = indexer.get_context(question, top_k=top_k)
        results = indexer.search(question, top_k=3)

    # Show results
    print(f"\n[2] Retrieved schema context ({len(context)} chars)")
    print("\n    Top schema matches:")
    for i, r in enumerate(results[:3], 1):
        print(f"      {i}. {r['name']} (score: {r['score']:.3f})")

    # Generate query
    print("\n[3] Generating GraphQL query...")
    from src.llm.llm_client import get_llm
    llm = get_llm()

    query = await generate_query(llm, question, context)

    print("\n" + "=" * 70)
    print("Generated GraphQL Query:")
    print("=" * 70)
    print(query)

    # Close Weaviate connection if used
    if use_weaviate and hasattr(indexer, 'close'):
        indexer.close()

    return query


async def test_dataset(
    schema_path: str,
    dataset_path: str,
    limit: int = 5,
    use_weaviate: bool = False
):
    """Test pipeline with dataset"""

    print("=" * 70)
    print("NL2GraphQL Dataset Test")
    print("=" * 70)

    # Load schema
    print("\n[1] Loading schema...")
    indexer = SimpleSchemaIndexer()
    count = indexer.load_schema(schema_path)
    print(f"    Indexed {count} schema types")

    # Load dataset
    print(f"\n[2] Loading dataset: {dataset_path}")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    questions = dataset[:limit]
    print(f"    Testing {len(questions)} questions")

    # Initialize LLM
    from src.llm.llm_client import get_llm
    llm = get_llm()

    results = []
    for i, item in enumerate(questions, 1):
        question = item.get('question', str(item))
        expected_score = item.get('score', 0)

        print(f"\n{'='*70}")
        print(f"[{i}/{len(questions)}] Question: {question[:80]}...")
        print("-" * 70)

        # Retrieve context
        context = indexer.get_context(question, top_k=5)

        # Generate query
        try:
            query = await generate_query(llm, question, context)
            results.append({
                'question': question,
                'expected_score': expected_score,
                'query': query,
                'success': True
            })
            print(f"Generated Query:\n{query[:500]}...")
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                'question': question,
                'expected_score': expected_score,
                'query': None,
                'success': False,
                'error': str(e)
            })

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    success_count = sum(1 for r in results if r['success'])
    print(f"Success: {success_count}/{len(results)}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Test NL2GraphQL Pipeline")
    parser.add_argument("--schema", default="data/schemas/schema_subnet.json",
                        help="Path to schema JSON file")
    parser.add_argument("--question", "-q", help="Single question to test")
    parser.add_argument("--dataset", "-d", help="Dataset JSON file")
    parser.add_argument("--limit", "-n", type=int, default=5, help="Number of questions to test")
    parser.add_argument("--top-k", type=int, default=5, help="Number of schema chunks to retrieve")
    parser.add_argument("--use-weaviate", "-w", action="store_true",
                        help="Use Weaviate (requires Docker)")

    args = parser.parse_args()

    if not Path(args.schema).exists():
        print(f"Error: Schema not found: {args.schema}")
        return

    if args.question:
        asyncio.run(test_pipeline(args.schema, args.question, args.top_k, args.use_weaviate))
    elif args.dataset:
        if not Path(args.dataset).exists():
            print(f"Error: Dataset not found: {args.dataset}")
            return
        asyncio.run(test_dataset(args.schema, args.dataset, args.limit, args.use_weaviate))
    else:
        # Default: test with a sample question
        sample_question = "What is the total stake of all indexers?"
        print("No question or dataset provided. Testing with sample question.")
        asyncio.run(test_pipeline(args.schema, sample_question, args.top_k, args.use_weaviate))


if __name__ == "__main__":
    main()
