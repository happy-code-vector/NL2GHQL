"""
Test NL2GraphQL Pipeline: RAG + Query Generation + Execution + Answer Generation

Usage:
    # Single question (in-memory, no Docker needed)
    python scripts/test_nl2graphql.py --question "What is the total stake?"

    # Use Weaviate (requires Docker, but faster after first index)
    python scripts/test_nl2graphql.py --question "What is the total stake?" --use-weaviate

    # Test with dataset (no execution)
    python scripts/test_nl2graphql.py --dataset full_dataset.json --limit 5

    # Full test with real endpoint (execute queries and generate answers)
    python scripts/test_nl2graphql.py --dataset full_dataset.json --endpoint

    # Full test with Weaviate and real endpoint
    python scripts/test_nl2graphql.py --dataset full_dataset.json --use-weaviate --endpoint --limit 10
"""

import sys
import json
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import numpy as np
from sentence_transformers import SentenceTransformer
from graphql import parse, validate, build_schema, GraphQLSyntaxError


# --- Schema Validator ---

class SchemaValidator:
    """Validates GraphQL queries against a schema"""

    def __init__(self):
        self.schema = None
        self.field_index: Dict[str, Set[str]] = {}

    def load_schema(self, schema_path: str) -> bool:
        """Load schema from JSON introspection or GraphQL SDL file"""
        try:
            if schema_path.endswith('.graphql'):
                with open(schema_path, 'r', encoding='utf-8') as f:
                    sdl = f.read()
                self.schema = build_schema(sdl)
            else:
                with open(schema_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                sdl = self._introspection_to_sdl(data)
                self.schema = build_schema(sdl)

            self._build_field_index()
            return True

        except Exception as e:
            print(f"    Warning: Could not load schema for validation: {e}")
            return False

    def _introspection_to_sdl(self, data: Dict) -> str:
        """Convert introspection JSON to GraphQL SDL"""
        types_data = data.get('types', {})
        if isinstance(types_data, dict):
            types_list = list(types_data.values())
        else:
            types_list = types_data

        sdl_parts = []

        for type_info in types_list:
            if not isinstance(type_info, dict):
                continue

            name = type_info.get('name', '')
            kind = type_info.get('kind', '')

            if name.startswith('__'):
                continue

            if kind == 'OBJECT' or kind == 'INTERFACE':
                fields = type_info.get('fields', [])
                if not fields:
                    continue

                field_strs = []
                for field in fields:
                    if not isinstance(field, dict):
                        continue
                    fname = field.get('name', '')
                    ftype = self._type_to_str(field.get('type', {}))

                    args = field.get('args', [])
                    if args:
                        arg_strs = []
                        for arg in args:
                            if not isinstance(arg, dict):
                                continue
                            arg_name = arg.get('name', '')
                            arg_type = self._type_to_str(arg.get('type', {}))
                            arg_strs.append(f"{arg_name}: {arg_type}")
                        field_strs.append(f"  {fname}({', '.join(arg_strs)}): {ftype}")
                    else:
                        field_strs.append(f"  {fname}: {ftype}")

                if field_strs:
                    sdl_parts.append(f"type {name} {{\n" + "\n".join(field_strs) + "\n}")

            elif kind == 'ENUM':
                enum_values = type_info.get('enumValues', [])
                if enum_values:
                    values = [e.get('name', '') if isinstance(e, dict) else str(e) for e in enum_values]
                    sdl_parts.append(f"enum {name} {{ " + " ".join(values) + " }")

            elif kind == 'INPUT_OBJECT':
                fields = type_info.get('inputFields', [])
                if not fields:
                    continue

                field_strs = []
                for field in fields:
                    if not isinstance(field, dict):
                        continue
                    fname = field.get('name', '')
                    ftype = self._type_to_str(field.get('type', {}))
                    field_strs.append(f"  {fname}: {ftype}")

                if field_strs:
                    sdl_parts.append(f"input {name} {{\n" + "\n".join(field_strs) + "\n}")

        return "\n\n".join(sdl_parts)

    def _type_to_str(self, type_info: Dict) -> str:
        """Convert type dict to string"""
        if not type_info:
            return "String"
        kind = type_info.get('kind', '')
        name = type_info.get('name', '')
        of_type = type_info.get('ofType', {})
        if kind == 'NON_NULL':
            return self._type_to_str(of_type) + '!'
        elif kind == 'LIST':
            return f"[{self._type_to_str(of_type)}]"
        return name or "String"

    def _build_field_index(self):
        """Build index of fields per type for quick lookup"""
        if not self.schema:
            return

        for type_name, type_def in self.schema.type_map.items():
            if type_name.startswith('__'):
                continue

            fields = set()
            if hasattr(type_def, 'fields') and type_def.fields:
                for field_name in type_def.fields:
                    fields.add(field_name)

            if fields:
                self.field_index[type_name] = fields

    def validate(self, query: str) -> Tuple[bool, Optional[str]]:
        """Validate query against schema"""
        if not self.schema:
            return True, None

        try:
            document = parse(query)
            errors = validate(self.schema, document)

            if errors:
                error_msgs = []
                for err in errors:
                    error_msgs.append(str(err.message))
                return False, "; ".join(error_msgs)

            return True, None

        except GraphQLSyntaxError as e:
            return False, f"Syntax error at line {e.line}: {e.message}"
        except Exception as e:
            return False, str(e)


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

            if name.startswith('__') or name in ['Boolean', 'String', 'Int', 'Float', 'ID']:
                continue

            fields = type_info.get('fields', []) or type_info.get('inputFields', []) or []
            enum_values = type_info.get('enumValues', [])
            description = type_info.get('description', '') or ''

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

        if self.chunks:
            texts = [c['text'] for c in self.chunks]
            self.embeddings = self.encoder.encode(texts, show_progress_bar=True)

        return len(self.chunks)

    def _type_to_str(self, type_info: Dict) -> str:
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

    def get_schema_context(self, query: str, top_k: int = 5) -> str:
        results = self.search(query, top_k)
        parts = []
        for r in results:
            if r['description']:
                parts.append(f"# {r['description']}")
            parts.append(r['definition'])
            parts.append("")
        return "\n".join(parts)

    def get_context(self, query: str, top_k: int = 5) -> str:
        return self.get_schema_context(query, top_k)


# --- Query Syntax Validator ---

def validate_graphql_syntax(query: str) -> Tuple[bool, Optional[str]]:
    try:
        parse(query.strip())
        return True, None
    except GraphQLSyntaxError as e:
        return False, f"Syntax error at line {e.line}: {e.message}"
    except Exception as e:
        return False, str(e)


# --- Test Pipeline ---

async def test_pipeline(
    schema_path: str,
    question: str,
    top_k: int = 5,
    use_weaviate: bool = False,
    execute: bool = False
):
    """Test the full pipeline for a single question"""

    print("=" * 70)
    print("NL2GraphQL Pipeline Test")
    if execute:
        from src.agent.enhanced_graphql_agent import SUBQUERY_ENDPOINT
        print(f"Endpoint: {SUBQUERY_ENDPOINT}")
    print("=" * 70)

    # Load schema validator
    schema_validator = SchemaValidator()
    schema_validator.load_schema(schema_path)

    # Choose indexer
    indexer = None
    if use_weaviate:
        print("\n[1] Using Weaviate...")
        try:
            from src.rag.weaviate_indexer import WeaviateSchemaIndexer, parse_schema_file
            indexer = WeaviateSchemaIndexer()

            if indexer.client.collections.exists(indexer.COLLECTION_NAME):
                collection = indexer.client.collections.get(indexer.COLLECTION_NAME)
                result = collection.aggregate.over_all(total_count=True)
                if result.total_count > 0:
                    print(f"    Using existing index with {result.total_count} types")
                else:
                    indexer.create_collection()
                    chunks = parse_schema_file(schema_path.replace('.json', '.graphql'), "subquery")
                    indexer.index_chunks(chunks)
            else:
                indexer.create_collection()
                chunks = parse_schema_file(schema_path.replace('.json', '.graphql'), "subquery")
                indexer.index_chunks(chunks)

        except Exception as e:
            print(f"    Weaviate error: {e}, falling back to in-memory...")
            indexer = None

    if indexer is None:
        print("\n[1] Loading schema (in-memory)...")
        indexer = SimpleSchemaIndexer()
        count = indexer.load_schema(schema_path)
        print(f"    Indexed {count} schema types")

    # Use EnhancedGraphQLAgent
    print("\n[2] Generating GraphQL query...")
    from src.agent.enhanced_graphql_agent import EnhancedGraphQLAgent
    from src.llm.llm_client import get_llm

    llm = get_llm()
    agent = EnhancedGraphQLAgent(
        schema_indexer=indexer,
        llm_client=llm,
        validator=schema_validator
    )

    # Generate query
    # Pass True as endpoint to use agent's built-in SubQuery endpoint
    result = await agent.answer_question(
        question=question,
        endpoint=True if execute else None,
        schema_context_k=top_k
    )

    query = result.get('query', '')

    # Validate
    syntax_valid, syntax_error = validate_graphql_syntax(query) if query else (False, "No query generated")
    schema_valid, schema_error = schema_validator.validate(query) if query else (False, "No query")

    print("\n" + "=" * 70)
    print("Generated Query:")
    print("=" * 70)
    status = f"SYNTAX: {'OK' if syntax_valid else 'ERR'} | SCHEMA: {'OK' if schema_valid else 'ERR'}"
    print(f"[{status}]")
    print(query)

    if result.get('error'):
        print(f"\nError: {result['error']}")

    if execute and result.get('query_result'):
        print("\n" + "=" * 70)
        print("Query Result:")
        print("=" * 70)
        print(json.dumps(result['query_result'], indent=2)[:2000])

    if result.get('answer'):
        print("\n" + "=" * 70)
        print("Answer:")
        print("=" * 70)
        print(result['answer'])

    print(f"\nTime: {result.get('elapsed_time', 0):.2f}s")

    if use_weaviate and hasattr(indexer, 'close'):
        indexer.close()

    return result


# --- Full Dataset Test ---

async def test_dataset(
    schema_path: str,
    dataset_path: str,
    limit: int = 0,  # 0 = all
    use_weaviate: bool = False,
    execute: bool = False,
    output_file: Optional[str] = None
):
    """Full test with dataset - complete pipeline matching real agent behavior"""

    print("=" * 70)
    print("NL2GraphQL Full Dataset Test")
    print(f"Execute queries: {execute}")
    if execute:
        from src.agent.enhanced_graphql_agent import SUBQUERY_ENDPOINT
        print(f"Endpoint: {SUBQUERY_ENDPOINT}")
    print("=" * 70)

    # Load schema validator
    schema_validator = SchemaValidator()
    if schema_validator.load_schema(schema_path):
        print("\n[0] Schema validator loaded")

    # Choose indexer
    indexer = None
    if use_weaviate:
        print("\n[1] Using Weaviate...")
        try:
            from src.rag.weaviate_indexer import WeaviateSchemaIndexer, parse_schema_file
            indexer = WeaviateSchemaIndexer()

            if indexer.client.collections.exists(indexer.COLLECTION_NAME):
                collection = indexer.client.collections.get(indexer.COLLECTION_NAME)
                result = collection.aggregate.over_all(total_count=True)
                if result.total_count > 0:
                    print(f"    Using existing index with {result.total_count} types")
                else:
                    indexer.create_collection()
                    gql_path = schema_path.replace('.json', '.graphql')
                    chunks = parse_schema_file(gql_path, "subquery")
                    indexer.index_chunks(chunks)
            else:
                indexer.create_collection()
                gql_path = schema_path.replace('.json', '.graphql')
                chunks = parse_schema_file(gql_path, "subquery")
                indexer.index_chunks(chunks)

        except Exception as e:
            print(f"    Weaviate error: {e}, falling back...")
            indexer = None

    if indexer is None:
        print("\n[1] Loading schema (in-memory)...")
        indexer = SimpleSchemaIndexer()
        try:
            count = indexer.load_schema(schema_path)
        except json.JSONDecodeError:
            gql_path = schema_path.replace('.json', '.graphql')
            count = indexer.load_schema(gql_path)
        print(f"    Indexed {count} schema types")

    # Load dataset
    print(f"\n[2] Loading dataset: {dataset_path}")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    total_questions = len(dataset) if limit == 0 else min(limit, len(dataset))
    questions = dataset[:total_questions] if limit > 0 else dataset
    print(f"    Testing {len(questions)} questions")

    # Initialize agent
    from src.agent.enhanced_graphql_agent import EnhancedGraphQLAgent
    from src.llm.llm_client import get_llm

    llm = get_llm()
    agent = EnhancedGraphQLAgent(
        schema_indexer=indexer,
        llm_client=llm,
        validator=schema_validator
    )

    results = []
    total_score = 0
    total_expected = 0

    for i, item in enumerate(questions, 1):
        question = item.get('question', str(item))
        expected_score = item.get('score', 0)
        question_id = item.get('id', str(i))

        print(f"\n{'='*70}")
        print(f"[{i}/{len(questions)}] ID: {question_id}")
        print(f"Question: {question[:100]}...")
        print(f"Expected score: {expected_score}")
        print("-" * 70)

        try:
            # Full pipeline - same as real agent
            # Pass True as endpoint to use agent's built-in SubQuery endpoint
            result = await agent.answer_question(
                question=question,
                endpoint=True if execute else None,
                schema_context_k=5
            )

            query = result.get('query', '')
            answer = result.get('answer', '')
            query_result = result.get('query_result')
            error = result.get('error')
            elapsed = result.get('elapsed_time', 0)

            # Validate
            syntax_valid, _ = validate_graphql_syntax(query) if query else (False, None)
            schema_valid, _ = schema_validator.validate(query) if query else (False, None)
            exec_success = query_result is not None

            # Calculate our score (simple heuristic for now)
            calc_score = 0
            if query and syntax_valid:
                calc_score += 10
            if schema_valid:
                calc_score += 20
            if exec_success:
                calc_score += 30
            if answer:
                calc_score += 40

            total_score += calc_score
            total_expected += expected_score

            # Status
            status_parts = []
            status_parts.append("SYN" if syntax_valid else "SYN!")
            status_parts.append("SCH" if schema_valid else "SCH!")
            if execute:
                status_parts.append("EXE" if exec_success else "EXE!")
            status_parts.append("ANS" if answer else "ANS!")
            status = " | ".join(status_parts)

            print(f"[{status}]")
            print(f"Query: {query[:200]}..." if len(query) > 200 else f"Query: {query}")

            if answer:
                print(f"Answer: {answer[:200]}..." if len(answer) > 200 else f"Answer: {answer}")

            if error:
                print(f"Error: {error[:100]}")

            print(f"Time: {elapsed:.2f}s | Score: {calc_score} (expected: {expected_score})")

            results.append({
                'id': question_id,
                'question': question,
                'expected_score': expected_score,
                'calculated_score': calc_score,
                'query': query,
                'answer': answer,
                'syntax_valid': syntax_valid,
                'schema_valid': schema_valid,
                'exec_success': exec_success if execute else None,
                'has_answer': bool(answer),
                'error': error,
                'elapsed_time': elapsed,
                'timestamp': datetime.now().isoformat()
            })

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                'id': question_id,
                'question': question,
                'expected_score': expected_score,
                'calculated_score': 0,
                'error': str(e),
                'success': False
            })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    syntax_ok = sum(1 for r in results if r.get('syntax_valid'))
    schema_ok = sum(1 for r in results if r.get('schema_valid'))
    exec_ok = sum(1 for r in results if r.get('exec_success')) if execute else len(results)
    has_answer = sum(1 for r in results if r.get('has_answer'))
    avg_time = sum(r.get('elapsed_time', 0) for r in results) / len(results) if results else 0

    print(f"Total questions: {len(results)}")
    print(f"Syntax valid: {syntax_ok}/{len(results)} ({100*syntax_ok/len(results):.1f}%)")
    print(f"Schema valid: {schema_ok}/{len(results)} ({100*schema_ok/len(results):.1f}%)")
    if execute:
        print(f"Exec success: {exec_ok}/{len(results)} ({100*exec_ok/len(results):.1f}%)")
    print(f"Has answer: {has_answer}/{len(results)} ({100*has_answer/len(results):.1f}%)")
    print(f"Avg time: {avg_time:.2f}s")
    print(f"Total score: {total_score} / Expected: {total_expected}")

    # Save results
    if output_file:
        output_path = Path(output_file)
    else:
        output_path = Path(dataset_path).parent / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'total_questions': len(results),
                'syntax_valid': syntax_ok,
                'schema_valid': schema_ok,
                'exec_success': exec_ok if execute else None,
                'has_answer': has_answer,
                'total_score': total_score,
                'expected_score': total_expected,
                'avg_time': avg_time,
                'timestamp': datetime.now().isoformat()
            },
            'results': results
        }, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Close connections
    if use_weaviate and hasattr(indexer, 'close'):
        indexer.close()

    return results


def main():
    parser = argparse.ArgumentParser(description="Test NL2GraphQL Pipeline")
    parser.add_argument("--schema", default="data/schemas/schema_subnet.json",
                        help="Path to schema JSON file")
    parser.add_argument("--question", "-q", help="Single question to test")
    parser.add_argument("--dataset", "-d", help="Dataset JSON file")
    parser.add_argument("--limit", "-n", type=int, default=0,
                        help="Number of questions to test (0 = all)")
    parser.add_argument("--top-k", type=int, default=5, help="Schema chunks to retrieve")
    parser.add_argument("--use-weaviate", "-w", action="store_true",
                        help="Use Weaviate (requires Docker)")
    parser.add_argument("--endpoint", "-e", action="store_true",
                        help="Execute queries against real SubQuery endpoint")
    parser.add_argument("--output", "-o", help="Output file for results")

    args = parser.parse_args()

    if not Path(args.schema).exists():
        print(f"Error: Schema not found: {args.schema}")
        return

    if args.question:
        asyncio.run(test_pipeline(
            args.schema, args.question, args.top_k, args.use_weaviate, args.endpoint
        ))
    elif args.dataset:
        if not Path(args.dataset).exists():
            print(f"Error: Dataset not found: {args.dataset}")
            return
        asyncio.run(test_dataset(
            args.schema, args.dataset, args.limit, args.use_weaviate, args.endpoint, args.output
        ))
    else:
        sample_question = "What is the total stake of all indexers?"
        print("No question or dataset provided. Testing with sample question.")
        asyncio.run(test_pipeline(
            args.schema, sample_question, args.top_k, args.use_weaviate, args.endpoint
        ))


if __name__ == "__main__":
    main()
