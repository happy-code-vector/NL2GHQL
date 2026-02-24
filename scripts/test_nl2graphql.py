"""
Test NL2GraphQL Pipeline: RAG + Query Generation

Usage:
    # Single question (in-memory, no Docker needed)
    python scripts/test_nl2graphql.py --question "What is the total stake?"

    # Use Weaviate (requires Docker, but faster after first index)
    python scripts/test_nl2graphql.py --question "What is the total stake?" --use-weaviate

    # Test with dataset
    python scripts/test_nl2graphql.py --dataset full_dataset.json --limit 5

    # Execute query against endpoint
    python scripts/test_nl2graphql.py --question "What is the total stake?" --endpoint https://api.example.com/graphql
"""

import sys
import json
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import httpx
import numpy as np
from sentence_transformers import SentenceTransformer
from graphql import parse, validate, build_schema, GraphQLSyntaxError


# --- Schema Validator ---

class SchemaValidator:
    """Validates GraphQL queries against a schema"""

    def __init__(self):
        self.schema = None
        self.field_index: Dict[str, Set[str]] = {}  # type_name -> set of field names

    def load_schema(self, schema_path: str) -> bool:
        """Load schema from JSON introspection or GraphQL SDL file"""
        try:
            # Try GraphQL SDL first
            if schema_path.endswith('.graphql'):
                with open(schema_path, 'r', encoding='utf-8') as f:
                    sdl = f.read()
                self.schema = build_schema(sdl)
            else:
                # Load from JSON introspection
                with open(schema_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Convert introspection result to SDL
                sdl = self._introspection_to_sdl(data)
                self.schema = build_schema(sdl)

            # Build field index for quick lookup
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

            # Skip introspection types
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
            return True, None  # No schema loaded, skip validation

        try:
            document = parse(query)

            # Validate against schema
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

    def check_fields_exist(self, query: str) -> Tuple[bool, List[str]]:
        """Check if all fields in query exist in schema (lighter check)"""
        missing = []
        try:
            doc = parse(query)
            # Walk the AST and check field names
            for definition in doc.definitions:
                if hasattr(definition, 'selection_set'):
                    self._check_selection_set(definition.selection_set, missing)
            return len(missing) == 0, missing
        except:
            return True, []  # Parse error, skip check

    def _check_selection_set(self, selection_set, missing: List[str]):
        """Recursively check selection set fields"""
        if not selection_set:
            return
        for selection in selection_set.selections:
            if hasattr(selection, 'name'):
                field_name = selection.name.value
                # Skip common fields that might not be indexed
                if field_name in ['nodes', 'edges', 'totalCount', 'pageInfo']:
                    pass
            if hasattr(selection, 'selection_set'):
                self._check_selection_set(selection.selection_set, missing)


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

    def get_schema_context(self, query: str, top_k: int = 5) -> str:
        """Get schema context for query (interface matching agent requirements)"""
        results = self.search(query, top_k)
        parts = []
        for r in results:
            if r['description']:
                parts.append(f"# {r['description']}")
            parts.append(r['definition'])
            parts.append("")
        return "\n".join(parts)

    # Keep old method for backwards compatibility
    def get_context(self, query: str, top_k: int = 5) -> str:
        return self.get_schema_context(query, top_k)


# --- Query Syntax Validator ---

def validate_graphql_syntax(query: str) -> Tuple[bool, Optional[str]]:
    """Validate GraphQL syntax only"""
    try:
        parse(query.strip())
        return True, None
    except GraphQLSyntaxError as e:
        return False, f"Syntax error at line {e.line}: {e.message}"
    except Exception as e:
        return False, str(e)


# --- Query Executor ---

async def execute_query(query: str, endpoint: str, timeout: float = 30.0) -> Tuple[bool, Optional[Dict], Optional[str]]:
    """Execute GraphQL query against endpoint"""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                endpoint,
                json={"query": query},
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            data = response.json()

            if "errors" in data:
                error_msgs = []
                for err in data["errors"]:
                    error_msgs.append(err.get("message", str(err)))
                return False, data, "; ".join(error_msgs)

            return True, data.get("data"), None

    except httpx.TimeoutException:
        return False, None, "Request timed out"
    except httpx.HTTPStatusError as e:
        return False, None, f"HTTP error: {e.response.status_code}"
    except Exception as e:
        return False, None, str(e)


# --- Test Pipeline using EnhancedGraphQLAgent ---

async def test_pipeline(
    schema_path: str,
    question: str,
    top_k: int = 5,
    use_weaviate: bool = False,
    endpoint: Optional[str] = None
):
    """Test the full pipeline for a single question using EnhancedGraphQLAgent"""

    print("=" * 70)
    print("NL2GraphQL Pipeline Test")
    print("=" * 70)

    # Load schema validator
    schema_validator = SchemaValidator()
    if schema_validator.load_schema(schema_path):
        print("\n[0] Schema validator loaded")
    else:
        schema_validator = None

    # Choose indexer
    indexer = None
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
                    chunks = parse_schema_file(schema_path.replace('.json', '.graphql'), "subquery")
                    indexer.index_chunks(chunks)
            else:
                print("    Creating new Weaviate index...")
                indexer.create_collection()
                chunks = parse_schema_file(schema_path.replace('.json', '.graphql'), "subquery")
                indexer.index_chunks(chunks)

        except Exception as e:
            print(f"    Weaviate error: {e}")
            print("    Falling back to in-memory indexer...")
            indexer = None

    if indexer is None:
        print("\n[1] Loading schema (in-memory)...")
        indexer = SimpleSchemaIndexer()
        count = indexer.load_schema(schema_path)
        print(f"    Indexed {count} schema types")

    # Show retrieved context
    context = indexer.get_schema_context(question, top_k=top_k)
    results = indexer.search(question, top_k=3) if hasattr(indexer, 'search') else []

    print(f"\n[2] Retrieved schema context ({len(context)} chars)")
    if results:
        print("\n    Top schema matches:")
        for i, r in enumerate(results[:3], 1):
            print(f"      {i}. {r['name']} (score: {r['score']:.3f})")

    # Use EnhancedGraphQLAgent
    print("\n[3] Generating GraphQL query with EnhancedGraphQLAgent...")
    from src.agent.enhanced_graphql_agent import EnhancedGraphQLAgent
    from src.llm.llm_client import get_llm

    llm = get_llm()
    agent = EnhancedGraphQLAgent(
        schema_indexer=indexer,
        llm_client=llm,
        validator=schema_validator  # Pass schema validator
    )

    # Generate query using agent
    result = await agent.answer_question(
        question=question,
        endpoint=endpoint,  # Pass endpoint for execution
        schema_context_k=top_k
    )

    query = result.get('query', '')

    # Validate generated query (syntax)
    syntax_valid, syntax_error = validate_graphql_syntax(query) if query else (False, "No query generated")

    # Validate against schema
    schema_valid = True
    schema_error = None
    if schema_validator and query:
        schema_valid, schema_error = schema_validator.validate(query)

    print("\n" + "=" * 70)
    print("Generated GraphQL Query:")
    print("=" * 70)

    # Status indicators
    status_parts = []
    if syntax_valid:
        status_parts.append("SYNTAX OK")
    else:
        status_parts.append("SYNTAX ERROR")

    if schema_validator:
        if schema_valid:
            status_parts.append("SCHEMA OK")
        else:
            status_parts.append("SCHEMA ERROR")

    status = " | ".join(status_parts)
    print(f"[{status}]")
    print(query)

    if not syntax_valid and syntax_error:
        print(f"\nSyntax error: {syntax_error}")
    if not schema_valid and schema_error:
        print(f"\nSchema validation error: {schema_error}")

    if result.get('error'):
        print(f"\nAgent error: {result['error']}")

    # Show execution result if endpoint was provided
    if endpoint and result.get('query_result'):
        print("\n" + "=" * 70)
        print("Query Result:")
        print("=" * 70)
        print(json.dumps(result['query_result'], indent=2)[:1000])

    if result.get('answer'):
        print("\n" + "=" * 70)
        print("Answer:")
        print("=" * 70)
        print(result['answer'])

    # Close Weaviate connection if used
    if use_weaviate and hasattr(indexer, 'close'):
        indexer.close()

    return query


async def test_dataset(
    schema_path: str,
    dataset_path: str,
    limit: int = 5,
    use_weaviate: bool = False,
    endpoint: Optional[str] = None
):
    """Test pipeline with dataset using EnhancedGraphQLAgent"""

    print("=" * 70)
    print("NL2GraphQL Dataset Test")
    print("=" * 70)

    # Load schema validator
    schema_validator = SchemaValidator()
    if schema_validator.load_schema(schema_path):
        print("\n[0] Schema validator loaded")
    else:
        schema_validator = None

    # Choose indexer
    indexer = None
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
                    gql_path = schema_path.replace('.json', '.graphql')
                    chunks = parse_schema_file(gql_path, "subquery")
                    indexer.index_chunks(chunks)
            else:
                print("    Creating new Weaviate index...")
                indexer.create_collection()
                gql_path = schema_path.replace('.json', '.graphql')
                chunks = parse_schema_file(gql_path, "subquery")
                indexer.index_chunks(chunks)

        except Exception as e:
            print(f"    Weaviate error: {e}")
            print("    Falling back to in-memory indexer...")
            indexer = None

    if indexer is None:
        print("\n[1] Loading schema (in-memory)...")
        indexer = SimpleSchemaIndexer()
        try:
            count = indexer.load_schema(schema_path)
        except json.JSONDecodeError:
            print("    JSON corrupted, using GraphQL file...")
            gql_path = schema_path.replace('.json', '.graphql')
            count = indexer.load_schema(gql_path)
        print(f"    Indexed {count} schema types")

    # Load dataset
    print(f"\n[2] Loading dataset: {dataset_path}")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    questions = dataset[:limit]
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
    for i, item in enumerate(questions, 1):
        question = item.get('question', str(item))
        expected_score = item.get('score', 0)

        print(f"\n{'='*70}")
        print(f"[{i}/{len(questions)}] Question: {question[:80]}...")
        print("-" * 70)

        try:
            # Use agent to generate query
            result = await agent.answer_question(
                question=question,
                endpoint=endpoint,
                schema_context_k=5
            )

            query = result.get('query', '')
            syntax_valid, syntax_error = validate_graphql_syntax(query) if query else (False, "No query generated")

            # Schema validation
            schema_valid = True
            schema_error = None
            if schema_validator and query:
                schema_valid, schema_error = schema_validator.validate(query)

            # Execution result
            exec_success = result.get('query_result') is not None if endpoint else None

            results.append({
                'question': question,
                'expected_score': expected_score,
                'query': query,
                'syntax_valid': syntax_valid,
                'schema_valid': schema_valid,
                'exec_success': exec_success,
                'validation_errors': [e for e in [syntax_error, schema_error] if e],
                'success': bool(query),
                'elapsed_time': result.get('elapsed_time', 0)
            })

            # Status
            status_parts = []
            status_parts.append("SYNTAX OK" if syntax_valid else "SYNTAX ERR")
            if schema_validator:
                status_parts.append("SCHEMA OK" if schema_valid else "SCHEMA ERR")
            if endpoint:
                status_parts.append("EXEC OK" if exec_success else "EXEC FAIL")

            status = " | ".join(status_parts)
            print(f"[{status}]")
            print(f"{query[:500]}...")

            if not syntax_valid and syntax_error:
                print(f"  Syntax error: {syntax_error[:100]}")
            if not schema_valid and schema_error:
                print(f"  Schema error: {schema_error[:100]}")

            print(f"  Time: {result.get('elapsed_time', 0):.2f}s")

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
    syntax_ok = sum(1 for r in results if r.get('syntax_valid'))
    schema_ok = sum(1 for r in results if r.get('schema_valid', True))
    exec_ok = sum(1 for r in results if r.get('exec_success', True))
    avg_time = sum(r.get('elapsed_time', 0) for r in results) / len(results) if results else 0

    print(f"Generated: {success_count}/{len(results)}")
    print(f"Syntax valid: {syntax_ok}/{len(results)}")
    if schema_validator:
        print(f"Schema valid: {schema_ok}/{len(results)}")
    if endpoint:
        print(f"Exec success: {exec_ok}/{len(results)}")
    print(f"Avg time: {avg_time:.2f}s")

    # Close Weaviate connection if used
    if use_weaviate and hasattr(indexer, 'close'):
        indexer.close()

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
    parser.add_argument("--endpoint", "-e", help="GraphQL endpoint URL to execute queries")

    args = parser.parse_args()

    if not Path(args.schema).exists():
        print(f"Error: Schema not found: {args.schema}")
        return

    if args.question:
        asyncio.run(test_pipeline(args.schema, args.question, args.top_k, args.use_weaviate, args.endpoint))
    elif args.dataset:
        if not Path(args.dataset).exists():
            print(f"Error: Dataset not found: {args.dataset}")
            return
        asyncio.run(test_dataset(args.schema, args.dataset, args.limit, args.use_weaviate, args.endpoint))
    else:
        # Default: test with a sample question
        sample_question = "What is the total stake of all indexers?"
        print("No question or dataset provided. Testing with sample question.")
        asyncio.run(test_pipeline(args.schema, sample_question, args.top_k, args.use_weaviate, args.endpoint))


if __name__ == "__main__":
    main()
