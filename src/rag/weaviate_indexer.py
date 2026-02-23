"""
Weaviate Schema Indexer for GraphQL

Implements Hybrid Search (BM25 + Vector) with native GraphQL support.
Auto-vectorization with OpenAI or local models.

Installation:
    pip install weaviate-client

Docker:
    docker run -d -p 8080:8080 -p 50051:50051 semitechnologies/weaviate:latest
"""

import re
import json
from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path

try:
    import weaviate
    import weaviate.classes as wvc
    from weaviate.classes.config import Configure, Property, DataType
    from weaviate.classes.query import MetadataQuery
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False


@dataclass
class SchemaChunk:
    """Represents a parsed schema definition chunk"""
    name: str
    type: str
    definition: str
    fields: List[str]
    description: Optional[str] = None
    project: Optional[str] = None


class WeaviateSchemaIndexer:
    """
    Weaviate-based schema indexer with native Hybrid Search.

    Features:
    - Built-in Hybrid Search (BM25 + Vector)
    - Native GraphQL API
    - Auto-vectorization (optional)
    - RAG queries (retrieval + generation in one call)
    """

    COLLECTION_NAME = "GraphQLSchema"

    def __init__(
        self,
        weaviate_url: str = "http://localhost:8080",
        use_openai_vectorizer: bool = False,
        openai_api_key: Optional[str] = None,
        local_embedding_model: str = "all-MiniLM-L6-v2"
    ):
        if not WEAVIATE_AVAILABLE:
            raise ImportError("weaviate-client not installed. Run: pip install weaviate-client")

        self.weaviate_url = weaviate_url
        self.use_openai_vectorizer = use_openai_vectorizer
        self.local_embedding_model = local_embedding_model

        # Connect to Weaviate
        self.client = weaviate.connect_to_local(
            host=weaviate_url.replace("http://", "").replace(":8080", ""),
            port=8080
        )

        # Local embedder (fallback if not using OpenAI)
        self._local_embedder = None
        if not use_openai_vectorizer:
            from sentence_transformers import SentenceTransformer
            self._local_embedder = SentenceTransformer(local_embedding_model)

    def create_collection(self, recreate: bool = True) -> None:
        """Create the GraphQL Schema collection with Hybrid Search"""
        if recreate and self.client.collections.exists(self.COLLECTION_NAME):
            self.client.collections.delete(self.COLLECTION_NAME)

        if not self.client.collections.exists(self.COLLECTION_NAME):
            # Configure vectorizer
            if self.use_openai_vectorizer:
                vectorizer_config = Configure.Vectorizer.text2vec_openai()
            else:
                vectorizer_config = Configure.Vectorizer.none()  # We'll provide vectors manually

            self.client.collections.create(
                name=self.COLLECTION_NAME,
                description="GraphQL Schema types and definitions for RAG",
                vectorizer_config=vectorizer_config,
                properties=[
                    Property(name="typeName", data_type=DataType.TEXT),
                    Property(name="typeKind", data_type=DataType.TEXT),
                    Property(name="definition", data_type=DataType.TEXT),
                    Property(name="fields", data_type=DataType.TEXT_ARRAY),
                    Property(name="description", data_type=DataType.TEXT),
                    Property(name="project", data_type=DataType.TEXT),
                    Property(name="content", data_type=DataType.TEXT),  # Combined searchable text
                ],
                # Enable BM25 for keyword search
                inverted_index_config=Configure.inverted_index(
                    index_null_state=True,
                    index_property_length=True,
                    index_timestamps=True
                )
            )

    def _chunk_to_text(self, chunk: SchemaChunk) -> str:
        """Convert chunk to searchable text"""
        parts = []
        if chunk.description:
            parts.append(f"Description: {chunk.description}")
        parts.append(f"Type: {chunk.type} named {chunk.name}")
        if chunk.fields:
            parts.append(f"Fields: {', '.join(chunk.fields)}")
        parts.append(chunk.definition)
        return "\n".join(parts)

    def index_chunks(self, chunks: List[SchemaChunk]) -> int:
        """Index schema chunks into Weaviate"""
        if not chunks:
            return 0

        collection = self.client.collections.get(self.COLLECTION_NAME)

        with collection.batch.dynamic() as batch:
            for i, chunk in enumerate(chunks):
                text = self._chunk_to_text(chunk)

                properties = {
                    "typeName": chunk.name,
                    "typeKind": chunk.type,
                    "definition": chunk.definition,
                    "fields": chunk.fields,
                    "description": chunk.description or "",
                    "project": chunk.project or "",
                    "content": text
                }

                if self._local_embedder:
                    # Manual vectorization
                    vector = self._local_embedder.encode(text).tolist()
                    batch.add_object(
                        properties=properties,
                        vector=vector
                    )
                else:
                    # Auto-vectorization (Weaviate handles it)
                    batch.add_object(properties=properties)

        return len(chunks)

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.5,
        project: Optional[str] = None
    ) -> List[Dict]:
        """
        Hybrid search combining BM25 (keyword) and vector (semantic) search.

        Args:
            query: Search query
            top_k: Number of results
            alpha: Weight for vector vs BM25 (0.5 = equal, 0.0 = keyword only, 1.0 = semantic only)
            project: Optional project filter

        Returns:
            List of search results with scores
        """
        collection = self.client.collections.get(self.COLLECTION_NAME)

        # Build filters
        filters = None
        if project:
            from weaviate.classes.query import Filter
            filters = Filter.by_property("project").equal(project)

        # Get query vector if using local embedder
        query_vector = None
        if self._local_embedder:
            query_vector = self._local_embedder.encode(query).tolist()

        # Hybrid search
        if query_vector:
            # Manual hybrid: vector + BM25
            response = collection.query.hybrid(
                query=query,
                vector=query_vector,
                alpha=alpha,
                limit=top_k,
                filters=filters,
                return_metadata=MetadataQuery(score=True, explain_score=True)
            )
        else:
            # Auto hybrid (Weaviate handles vectorization)
            response = collection.query.hybrid(
                query=query,
                alpha=alpha,
                limit=top_k,
                filters=filters,
                return_metadata=MetadataQuery(score=True, explain_score=True)
            )

        results = []
        for obj in response.objects:
            results.append({
                "score": obj.metadata.score,
                "name": obj.properties.get("typeName", ""),
                "type": obj.properties.get("typeKind", ""),
                "definition": obj.properties.get("definition", ""),
                "fields": obj.properties.get("fields", []),
                "description": obj.properties.get("description", ""),
                "project": obj.properties.get("project", ""),
                "explain": obj.metadata.explain_score if hasattr(obj.metadata, 'explain_score') else None
            })

        return results

    def bm25_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Pure keyword search (BM25)"""
        collection = self.client.collections.get(self.COLLECTION_NAME)

        response = collection.query.bm25(
            query=query,
            query_properties=["content", "typeName", "description"],
            limit=top_k,
            return_metadata=MetadataQuery(score=True)
        )

        return self._parse_response(response)

    def vector_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Pure semantic/vector search"""
        collection = self.client.collections.get(self.COLLECTION_NAME)

        if self._local_embedder:
            query_vector = self._local_embedder.encode(query).tolist()
            response = collection.query.near_vector(
                near_vector=query_vector,
                limit=top_k,
                return_metadata=MetadataQuery(distance=True)
            )
        else:
            response = collection.query.near_text(
                query=query,
                limit=top_k,
                return_metadata=MetadataQuery(distance=True)
            )

        return self._parse_response(response)

    def _parse_response(self, response) -> List[Dict]:
        """Parse Weaviate response to dict list"""
        results = []
        for obj in response.objects:
            score = getattr(obj.metadata, 'score', None) or getattr(obj.metadata, 'distance', 0)
            results.append({
                "score": score,
                "name": obj.properties.get("typeName", ""),
                "type": obj.properties.get("typeKind", ""),
                "definition": obj.properties.get("definition", ""),
                "fields": obj.properties.get("fields", []),
                "description": obj.properties.get("description", ""),
            })
        return results

    def get_schema_context(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.5,
        project: Optional[str] = None
    ) -> str:
        """Get relevant schema context using hybrid search"""
        results = self.hybrid_search(query, top_k, alpha, project)

        context_parts = []
        for result in results:
            context_parts.append(f"# {result['type']} {result['name']}")
            if result['description']:
                context_parts.append(f"# {result['description']}")
            context_parts.append(result['definition'])
            context_parts.append("")

        return "\n".join(context_parts)

    def rag_query(
        self,
        query: str,
        prompt: str,
        top_k: int = 5,
        alpha: float = 0.5
    ) -> str:
        """
        Retrieval + Generation in one call (RAG).

        This retrieves relevant schema chunks and prompts an LLM to generate
        a response in a single step.
        """
        collection = self.client.collections.get(self.COLLECTION_NAME)

        if self._local_embedder:
            query_vector = self._local_embedder.encode(query).tolist()
            response = collection.generate.hybrid(
                query=query,
                vector=query_vector,
                alpha=alpha,
                limit=top_k,
                single_prompt=prompt,
                return_metadata=MetadataQuery(score=True)
            )
        else:
            response = collection.generate.hybrid(
                query=query,
                alpha=alpha,
                limit=top_k,
                single_prompt=prompt,
                return_metadata=MetadataQuery(score=True)
            )

        if response.objects:
            return response.objects[0].generated
        return ""

    def get_all_type_names(self, project: Optional[str] = None) -> List[str]:
        """Get all type names in the collection"""
        collection = self.client.collections.get(self.COLLECTION_NAME)

        filters = None
        if project:
            from weaviate.classes.query import Filter
            filters = Filter.by_property("project").equal(project)

        response = collection.query.fetch_objects(
            filters=filters,
            limit=1000,
            return_properties=["typeName"]
        )

        return list(set(obj.properties.get("typeName", "") for obj in response.objects))

    def graphql_query(self, gql_query: str) -> Dict:
        """
        Execute a raw GraphQL query against Weaviate.
        This is for advanced users who want to use Weaviate's native GraphQL API.
        """
        return self.client.graphql_raw_query(gql_query)

    def close(self):
        """Close the Weaviate connection"""
        self.client.close()


def parse_schema_file(file_path: str, project: Optional[str] = None) -> List[SchemaChunk]:
    """Parse a GraphQL schema file into chunks"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    return parse_schema_content(content, project)


def parse_schema_content(content: str, project: Optional[str] = None) -> List[SchemaChunk]:
    """Parse GraphQL schema content into chunks"""
    chunks = []

    patterns = {
        "type": r'type\s+(\w+)\s*(?:implements\s+[\w&\s]+)?\s*\{([^}]+)\}',
        "interface": r'interface\s+(\w+)\s*\{([^}]+)\}',
        "enum": r'enum\s+(\w+)\s*\{([^}]+)\}',
        "input": r'input\s+(\w+)\s*\{([^}]+)\}',
        "scalar": r'scalar\s+(\w+)',
    }

    field_pattern = r'(\w+)\s*(?:\([^)]*\))?\s*:\s*([\[\]\w!]+)'

    chunk_id = 0
    for type_kind, pattern in patterns.items():
        for match in re.finditer(pattern, content, re.DOTALL):
            if type_kind == "scalar":
                name = match.group(1)
                definition = match.group(0)
                fields = []
            else:
                name = match.group(1)
                body = match.group(2)
                definition = match.group(0)
                fields = re.findall(field_pattern, body)
                fields = [f"{f[0]}: {f[1]}" for f in fields]

            chunk = SchemaChunk(
                name=name,
                type=type_kind,
                definition=definition.strip(),
                fields=fields,
                project=project
            )
            chunks.append(chunk)
            chunk_id += 1

    return chunks


def main():
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Weaviate Schema Indexer")
    parser.add_argument("--schema", required=True, help="Path to GraphQL schema file")
    parser.add_argument("--project", default=None, help="Project name")
    parser.add_argument("--weaviate-url", default="http://localhost:8080")
    parser.add_argument("--search", default=None, help="Search query (optional)")
    parser.add_argument("--alpha", type=float, default=0.5, help="Hybrid search alpha (0=keyword, 1=semantic)")
    parser.add_argument("--use-openai", action="store_true", help="Use OpenAI for vectorization")

    args = parser.parse_args()

    indexer = WeaviateSchemaIndexer(
        weaviate_url=args.weaviate_url,
        use_openai_vectorizer=args.use_openai
    )

    if args.search:
        # Search mode
        print(f"\nHybrid Search (alpha={args.alpha}) for: '{args.search}'")
        print("=" * 60)

        results = indexer.hybrid_search(args.search, top_k=5, alpha=args.alpha)

        for r in results:
            print(f"\n{r['type']} {r['name']} (score: {r['score']:.4f})")
            if r['description']:
                print(f"  Description: {r['description'][:100]}...")
            print(f"  Fields: {', '.join(r['fields'][:5])}")
    else:
        # Index mode
        print("Creating collection...")
        indexer.create_collection()

        print(f"Parsing schema: {args.schema}")
        chunks = parse_schema_file(args.schema, args.project)
        print(f"Found {len(chunks)} schema definitions")

        print("Indexing...")
        count = indexer.index_chunks(chunks)
        print(f"[OK] Indexed {count} schema definitions into Weaviate")

    indexer.close()


if __name__ == "__main__":
    main()
