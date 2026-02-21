"""
GraphQL Schema Indexer for Qdrant Vector Database

This module parses GraphQL schema files and indexes them into Qdrant
for efficient semantic retrieval during query generation.
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    VectorParams,
    Distance,
    Filter,
    FieldCondition,
    MatchValue
)
from sentence_transformers import SentenceTransformer
import numpy as np


@dataclass
class SchemaChunk:
    """Represents a parsed schema definition chunk"""
    id: int
    name: str
    type: str  # type, interface, enum, union, input, scalar
    definition: str
    fields: List[str]
    description: Optional[str] = None
    project: Optional[str] = None


class SchemaParser:
    """Parses GraphQL schema files into structured chunks"""

    # Regex patterns for different GraphQL definition types
    PATTERNS = {
        "type": r'(type\s+(\w+)\s*(?:implements\s+[\w&\s]+)?\s*\{([^}]+)\})',
        "interface": r'(interface\s+(\w+)\s*\{([^}]+)\})',
        "enum": r'(enum\s+(\w+)\s*\{([^}]+)\})',
        "union": r'(union\s+(\w+)\s*=\s*([\w|\s]+))',
        "input": r'(input\s+(\w+)\s*\{([^}]+)\})',
        "scalar": r'(scalar\s+(\w+))',
        "query": r'(type\s+Query\s*\{([^}]+)\})',
        "mutation": r'(type\s+Mutation\s*\{([^}]+)\})',
        "subscription": r'(type\s+Subscription\s*\{([^}]+)\})',
    }

    # Regex for field extraction
    FIELD_PATTERN = r'(\w+)\s*(?:\([^)]*\))?\s*:\s*([\[\]\w!]+)'
    DESCRIPTION_PATTERN = r'"""([^"]*)"""'

    def __init__(self):
        self.chunks: List[SchemaChunk] = []

    def parse_file(self, file_path: str, project: Optional[str] = None) -> List[SchemaChunk]:
        """Parse a GraphQL schema file into chunks"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        return self.parse_content(content, project)

    def parse_content(self, content: str, project: Optional[str] = None) -> List[SchemaChunk]:
        """Parse GraphQL schema content into chunks"""
        self.chunks = []
        chunk_id = 0

        # Extract descriptions (docstrings)
        descriptions = self._extract_descriptions(content)

        for def_type, pattern in self.PATTERNS.items():
            matches = re.finditer(pattern, content, re.DOTALL)

            for match in matches:
                full_definition = match.group(1).strip()
                name = match.group(2) if len(match.groups()) > 1 else ""

                # Extract fields for types with body
                fields = []
                if len(match.groups()) > 2:
                    body = match.group(3)
                    fields = self._extract_fields(body)

                # Look for associated description
                description = descriptions.get(name)

                chunk = SchemaChunk(
                    id=chunk_id,
                    name=name,
                    type=def_type,
                    definition=full_definition,
                    fields=fields,
                    description=description,
                    project=project
                )
                self.chunks.append(chunk)
                chunk_id += 1

        return self.chunks

    def _extract_fields(self, body: str) -> List[str]:
        """Extract field definitions from a type body"""
        fields = []
        for match in re.finditer(self.FIELD_PATTERN, body):
            field_name = match.group(1)
            field_type = match.group(2)
            fields.append(f"{field_name}: {field_type}")
        return fields

    def _extract_descriptions(self, content: str) -> Dict[str, str]:
        """Extract type descriptions from triple-quoted strings"""
        descriptions = {}

        # Pattern to find descriptions followed by type definitions
        pattern = r'"""([^"]*)"""\s*(?:type|interface|enum|input)\s+(\w+)'
        for match in re.finditer(pattern, content, re.DOTALL):
            description = match.group(1).strip()
            type_name = match.group(2)
            descriptions[type_name] = description

        return descriptions


class SchemaIndexer:
    """Indexes GraphQL schema chunks into Qdrant vector database"""

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "graphql_schema",
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384
    ):
        self.client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        self.encoder = SentenceTransformer(embedding_model)
        self.embedding_dim = embedding_dim
        self.parser = SchemaParser()

    def create_collection(self, recreate: bool = True) -> None:
        """Create or recreate the Qdrant collection"""
        if recreate and self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            )

    def index_schema_file(
        self,
        file_path: str,
        project: Optional[str] = None
    ) -> int:
        """Index a GraphQL schema file into Qdrant"""
        chunks = self.parser.parse_file(file_path, project)
        return self._index_chunks(chunks)

    def index_schema_content(
        self,
        content: str,
        project: Optional[str] = None
    ) -> int:
        """Index GraphQL schema content into Qdrant"""
        chunks = self.parser.parse_content(content, project)
        return self._index_chunks(chunks)

    def _index_chunks(self, chunks: List[SchemaChunk]) -> int:
        """Index parsed schema chunks into Qdrant"""
        if not chunks:
            return 0

        # Create embeddings for chunks
        texts = [self._chunk_to_text(chunk) for chunk in chunks]
        embeddings = self.encoder.encode(texts)

        # Create points for Qdrant
        points = []
        for chunk, vector in zip(chunks, embeddings):
            points.append(PointStruct(
                id=chunk.id,
                vector=vector.tolist(),
                payload={
                    "name": chunk.name,
                    "type": chunk.type,
                    "definition": chunk.definition,
                    "fields": chunk.fields,
                    "description": chunk.description or "",
                    "project": chunk.project or "",
                    "text": self._chunk_to_text(chunk)
                }
            ))

        # Upsert to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        return len(points)

    def _chunk_to_text(self, chunk: SchemaChunk) -> str:
        """Convert a schema chunk to searchable text"""
        parts = []

        if chunk.description:
            parts.append(f"Description: {chunk.description}")

        parts.append(f"Type: {chunk.type} {chunk.name}")

        if chunk.fields:
            parts.append(f"Fields: {', '.join(chunk.fields[:10])}")  # Limit fields

        parts.append(chunk.definition)

        return "\n".join(parts)

    def search(
        self,
        query: str,
        top_k: int = 5,
        project: Optional[str] = None,
        type_filter: Optional[str] = None
    ) -> List[Dict]:
        """Search for relevant schema chunks"""
        query_vector = self.encoder.encode(query).tolist()

        # Build filter if specified
        query_filter = None
        if project or type_filter:
            conditions = []
            if project:
                conditions.append(
                    FieldCondition(key="project", match=MatchValue(value=project))
                )
            if type_filter:
                conditions.append(
                    FieldCondition(key="type", match=MatchValue(value=type_filter))
                )
            query_filter = Filter(must=conditions) if conditions else None

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter
        )

        return [
            {
                "score": hit.score,
                "name": hit.payload["name"],
                "type": hit.payload["type"],
                "definition": hit.payload["definition"],
                "fields": hit.payload["fields"],
                "description": hit.payload.get("description", ""),
                "project": hit.payload.get("project", "")
            }
            for hit in results
        ]

    def get_schema_context(
        self,
        query: str,
        top_k: int = 5,
        project: Optional[str] = None
    ) -> str:
        """Get relevant schema context for a query"""
        results = self.search(query, top_k, project)

        context_parts = []
        for result in results:
            context_parts.append(f"# {result['type']} {result['name']}")
            if result['description']:
                context_parts.append(f"# {result['description']}")
            context_parts.append(result['definition'])
            context_parts.append("")

        return "\n".join(context_parts)

    def get_all_type_names(self, project: Optional[str] = None) -> List[str]:
        """Get all type names in the index"""
        query_filter = None
        if project:
            query_filter = Filter(
                must=[FieldCondition(key="project", match=MatchValue(value=project))]
            )

        results = self.client.scroll(
            collection_name=self.collection_name,
            limit=1000,
            with_payload=["name"],
            query_filter=query_filter
        )

        return list(set(point.payload["name"] for point in results[0]))


def main():
    """CLI entry point for indexing schemas"""
    import argparse

    parser = argparse.ArgumentParser(description="Index GraphQL schemas into Qdrant")
    parser.add_argument("--schema", required=True, help="Path to GraphQL schema file")
    parser.add_argument("--project", default=None, help="Project name for the schema")
    parser.add_argument("--qdrant-url", default="http://localhost:6333", help="Qdrant URL")
    parser.add_argument("--collection", default="graphql_schema", help="Collection name")

    args = parser.parse_args()

    indexer = SchemaIndexer(
        qdrant_url=args.qdrant_url,
        collection_name=args.collection
    )

    indexer.create_collection()
    count = indexer.index_schema_file(args.schema, args.project)

    print(f"Indexed {count} schema definitions into {args.collection}")


if __name__ == "__main__":
    main()
