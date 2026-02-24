"""
Hybrid Schema Indexer with Qdrant

Implements Hybrid Search (Dense + Sparse/BM25) for better GraphQL schema retrieval.
This addresses the "exact field name" problem mentioned in Gemini's response.

Hybrid Search = Vector Search (semantic) + Keyword Search (BM25)
- User asks "revenue" → Vector finds `totalRevenue` semantically
- User asks "totalRevenue" → Keyword search finds exact field name
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    Distance,
    Filter,
    FieldCondition,
    MatchValue,
    MatchText
)
from sentence_transformers import SentenceTransformer
import numpy as np


@dataclass
class SchemaChunk:
    """Represents a parsed schema definition chunk"""
    id: int
    name: str
    type: str
    definition: str
    fields: List[str]
    description: Optional[str] = None
    project: Optional[str] = None


class HybridSchemaIndexer:
    """
    Hybrid Schema Indexer using Qdrant with Dense + Sparse vectors.

    Dense vectors: Semantic understanding (via sentence-transformers)
    Sparse vectors: Keyword matching (BM25-like, via Qdrant's sparse index)
    """

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "graphql_schema_hybrid",
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384
    ):
        # Use gRPC for better performance and Windows compatibility
        self.client = QdrantClient(
            host="127.0.0.1",
            port=6333,
            grpc_port=6334,
            prefer_grpc=True
        )
        self.collection_name = collection_name
        self.encoder = SentenceTransformer(embedding_model)
        self.embedding_dim = embedding_dim

    def create_collection(self, recreate: bool = True) -> None:
        """Create collection with both dense and sparse vectors"""
        if recreate and self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                },
                # Sparse vectors for keyword/BM25 search
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(on_disk=False)
                    )
                }
            )

    def _text_to_sparse_vector(self, text: str) -> Dict[int, float]:
        """
        Convert text to sparse vector (simple BM25-like tokenization).
        In production, use a proper BM25 encoder like SPLADE.
        """
        tokens = re.findall(r'\b\w+\b', text.lower())
        token_freq = {}
        for token in tokens:
            # Simple hash-based token ID
            token_id = hash(token) % (2**31)
            token_freq[token_id] = token_freq.get(token_id, 0) + 1

        # Normalize frequencies
        max_freq = max(token_freq.values()) if token_freq else 1
        return {k: v / max_freq for k, v in token_freq.items()}

    def _chunk_to_text(self, chunk: SchemaChunk) -> str:
        """Convert chunk to searchable text"""
        parts = []
        if chunk.description:
            parts.append(f"Description: {chunk.description}")
        parts.append(f"Type: {chunk.type} {chunk.name}")
        if chunk.fields:
            parts.append(f"Fields: {', '.join(chunk.fields[:15])}")
        parts.append(chunk.definition)
        return "\n".join(parts)

    def index_chunks(self, chunks: List[SchemaChunk]) -> int:
        """Index schema chunks with both dense and sparse vectors"""
        if not chunks:
            return 0

        points = []
        for chunk in chunks:
            text = self._chunk_to_text(chunk)

            # Dense vector (semantic)
            dense_vector = self.encoder.encode(text).tolist()

            # Sparse vector (keyword)
            sparse_vector = self._text_to_sparse_vector(text)

            points.append(PointStruct(
                id=chunk.id,
                vector={
                    "dense": dense_vector,
                    "sparse": sparse_vector
                },
                payload={
                    "name": chunk.name,
                    "type": chunk.type,
                    "definition": chunk.definition,
                    "fields": chunk.fields,
                    "description": chunk.description or "",
                    "project": chunk.project or "",
                    "text": text
                }
            ))

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        return len(points)

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.5,
        project: Optional[str] = None
    ) -> List[Dict]:
        """
        Hybrid search combining dense (semantic) and sparse (keyword) vectors.

        Args:
            query: Search query
            top_k: Number of results
            alpha: Weight for dense vs sparse (0.5 = equal, 0.0 = keyword only, 1.0 = semantic only)
            project: Optional project filter

        Returns:
            List of search results with scores
        """
        # Dense query
        dense_vector = self.encoder.encode(query).tolist()

        # Sparse query
        sparse_vector = self._text_to_sparse_vector(query)

        # Build filter
        query_filter = None
        if project:
            query_filter = Filter(
                must=[FieldCondition(key="project", match=MatchValue(value=project))]
            )

        # Perform hybrid search
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=dense_vector,
            using="dense",
            limit=top_k,
            query_filter=query_filter,
            with_payload=True
        )

        # Also get sparse results
        sparse_results = self.client.query_points(
            collection_name=self.collection_name,
            query=sparse_vector,
            using="sparse",
            limit=top_k,
            query_filter=query_filter,
            with_payload=True
        )

        # Fuse results (Reciprocal Rank Fusion)
        return self._fuse_results(
            dense_results=results.points,
            sparse_results=sparse_results.points,
            alpha=alpha,
            top_k=top_k
        )

    def _fuse_results(
        self,
        dense_results: List,
        sparse_results: List,
        alpha: float,
        top_k: int
    ) -> List[Dict]:
        """Fuse dense and sparse results using Reciprocal Rank Fusion"""
        scores = {}
        k = 60  # RRF constant

        # Score dense results
        for rank, result in enumerate(dense_results):
            point_id = result.id
            rrf_score = alpha / (k + rank + 1)
            if point_id not in scores:
                scores[point_id] = {"score": 0, "payload": result.payload}
            scores[point_id]["score"] += rrf_score

        # Score sparse results
        for rank, result in enumerate(sparse_results):
            point_id = result.id
            rrf_score = (1 - alpha) / (k + rank + 1)
            if point_id not in scores:
                scores[point_id] = {"score": 0, "payload": result.payload}
            scores[point_id]["score"] += rrf_score

        # Sort by fused score
        sorted_results = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)

        return [
            {
                "id": pid,
                "score": data["score"],
                "name": data["payload"].get("name", ""),
                "type": data["payload"].get("type", ""),
                "definition": data["payload"].get("definition", ""),
                "fields": data["payload"].get("fields", []),
                "description": data["payload"].get("description", ""),
            }
            for pid, data in sorted_results[:top_k]
        ]

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
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Hybrid Schema Indexer")
    parser.add_argument("--schema", required=False, help="Path to GraphQL schema file (required for indexing)")
    parser.add_argument("--project", default=None, help="Project name")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--collection", default="graphql_schema_hybrid")
    parser.add_argument("--search", default=None, help="Search query (optional)")

    args = parser.parse_args()

    indexer = HybridSchemaIndexer(
        qdrant_url=args.qdrant_url,
        collection_name=args.collection
    )

    if args.search:
        # Search mode
        results = indexer.hybrid_search(args.search, top_k=5)
        print(f"\nSearch results for: '{args.search}'")
        for r in results:
            print(f"  - {r['name']} (score: {r['score']:.4f})")
    else:
        # Index mode - requires schema
        if not args.schema:
            print("Error: --schema is required for indexing mode")
            print("Usage: python -m src.rag.hybrid_indexer --schema path/to/schema.graphql")
            return

        indexer.create_collection()

        # Parse schema
        from src.rag.schema_indexer import SchemaParser
        parser_obj = SchemaParser()
        chunks = parser_obj.parse_file(args.schema, args.project)

        count = indexer.index_chunks(chunks)
        print(f"Indexed {count} schema definitions with hybrid search")


if __name__ == "__main__":
    main()
