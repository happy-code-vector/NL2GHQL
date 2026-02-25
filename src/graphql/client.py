"""GraphQL Client with introspection and caching."""

import time
import json
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
import httpx


@dataclass
class SchemaCache:
    """Cache for GraphQL schema introspection."""
    enabled: bool = True
    schema: Optional[Dict] = None
    timestamp: float = 0
    ttl_seconds: int = 3600  # 1 hour

    def is_valid(self) -> bool:
        """Check if cache is still valid."""
        if not self.enabled or not self.schema:
            return False
        return (time.time() - self.timestamp) < self.ttl_seconds

    def set(self, schema: Dict):
        """Update cache."""
        self.schema = schema
        self.timestamp = time.time()

    def clear(self):
        """Clear cache."""
        self.schema = None
        self.timestamp = 0


class GraphQLClient:
    """GraphQL client with introspection and caching."""

    INTROSPECTION_QUERY = """
    query IntrospectionQuery {
      __schema {
        queryType { name }
        mutationType { name }
        types {
          ...FullType
        }
      }
    }

    fragment FullType on __Type {
      kind
      name
      description
      fields(includeDeprecated: true) {
        name
        description
        args {
          ...InputValue
        }
        type {
          ...TypeRef
        }
        isDeprecated
        deprecationReason
      }
      inputFields {
        ...InputValue
      }
      interfaces {
        ...TypeRef
      }
      enumValues(includeDeprecated: true) {
        name
        description
        isDeprecated
        deprecationReason
      }
      possibleTypes {
        ...TypeRef
      }
    }

    fragment InputValue on __InputValue {
      name
      description
      type {
        ...TypeRef
      }
      defaultValue
    }

    fragment TypeRef on __Type {
      kind
      name
      ofType {
        kind
        name
        ofType {
          kind
          name
          ofType {
            kind
            name
          }
        }
      }
    }
    """

    def __init__(
        self,
        endpoint: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
        enable_cache: bool = True,
    ):
        """Initialize GraphQL client.

        Args:
            endpoint: GraphQL API endpoint URL
            headers: Optional headers for authentication
            timeout: Request timeout in seconds
            enable_cache: Enable schema caching
        """
        self.endpoint = endpoint
        self.headers = headers or {"Content-Type": "application/json"}
        self.timeout = timeout
        self._cache = SchemaCache(enabled=enable_cache)

    async def introspect_schema(self) -> Dict[str, Any]:
        """Get full schema via introspection.

        Returns:
            Schema dictionary with types, queryType, mutationType
        """
        # Check cache first
        if self._cache.is_valid():
            return self._cache.schema

        # Perform introspection
        result = await self.execute(self.INTROSPECTION_QUERY)

        if "errors" in result:
            raise ValueError(f"Introspection failed: {result['errors']}")

        schema = result.get("data", {}).get("__schema", {})

        # Cache the result
        self._cache.set(schema)

        return schema

    async def execute(self, query: str, variables: Optional[Dict] = None) -> Dict:
        """Execute a GraphQL query.

        Args:
            query: GraphQL query string
            variables: Optional variables dict

        Returns:
            Query result dictionary
        """
        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                self.endpoint,
                json=payload,
                headers=self.headers,
            )
            response.raise_for_status()
            return response.json()

    def clear_cache(self):
        """Clear schema cache."""
        self._cache.clear()

    async def get_type_detail(self, type_name: str) -> Optional[Dict]:
        """Get details for a specific type.

        Args:
            type_name: Name of the GraphQL type

        Returns:
            Type definition dictionary or None
        """
        schema = await self.introspect_schema()
        types = schema.get("types", [])

        for t in types:
            if t.get("name") == type_name:
                return t

        return None

    async def get_queries(self) -> list:
        """Get all available queries.

        Returns:
            List of query field definitions
        """
        schema = await self.introspect_schema()
        query_type_name = schema.get("queryType", {}).get("name", "Query")
        query_type = await self.get_type_detail(query_type_name)

        if query_type:
            return query_type.get("fields", [])

        return []

    async def get_entity_types(self) -> list:
        """Get all entity types (non-connection, non-edge, non-filter types).

        Returns:
            List of type names
        """
        schema = await self.introspect_schema()
        types = schema.get("types", [])

        entity_types = []
        for t in types:
            name = t.get("name", "")

            # Skip introspection types and generated types
            if name.startswith("__"):
                continue
            if t.get("kind") != "OBJECT":
                continue
            if name.endswith("Connection"):
                continue
            if name.endswith("Edge"):
                continue
            if name.endswith("Filter"):
                continue
            if name.endswith("OrderBy"):
                continue
            if name in ["Query", "Mutation", "Subscription"]:
                continue

            entity_types.append(name)

        return entity_types
