"""LangChain tools for GraphQL operations."""

import json
import re
from typing import Optional, Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from src.graphql.client import GraphQLClient


class SchemaInfoInput(BaseModel):
    """Input for schema info tool."""
    pass


class TypeDetailInput(BaseModel):
    """Input for type detail tool."""
    type_name: str = Field(..., description="Name of the GraphQL type to get details for")


class QueryExecuteInput(BaseModel):
    """Input for query execute tool."""
    query: str = Field(..., description="GraphQL query string to validate and execute")


class GraphQLSchemaInfoTool(BaseTool):
    """Tool to get GraphQL schema information."""

    name: str = "graphql_schema_info"
    description: str = """Get comprehensive schema information including all types, queries, and mutations.
Use this FIRST to understand what data is available in the GraphQL API.
Returns a summary of all entity types and their main queries."""
    args_schema: Type[BaseModel] = SchemaInfoInput

    client: GraphQLClient = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, client: GraphQLClient):
        super().__init__()
        self.client = client

    def _run(self) -> str:
        """Get schema information synchronously."""
        import asyncio
        return asyncio.run(self._arun())

    async def _arun(self) -> str:
        """Get schema information asynchronously."""
        try:
            schema = await self.client.introspect_schema()
            return self._format_schema_info(schema)
        except Exception as e:
            return f"Error fetching schema: {str(e)}"

    def _format_schema_info(self, schema: dict) -> str:
        """Format schema info for display."""
        types = schema.get("types", [])
        query_type = schema.get("queryType", {}).get("name", "Query")

        query_type_info = None
        entity_types = []

        for t in types:
            type_name = t.get("name", "")

            if type_name.startswith("__"):
                continue

            if type_name == query_type:
                query_type_info = t
            elif t.get("kind") == "OBJECT" and not type_name.endswith("Connection"):
                if not type_name.endswith("Edge") and not type_name.endswith("Filter"):
                    entity_types.append(type_name)

        result = "SCHEMA OVERVIEW:\n\n"

        if query_type_info:
            result += "AVAILABLE QUERIES:\n"
            fields = query_type_info.get("fields", [])
            for field in fields[:30]:
                field_name = field.get("name", "")
                if not field_name.startswith("_"):
                    result += f"  - {field_name}\n"

        result += f"\nENTITY TYPES ({len(entity_types)}):\n"
        for et in entity_types[:20]:
            result += f"  - {et}\n"

        return result


class GraphQLTypeDetailTool(BaseTool):
    """Tool to get detailed type information."""

    name: str = "graphql_type_detail"
    description: str = """Get detailed information about a specific GraphQL type including all fields and their types.
Use this to explore entity structures and relationships before building queries.
Input: type_name (e.g., 'Indexer', 'Stake', 'Reward')"""
    args_schema: Type[BaseModel] = TypeDetailInput

    client: GraphQLClient = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, client: GraphQLClient):
        super().__init__()
        self.client = client

    def _run(self, type_name: str) -> str:
        """Get type detail synchronously."""
        import asyncio
        return asyncio.run(self._arun(type_name))

    async def _arun(self, type_name: str) -> str:
        """Get type detail asynchronously."""
        try:
            type_info = await self.client.get_type_detail(type_name)

            if not type_info:
                return f"Type '{type_name}' not found in schema."

            return self._format_type_detail(type_info)
        except Exception as e:
            return f"Error fetching type detail: {str(e)}"

    def _format_type_detail(self, type_info: dict) -> str:
        """Format type detail for display."""
        name = type_info.get("name", "Unknown")
        description = type_info.get("description", "")
        fields = type_info.get("fields", [])

        result = f"TYPE: {name}\n"

        if description:
            result += f"Description: {description}\n"

        result += "\nFIELDS:\n"

        for field in fields:
            field_name = field.get("name", "")
            field_type = self._format_type_ref(field.get("type", {}))

            if field.get("isDeprecated"):
                continue

            args = field.get("args", [])
            if args:
                args_str = ", ".join(
                    f"{a.get('name')}: {self._format_type_ref(a.get('type', {}))}"
                    for a in args
                )
                result += f"  - {field_name}({args_str}): {field_type}\n"
            else:
                result += f"  - {field_name}: {field_type}\n"

        return result

    def _format_type_ref(self, type_ref: dict) -> str:
        """Format a type reference for display."""
        kind = type_ref.get("kind", "")
        name = type_ref.get("name", "")
        of_type = type_ref.get("ofType")

        if kind == "NON_NULL":
            return f"{self._format_type_ref(of_type)}!"
        elif kind == "LIST":
            return f"[{self._format_type_ref(of_type)}]"
        else:
            return name or "Unknown"


class GraphQLQueryExecuteTool(BaseTool):
    """Tool to validate and execute GraphQL queries."""

    name: str = "graphql_query_validator_and_execute"
    description: str = """Validate and execute a GraphQL query.
Use this AFTER you have constructed a query based on schema information.
Returns the query results or error messages if the query is invalid.
Input: query (the complete GraphQL query string)"""
    args_schema: Type[BaseModel] = QueryExecuteInput

    client: GraphQLClient = None
    block_height: Optional[int] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        client: GraphQLClient,
        block_height: Optional[int] = None
    ):
        super().__init__()
        self.client = client
        self.block_height = block_height

    def _run(self, query: str) -> str:
        """Execute query synchronously."""
        import asyncio
        return asyncio.run(self._arun(query))

    async def _arun(self, query: str) -> str:
        """Execute query asynchronously."""
        try:
            processed_query = self._process_query(query)

            result = await self.client.execute(processed_query)

            if "errors" in result:
                return f"GraphQL Error: {json.dumps(result['errors'], indent=2)}"

            if not result.get("data"):
                return "Query executed successfully but returned no data."

            return f"QUERY RESULT:\n```json\n{json.dumps(result.get('data'), indent=2, default=str)}\n```"

        except ValueError as e:
            return f"GraphQL Error: {str(e)}"
        except Exception as e:
            return f"Execution Error: {str(e)}"

    def _process_query(self, query: str) -> str:
        """Process query to inject block height if needed."""
        if not self.block_height:
            return query

        if "blockHeight" not in query:
            pattern = r'(\w+)\s*\(([^)]*)\)'
            match = re.search(pattern, query)
            if match:
                entity_name = match.group(1)
                existing_args = match.group(2)

                if existing_args:
                    new_args = f'blockHeight: "{self.block_height}", {existing_args}'
                else:
                    new_args = f'blockHeight: "{self.block_height}"'

                query = query.replace(
                    f"{entity_name}({existing_args})",
                    f"{entity_name}({new_args})",
                    1
                )

        return query


def create_graphql_tools(
    client: GraphQLClient,
    block_height: Optional[int] = None
) -> list:
    """Create all GraphQL tools.

    Args:
        client: GraphQL client instance
        block_height: Optional block height for time-travel queries

    Returns:
        List of LangChain tools
    """
    return [
        GraphQLSchemaInfoTool(client),
        GraphQLTypeDetailTool(client),
        GraphQLQueryExecuteTool(client, block_height),
    ]
