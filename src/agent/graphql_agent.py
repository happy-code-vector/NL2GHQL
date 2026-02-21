"""
Hermes GraphQL Agent

A RAG-powered agent that answers natural language questions about blockchain data
by generating and executing GraphQL queries.
"""

import re
import json
import httpx
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum

from graphql import parse, validate, build_schema, GraphQLSyntaxError

from src.rag.schema_indexer import SchemaIndexer


class ProtocolType(Enum):
    SUBQL = "subql"
    THE_GRAPH = "the_graph"


@dataclass
class AgentResponse:
    """Response from the GraphQL agent"""
    answer: str
    query: Optional[str] = None
    query_result: Optional[Dict] = None
    is_valid: bool = True
    error: Optional[str] = None
    elapsed_time: float = 0.0


class GraphQLQueryValidator:
    """Validates GraphQL queries against schema"""

    def __init__(self, schema_content: str):
        self.schema = build_schema(schema_content)

    def validate(self, query: str) -> Tuple[bool, Optional[str]]:
        """Validate a GraphQL query"""
        try:
            ast = parse(query)
            errors = validate(self.schema, ast)
            if errors:
                return False, str(errors[0])
            return True, None
        except GraphQLSyntaxError as e:
            return False, f"Syntax Error: {e.message}"
        except Exception as e:
            return False, str(e)


class GraphQLExecutor:
    """Executes GraphQL queries against endpoints"""

    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)

    async def execute(
        self,
        endpoint: str,
        query: str,
        variables: Optional[Dict] = None
    ) -> Tuple[Optional[Dict], Optional[str]]:
        """Execute a GraphQL query"""
        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        try:
            response = await self.client.post(
                endpoint,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            result = response.json()

            if "errors" in result:
                return None, str(result["errors"])

            return result.get("data"), None

        except httpx.TimeoutException:
            return None, "Query timed out"
        except httpx.HTTPStatusError as e:
            return None, f"HTTP error: {e.response.status_code}"
        except Exception as e:
            return None, str(e)

    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()


class HermesGraphQLAgent:
    """
    RAG-powered GraphQL agent for answering blockchain questions.

    This agent:
    1. Understands natural language questions
    2. Retrieves relevant schema parts using RAG
    3. Generates valid GraphQL queries
    4. Executes queries against live endpoints
    5. Extracts answers from results
    """

    # System prompt for query generation
    QUERY_GEN_PROMPT = """You are an expert GraphQL query generator for blockchain data.
Given a natural language question and relevant schema context, generate a valid GraphQL query.

IMPORTANT RULES:
1. Only use types and fields that exist in the provided schema
2. For SubQL endpoints, use blockHeight parameter: `fieldName(blockHeight: "NUMBER")`
3. For The Graph endpoints, use block parameter: `fieldName(block: {{number: NUMBER}})`
4. Always include the required parameters
5. For count questions, use aggregation or count patterns
6. For "highest/lowest" questions, use orderBy and limit
7. Return ONLY the GraphQL query, no explanations

Schema Context:
{schema_context}

Protocol: {protocol}

Question: {question}

GraphQL Query:"""

    # System prompt for answer extraction
    ANSWER_GEN_PROMPT = """You are a helpful assistant that answers questions about blockchain data.

Question: {question}

GraphQL Query Used:
{query}

Query Result:
{result}

Provide a concise, direct answer to the question based on the query result.
Focus on the specific numerical value or entity requested.
Do not include explanations about the query or data source.

Answer:"""

    def __init__(
        self,
        schema_indexer: SchemaIndexer,
        llm_client,  # vLLM or similar client
        validator: Optional[GraphQLQueryValidator] = None,
        executor: Optional[GraphQLExecutor] = None
    ):
        self.indexer = schema_indexer
        self.llm = llm_client
        self.validator = validator
        self.executor = executor or GraphQLExecutor()

    async def answer_question(
        self,
        question: str,
        endpoint: str,
        protocol: ProtocolType = ProtocolType.SUBQL,
        block_height: Optional[int] = None,
        schema_context_k: int = 5,
        max_retries: int = 2
    ) -> AgentResponse:
        """
        Answer a natural language question about blockchain data.

        Args:
            question: The natural language question
            endpoint: GraphQL API endpoint URL
            protocol: Protocol type (subql or the_graph)
            block_height: Optional block height for time-travel queries
            schema_context_k: Number of schema types to retrieve
            max_retries: Maximum number of query generation retries

        Returns:
            AgentResponse with the answer and metadata
        """
        import time
        start_time = time.time()

        try:
            # Step 1: Retrieve relevant schema context
            schema_context = self.indexer.get_schema_context(
                query=question,
                top_k=schema_context_k
            )

            # Step 2: Generate GraphQL query
            query = await self._generate_query(
                question=question,
                schema_context=schema_context,
                protocol=protocol,
                block_height=block_height
            )

            if not query:
                return AgentResponse(
                    answer="",
                    is_valid=False,
                    error="Failed to generate query",
                    elapsed_time=time.time() - start_time
                )

            # Step 3: Validate query
            if self.validator:
                is_valid, error = self.validator.validate(query)
                if not is_valid:
                    # Retry with error context
                    if max_retries > 0:
                        return await self._retry_with_error(
                            question=question,
                            endpoint=endpoint,
                            protocol=protocol,
                            block_height=block_height,
                            error=error,
                            retries_left=max_retries - 1,
                            start_time=start_time
                        )

                    return AgentResponse(
                        answer="",
                        query=query,
                        is_valid=False,
                        error=f"Query validation failed: {error}",
                        elapsed_time=time.time() - start_time
                    )

            # Step 4: Execute query
            result, exec_error = await self.executor.execute(endpoint, query)

            if exec_error:
                return AgentResponse(
                    answer="",
                    query=query,
                    is_valid=False,
                    error=f"Query execution failed: {exec_error}",
                    elapsed_time=time.time() - start_time
                )

            # Step 5: Generate answer from result
            answer = await self._generate_answer(
                question=question,
                query=query,
                result=result
            )

            return AgentResponse(
                answer=answer,
                query=query,
                query_result=result,
                is_valid=True,
                elapsed_time=time.time() - start_time
            )

        except Exception as e:
            return AgentResponse(
                answer="",
                is_valid=False,
                error=str(e),
                elapsed_time=time.time() - start_time
            )

    async def _generate_query(
        self,
        question: str,
        schema_context: str,
        protocol: ProtocolType,
        block_height: Optional[int]
    ) -> Optional[str]:
        """Generate a GraphQL query using the LLM"""
        # Build the prompt
        block_instruction = ""
        if block_height:
            if protocol == ProtocolType.SUBQL:
                block_instruction = f"Use blockHeight: \"{block_height}\" for time-travel query."
            else:
                block_instruction = f"Use block: {{number: {block_height}}} for time-travel query."

        prompt = self.QUERY_GEN_PROMPT.format(
            schema_context=schema_context,
            protocol=protocol.value,
            question=question
        )

        if block_instruction:
            prompt += f"\n\n{block_instruction}"

        # Generate using LLM
        response = await self.llm.generate(prompt)

        # Extract query from response
        query = self._extract_query(response)
        return query

    def _extract_query(self, response: str) -> Optional[str]:
        """Extract GraphQL query from LLM response"""
        # Try to find query in code blocks
        code_block_pattern = r'```(?:graphql|gql)?\s*([\s\S]*?)```'
        matches = re.findall(code_block_pattern, response)

        if matches:
            return matches[0].strip()

        # Try to find query starting with query/mutation/{
        query_pattern = r'(query\s+\w*\s*\{[\s\S]*\}|mutation\s+\w*\s*\{[\s\S]*\}|\{[\s\S]*\})'
        match = re.search(query_pattern, response)

        if match:
            return match.group(1).strip()

        # Return the whole response if nothing found
        return response.strip() if response.strip() else None

    async def _generate_answer(
        self,
        question: str,
        query: str,
        result: Dict
    ) -> str:
        """Generate a natural language answer from query result"""
        prompt = self.ANSWER_GEN_PROMPT.format(
            question=question,
            query=query,
            result=json.dumps(result, indent=2)
        )

        response = await self.llm.generate(prompt)
        return response.strip()

    async def _retry_with_error(
        self,
        question: str,
        endpoint: str,
        protocol: ProtocolType,
        block_height: Optional[int],
        error: str,
        retries_left: int,
        start_time: float
    ) -> AgentResponse:
        """Retry query generation with error context"""
        # Add error to schema context for retry
        schema_context = self.indexer.get_schema_context(question)
        schema_context += f"\n\nPREVIOUS ERROR: {error}\nPlease fix the query."

        query = await self._generate_query(
            question=question,
            schema_context=schema_context,
            protocol=protocol,
            block_height=block_height
        )

        if not query:
            return AgentResponse(
                answer="",
                is_valid=False,
                error="Failed to generate query on retry",
                elapsed_time=time.time() - start_time
            )

        # Continue with execution
        result, exec_error = await self.executor.execute(endpoint, query)

        if exec_error:
            return AgentResponse(
                answer="",
                query=query,
                is_valid=False,
                error=f"Query execution failed: {exec_error}",
                elapsed_time=time.time() - start_time
            )

        answer = await self._generate_answer(question, query, result)

        return AgentResponse(
            answer=answer,
            query=query,
            query_result=result,
            is_valid=True,
            elapsed_time=time.time() - start_time
        )

    async def explore_schema(self, endpoint: str) -> Dict[str, Any]:
        """Explore the GraphQL schema at an endpoint"""
        introspection_query = """
        query IntrospectionQuery {
            __schema {
                types {
                    name
                    kind
                    description
                    fields {
                        name
                        type {
                            name
                            kind
                        }
                    }
                }
            }
        }
        """

        result, error = await self.executor.execute(endpoint, introspection_query)
        if error:
            return {"error": error}

        return result
