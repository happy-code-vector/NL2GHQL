"""
ReAct-based GraphQL Agent with Error Recovery

Implements:
1. Tool-based schema introspection
2. LangGraph ReAct pattern for iterative reasoning
3. Automatic error recovery with retry
4. Support for both Gemini and OpenAI models
"""

import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from src.graphql.client import GraphQLClient
from src.agent.tools import create_graphql_tools
from src.agent.prompts import get_system_prompt
from src.llm.llm_client import LLMConfig, create_llm_client


@dataclass
class AgentResult:
    """Result from agent execution."""
    question: str
    graphql_query: Optional[str] = None
    query_result: Optional[Dict] = None
    answer: str = ""
    success: bool = False
    error: Optional[str] = None
    tool_calls: List[str] = None
    latency_ms: int = 0
    retries: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "question": self.question,
            "graphql_query": self.graphql_query,
            "query_result": self.query_result,
            "answer": self.answer,
            "success": self.success,
            "error": self.error,
            "tool_calls": self.tool_calls or [],
            "latency_ms": self.latency_ms,
            "retries": self.retries,
        }


class ReActGraphQLAgent:
    """
    ReAct agent for GraphQL query generation and execution.

    Uses LangGraph's ReAct pattern with tools for:
    - Schema introspection
    - Type exploration
    - Query validation and execution
    """

    def __init__(
        self,
        endpoint: str,
        headers: Optional[Dict[str, str]] = None,
        llm_config: Optional[LLMConfig] = None,
        max_retries: int = 3,
        recursion_limit: int = 12,
        enable_cache: bool = True,
    ):
        """Initialize the ReAct agent.

        Args:
            endpoint: GraphQL endpoint URL
            headers: Optional headers for authentication
            llm_config: LLM configuration (defaults to Gemini)
            max_retries: Maximum retry attempts on error
            recursion_limit: Maximum agent recursion depth
            enable_cache: Enable schema caching
        """
        self.endpoint = endpoint
        self.headers = headers or {"Content-Type": "application/json"}
        self.max_retries = max_retries
        self.recursion_limit = recursion_limit

        # Initialize LLM
        self.llm_config = llm_config or LLMConfig.from_env()
        self._init_llm()

        # Initialize GraphQL client
        self.client = GraphQLClient(
            endpoint=endpoint,
            headers=headers,
            enable_cache=enable_cache,
        )

        # Detect provider
        self.provider = self._detect_provider()

    def _init_llm(self):
        """Initialize the LLM based on provider."""
        if self.llm_config.provider.value == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            self.llm = ChatGoogleGenerativeAI(
                model=self.llm_config.model,
                google_api_key=self.llm_config.api_key,
                temperature=self.llm_config.temperature,
            )
        elif self.llm_config.provider.value == "openai":
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model=self.llm_config.model,
                api_key=self.llm_config.api_key,
                temperature=self.llm_config.temperature,
            )
        else:
            # Fallback to Gemini
            from langchain_google_genai import ChatGoogleGenerativeAI
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=self.llm_config.api_key,
                temperature=0.0,
            )

    def _detect_provider(self) -> str:
        """Detect GraphQL provider from endpoint."""
        endpoint = self.endpoint.lower()
        if "subquery" in endpoint or "subql" in endpoint or "onfinality" in endpoint:
            return "subql"
        elif "thegraph" in endpoint or "graph.network" in endpoint:
            return "thegraph"
        return "subql"  # Default

    async def query(
        self,
        question: str,
        block_height: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Process a natural language question and generate an answer.

        Args:
            question: Natural language question
            block_height: Optional block height for time-travel queries

        Returns:
            Dictionary with query, result, answer, and metadata
        """
        start_time = time.perf_counter()
        retries = 0
        last_error = None

        # Create tools with current block height
        tools = create_graphql_tools(self.client, block_height)

        # Get system prompt
        system_prompt = get_system_prompt(
            provider=self.provider,
            block_height=block_height
        )

        # Create ReAct agent
        agent = create_react_agent(
            model=self.llm,
            tools=tools,
        )

        # Retry loop for error recovery
        current_question = question
        for attempt in range(self.max_retries):
            try:
                # Build messages
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=current_question)
                ]

                # Invoke agent
                response = await agent.ainvoke(
                    {"messages": messages},
                    config={"recursion_limit": self.recursion_limit}
                )

                # Extract results
                response_messages = response.get("messages", [])
                final_message = response_messages[-1] if response_messages else None

                # Check for errors in tool responses
                has_error = False
                for msg in response_messages:
                    if hasattr(msg, "content") and "Error" in str(msg.content):
                        if "GraphQL Error" in str(msg.content):
                            has_error = True
                            last_error = msg.content
                            break

                # If error and retries remaining, try again with error context
                if has_error and attempt < self.max_retries - 1:
                    retries += 1
                    current_question = f"""Previous attempt failed with error:
{last_error}

Original question: {question}

Please fix the query and try again."""
                    continue

                # Extract answer
                answer = ""
                if final_message:
                    if isinstance(final_message, AIMessage):
                        answer = final_message.content or ""

                # Extract GraphQL query from tool calls
                graphql_query = self._extract_query(response_messages)

                # Extract query result
                query_result = self._extract_result(response_messages)

                # Extract tool calls made
                tool_calls = self._extract_tool_names(response_messages)

                latency_ms = int((time.perf_counter() - start_time) * 1000)

                return {
                    "question": question,
                    "graphql_query": graphql_query,
                    "query_result": query_result,
                    "answer": answer,
                    "tool_calls": tool_calls,
                    "success": True,
                    "error": None,
                    "latency_ms": latency_ms,
                    "retries": retries,
                }

            except Exception as e:
                last_error = str(e)
                retries += 1

                if attempt < self.max_retries - 1:
                    # Try again with simpler prompt
                    current_question = f"""Error occurred: {last_error}

Original question: {question}

Please try a simpler approach."""
                    continue

        # All retries failed
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        return {
            "question": question,
            "graphql_query": None,
            "query_result": None,
            "answer": None,
            "tool_calls": [],
            "success": False,
            "error": last_error,
            "latency_ms": latency_ms,
            "retries": retries,
        }

    def _extract_query(self, messages: List) -> Optional[str]:
        """Extract GraphQL query from tool call messages."""
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.get("name") == "graphql_query_validator_and_execute":
                        query = tc.get("args", {}).get("query", "")
                        if query:
                            return query
        return None

    def _extract_result(self, messages: List) -> Optional[Dict]:
        """Extract query result from tool response messages."""
        import json
        import re

        for msg in messages:
            if hasattr(msg, "name") and msg.name == "graphql_query_validator_and_execute":
                content = msg.content or ""
                if "QUERY RESULT" in content:
                    json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
                    if json_match:
                        try:
                            return json.loads(json_match.group(1))
                        except:
                            pass
        return None

    def _extract_tool_names(self, messages: List) -> List[str]:
        """Extract list of tools called."""
        tool_names = []
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    name = tc.get("name", "")
                    if name and name not in tool_names:
                        tool_names.append(name)
        return tool_names

    def clear_cache(self):
        """Clear schema cache."""
        self.client.clear_cache()
