"""GraphQL agent components"""

# New ReAct-based agent (recommended)
from src.agent.react_agent import ReActGraphQLAgent, AgentResult
from src.agent.tools import (
    GraphQLSchemaInfoTool,
    GraphQLTypeDetailTool,
    GraphQLQueryExecuteTool,
    create_graphql_tools,
)
from src.agent.prompts import get_system_prompt, SUBQL_RULES, THEGRAPH_RULES

# Legacy agent (for backward compatibility)
from src.agent.enhanced_graphql_agent import EnhancedGraphQLAgent

__all__ = [
    # New agent (recommended)
    "ReActGraphQLAgent",
    "AgentResult",
    # Tools
    "GraphQLSchemaInfoTool",
    "GraphQLTypeDetailTool",
    "GraphQLQueryExecuteTool",
    "create_graphql_tools",
    # Prompts
    "get_system_prompt",
    "SUBQL_RULES",
    "THEGRAPH_RULES",
    # Legacy
    "EnhancedGraphQLAgent",
]
