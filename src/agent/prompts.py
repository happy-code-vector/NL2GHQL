"""System prompts and templates for GraphQL agent."""

from typing import Optional


# PostGraphile/SubQL specific rules
SUBQL_RULES = """
POSTGRAPHILE v4 INFERENCE RULES:

- Each @entity type -> database table with 2 queries: singular(id) & plural(filter/pagination)
- Fields with @derivedFrom -> relationship fields, need subfield selection
- Foreign key fields ending in 'Id' -> direct ID access
- System tables (_pois, _metadatas, _metadata) -> ignore these

POSTGRAPHILE v4 QUERY PATTERNS:

1. ENTITY QUERIES:
   - Single query: entityName(id: ID!) -> EntityType
   - Collection query: entityNames(first: Int, filter: EntityFilter, orderBy: [EntityOrderBy!]) -> EntityConnection
   - PLURAL NAMING: If entity ends with 's' (e.g., Series), plural adds 'es' (e.g., serieses). Follow standard English pluralization rules.
   - Multiple queries: You can send multiple independent queries in a single GraphQL request if they have no data dependencies between them

2. RELATIONSHIP QUERIES:
   - Foreign key ID: fieldNameId (returns ID directly)
   - Single entity: fieldName { id, otherFields }
   - Collection relationships: fieldName { nodes { id, otherFields }, pageInfo { hasNextPage, endCursor }, totalCount }
   - With filters: fieldName(filter: { ... }) { nodes { ... }, totalCount }

3. FILTER PATTERNS (PostGraphile Format):

   STRING FILTERS:
   - equalTo, notEqualTo, distinctFrom, notDistinctFrom
   - in: [String!], notIn: [String!]
   - lessThan, lessThanOrEqualTo, greaterThan, greaterThanOrEqualTo
   - Case insensitive: equalToInsensitive, inInsensitive, etc.
   - isNull: Boolean

   BIGINT/NUMBER FILTERS:
   - equalTo, notEqualTo, distinctFrom, notDistinctFrom
   - lessThan, lessThanOrEqualTo, greaterThan, greaterThanOrEqualTo
   - in: [BigInt!], notIn: [BigInt!]
   - isNull: Boolean

   BOOLEAN FILTERS:
   - equalTo, notEqualTo, distinctFrom, notDistinctFrom
   - in: [Boolean!], notIn: [Boolean!]
   - isNull: Boolean

   EXAMPLES:
   - { id: { equalTo: "0x123" } }
   - { status: { in: ["active", "pending"] } }
   - { count: { greaterThan: 100 } }
   - { name: { equalToInsensitive: "alice" } }

4. AGGREGATION PATTERNS:

   CONNECTION-LEVEL AGGREGATES (on connection types):
   - Total count: entityNames { totalCount }
   - Aggregates: entityNames { aggregates { sum { field } avg { field } max { field } min { field } count } }
   - Grouped: entityNames { groupedAggregates(groupBy: [FIELD]) { keys sum { field } } }

   EXAMPLES:
   - Count: indexers { totalCount }
   - Sum: indexerRewards { aggregates { sum { amount } } }
   - Max: indexerRewards { aggregates { max { amount } } }
   - Grouped: rewards { groupedAggregates(groupBy: [INDEXER_ID]) { keys sum { amount } } }

5. PAGINATION:
   - Forward: first: Int, after: Cursor
   - Backward: last: Int, before: Cursor
   - PageInfo: hasNextPage, hasPreviousPage, startCursor, endCursor

6. TIME-TRAVEL QUERIES (SubQL):
   - Use blockHeight parameter: entityNames(blockHeight: "12345") { nodes { id } }
   - blockHeight must be a string number
   - All queries in a request should use the same blockHeight if needed

COMMON MISTAKES TO AVOID:
- Don't use 'edges' -> use 'nodes' directly
- Don't use 'where:' -> use 'filter:'
- Don't forget 'nodes' inside connection types
- Don't use 'orderDirection' -> use 'orderBy: [FIELD_DESC]' or 'orderBy: [FIELD_ASC]'
"""

# The Graph specific rules
THEGRAPH_RULES = """
THE GRAPH QUERY PATTERNS:

1. ENTITY QUERIES:
   - Single: entity(id: ID!) -> EntityType
   - Collection: entities(first: Int, where: EntityFilter, orderBy: String, orderDirection: String) -> [EntityType]
   - Multiple: Can combine multiple queries in one request

2. FILTER PATTERNS (The Graph Format):
   - Plain field: { field: value }
   - Operators: field_gt, field_gte, field_lt, field_lte, field_in, field_not, field_contains
   - String: field_startsWith, field_endsWith, field_contains, field_not_contains

   EXAMPLES:
   - { status: "active" }
   - { amount_gt: 100 }
   - { id_in: ["0x1", "0x2"] }

3. TIME-TRAVEL QUERIES (The Graph):
   - Use block parameter: entities(block: { number: 12345 }) { id }
   - Or block hash: entities(block: { hash: "0x..." }) { id }
"""

# Block height rules for SubQL
BLOCK_HEIGHT_RULES = """
BLOCK HEIGHT RULES (IMPORTANT):

When a blockHeight is specified, you MUST include it in your GraphQL query.

FOR SUBQL/POSTGRAPHILE:
- Add blockHeight parameter to the query
- Format: entityNames(blockHeight: "12345") { ... }
- The blockHeight value is always a STRING number

EXAMPLES:
CORRECT: { indexers(blockHeight: "5000000") { nodes { id totalStake } } }
CORRECT: { rewards(blockHeight: "5000000", filter: { eraIdx: { equalTo: 100 } }) { nodes { amount } } }

WRONG: { indexers { id } }  # Missing blockHeight when specified
WRONG: { indexers(blockHeight: 5000000) { id } }  # Should be string
"""


def get_system_prompt(
    provider: str = "subql",
    block_height: Optional[int] = None
) -> str:
    """Build the system prompt based on provider and block height.

    Args:
        provider: GraphQL provider type ('subql' or 'thegraph')
        block_height: Optional block height for time-travel queries

    Returns:
        System prompt string
    """
    base_prompt = """You are an expert GraphQL query generator and executor for blockchain data.
Your task is to understand natural language questions and generate accurate GraphQL queries.

## AVAILABLE TOOLS:
1. graphql_schema_info - Get comprehensive schema overview
2. graphql_type_detail - Get detailed information about a specific type
3. graphql_query_validator_and_execute - Validate and execute a GraphQL query

## YOUR APPROACH:
1. First, use graphql_schema_info to understand the available types and queries
2. If needed, use graphql_type_detail to explore specific types and their relationships
3. Construct a valid GraphQL query based on the schema
4. Use graphql_query_validator_and_execute to validate and execute your query
5. If there are errors, analyze them and retry with corrections
6. Provide a clear, natural language answer based on the query results

## IMPORTANT RULES:
- Always use the exact field names from the schema
- Use 'nodes' (not 'edges') for connection types in SubQL
- Use 'filter' (not 'where') for SubQL filters
- Always include blockHeight parameter when specified
- Provide concise, factual answers based on actual query results
"""

    # Add provider-specific rules
    if provider == "subql":
        base_prompt += f"\n\n{SUBQL_RULES}"
    elif provider == "thegraph":
        base_prompt += f"\n\n{THEGRAPH_RULES}"

    # Add block height rules if specified
    if block_height:
        base_prompt += f"\n\n{BLOCK_HEIGHT_RULES}"
        base_prompt += f"\n\n**CURRENT BLOCK HEIGHT: {block_height}**"
        base_prompt += "\nYou MUST include blockHeight: \"" + str(block_height) + "\" in ALL queries."

    return base_prompt


def get_block_rule_prompt(block_height: Optional[int], provider: str) -> str:
    """Get block height rule for a specific query.

    Args:
        block_height: Block height value
        provider: Provider type (subql or thegraph)

    Returns:
        Block rule string
    """
    if not block_height:
        return ""

    if provider == "subql":
        return f"""
BLOCK HEIGHT CONSTRAINT:
- You MUST use blockHeight: "{block_height}" in your queries
- Example: indexers(blockHeight: "{block_height}") {{ nodes {{ id }} }}
"""
    elif provider == "thegraph":
        return f"""
BLOCK HEIGHT CONSTRAINT:
- You MUST use block: {{ number: {block_height} }} in your queries
- Example: entities(block: {{ number: {block_height} }}) {{ id }}
"""

    return ""
