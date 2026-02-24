"""
Enhanced GraphQL Agent - Simplified with Hybrid Search

Implements:
1. Few-shot prompting with examples
2. Schema-aware RAG via Weaviate hybrid search
3. Safety boundaries for sensitive fields
4. Direct query generation (no intermediate intent extraction)
5. Query execution with proper headers for SubQuery endpoint
"""

import re
import json
from typing import Optional, Dict, Any, Tuple


# --- SubQuery Endpoint Headers ---

SUBQUERY_HEADERS = {
    'accept': 'application/json, multipart/mixed',
    'accept-language': 'en-US,en;q=0.9',
    'content-type': 'application/json',
    'origin': 'https://hermes.subquery.network',
    'referer': 'https://hermes.subquery.network/',
    'sec-ch-ua': '"Not(A:Brand";v="8", "Chromium";v="144", "Google Chrome";v="144"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'cross-site',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36',
    'x-endpoint': 'ask_mltn6ciom666gooh4vlxg9rwnbv20g'
}

SUBQUERY_ENDPOINT = "https://index-api.onfinality.io/sq/subquery/subquery-mainnet"


# --- Few-Shot Examples ---

FEW_SHOT_EXAMPLES = """
## Example 1: Count Query
Question: "How many indexers are currently active?"
Schema Context:
  type Indexer { id: ID!, active: Boolean!, totalStake: BigInt! }
  type Query { indexers(first: Int, filter: IndexerFilter): IndexersConnection! }
GraphQL Query:
```graphql
query {
  indexers(first: 1000, filter: { active: { equalTo: true } }) {
    totalCount
  }
}
```

## Example 2: Superlative Query
Question: "Which indexer has the highest total stake?"
Schema Context:
  type Indexer { id: ID!, totalStake: BigInt! }
  type Query { indexers(first: Int, orderBy: [IndexersOrderBy!]): IndexersConnection! }
GraphQL Query:
```graphql
query {
  indexers(first: 1, orderBy: [TOTAL_STAKE_DESC]) {
    nodes {
      id
      totalStake
    }
  }
}
```

## Example 3: Entity-Specific Query
Question: "What is the commission rate of indexer 0xe60554D90AF0e84A9C3d1A8643e41e49403945a6?"
Schema Context:
  type Indexer { id: ID!, commission: BigInt! }
  type Query { indexerById(id: ID!): Indexer }
GraphQL Query:
```graphql
query {
  indexerById(id: "0xe60554D90AF0e84A9C3d1A8643e41e49403945a6") {
    id
    commission
  }
}
```

## Example 4: Time-Travel Query (SubQL)
Question: "What was the total stake at block 5000000?"
Schema Context:
  type Indexer { id: ID!, totalStake: BigInt! }
  type Query { indexers(blockHeight: String): IndexersConnection! }
GraphQL Query:
```graphql
query {
  indexers(blockHeight: "5000000", first: 10) {
    nodes {
      id
      totalStake
    }
  }
}
```

## Example 5: Aggregation Query
Question: "What is the total delegated stake across all delegations?"
Schema Context:
  type Delegation { id: ID!, amount: BigInt! }
  type Query { delegations(first: Int): DelegationsConnection! }
GraphQL Query:
```graphql
query {
  delegations(first: 1000) {
    nodes {
      amount
    }
  }
}
```
"""


# --- Safety Configuration ---

SENSITIVE_FIELDS = [
    "privatekey", "secret", "password", "token", "apikey"
]


# --- Query Generation Prompt ---

QUERY_GENERATION_PROMPT = """You are an expert GraphQL query generator for blockchain data (SubQuery/PostGraphile protocol).

{few_shot_examples}

## Current Task

Generate a valid GraphQL query for:

**Question:** {question}

**Schema Context (from hybrid search):**
{schema_context}

**Protocol Rules:**
- Use `blockHeight: "NUMBER"` parameter for time-travel queries (SubQL format)
- Connection types return `nodes` array and `totalCount`
- Use `filter: {{ field: {{ operator: value }} }}` for filtering
- Use `orderBy: [FIELD_DESC]` or `orderBy: [FIELD_ASC]` for sorting

Output ONLY the GraphQL query, no explanations or markdown code blocks.

GraphQL Query:"""


class EnhancedGraphQLAgent:
    """
    Simplified GraphQL agent:
    - Trusts hybrid search for schema retrieval
    - No intermediate intent extraction
    - Direct question → query generation
    """

    def __init__(
        self,
        schema_indexer,
        llm_client,
        validator=None,
        executor=None
    ):
        self.indexer = schema_indexer
        self.llm = llm_client
        self.validator = validator

    def _check_safety(self, query: str) -> Tuple[bool, Optional[str]]:
        """Check if generated query accesses sensitive fields"""
        query_lower = query.lower()
        for field in SENSITIVE_FIELDS:
            # Match field as a word boundary (e.g., privateKey but not privateKeyHash)
            if re.search(rf'\b{field}\b', query_lower):
                return False, f"Query accesses sensitive field: {field}"
        return True, None

    async def generate_query(
        self,
        question: str,
        schema_context: str,
        protocol: str = "subql"
    ) -> str:
        """Generate GraphQL query directly from question and schema context"""
        prompt = QUERY_GENERATION_PROMPT.format(
            few_shot_examples=FEW_SHOT_EXAMPLES,
            question=question,
            schema_context=schema_context,
        )

        response = await self.llm.generate_async(prompt)
        return self._extract_query(response)

    def _extract_query(self, response: str) -> str:
        """Extract GraphQL query from LLM response"""
        # Try code blocks first
        code_block_pattern = r'```(?:graphql|gql)?\s*([\s\S]*?)```'
        matches = re.findall(code_block_pattern, response)
        if matches:
            return matches[0].strip()

        # Try to find query pattern
        query_pattern = r'(query\s+\w*\s*\{[\s\S]*\}|\{[\s\S]*\})'
        match = re.search(query_pattern, response)
        if match:
            return match.group(1).strip()

        return response.strip()

    async def answer_question(
        self,
        question: str,
        endpoint: str,
        protocol: str = "subql",
        block_height: Optional[int] = None,
        schema_context_k: int = 5,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """
        Main entry point - simplified pipeline:
        1. Retrieve relevant schema via hybrid search
        2. Generate query
        3. Check safety on generated query
        4. Validate generated query
        5. Execute query (if endpoint provided)
        6. Generate answer

        Args:
            question: Natural language question
            endpoint: GraphQL endpoint URL (pass True or SUBQUERY_ENDPOINT for real endpoint)
            protocol: GraphQL protocol type
            block_height: Optional block height for time-travel queries
            schema_context_k: Number of schema chunks to retrieve
            headers: Custom headers for the request (uses SUBQUERY_HEADERS if endpoint matches)
            timeout: Request timeout in seconds
        """
        import time
        start_time = time.time()

        result = {
            "answer": "",
            "query": None,
            "is_valid": True,
            "error": None,
            "elapsed_time": 0.0
        }

        try:
            # Step 1: Retrieve relevant schema via hybrid search
            search_query = question
            if block_height:
                search_query += f" block {block_height}"

            schema_context = self.indexer.get_schema_context(
                query=search_query,
                top_k=schema_context_k
            )

            # Step 2: Generate query
            query = await self.generate_query(
                question=question,
                schema_context=schema_context,
                protocol=protocol
            )
            result["query"] = query

            # Step 3: Check safety on generated query
            is_safe, safety_error = self._check_safety(query)
            if not is_safe:
                result["is_valid"] = False
                result["error"] = f"Safety violation: {safety_error}"
                result["elapsed_time"] = time.time() - start_time
                return result

            # Step 4: Validate generated query
            if self.validator:
                is_valid, error = self.validator.validate(query)
                if not is_valid:
                    result["is_valid"] = False
                    result["error"] = f"Generated query invalid: {error}"
                    result["elapsed_time"] = time.time() - start_time
                    return result

            # Step 5: Execute query if endpoint provided
            if endpoint:
                # Handle endpoint as boolean (use default SubQuery endpoint)
                if endpoint is True:
                    endpoint = SUBQUERY_ENDPOINT

                # Use SubQuery headers if endpoint matches or no custom headers provided
                request_headers = headers
                if request_headers is None and (endpoint == SUBQUERY_ENDPOINT or "onfinality.io" in endpoint):
                    request_headers = SUBQUERY_HEADERS
                elif request_headers is None:
                    request_headers = {"Content-Type": "application/json"}

                import httpx
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(
                        endpoint,
                        json={"query": query},
                        headers=request_headers
                    )
                    response.raise_for_status()
                    data = response.json()

                    if "errors" in data:
                        result["error"] = f"GraphQL error: {data['errors']}"
                    else:
                        result["query_result"] = data.get("data")

                        # Generate natural language answer
                        answer_prompt = f"""Based on the query result, provide a concise answer to the question.

Question: {question}

Result: {json.dumps(result["query_result"], indent=2)}

Answer (one sentence, direct and factual):"""

                        result["answer"] = await self.llm.generate_async(answer_prompt)

            result["elapsed_time"] = time.time() - start_time
            return result

        except Exception as e:
            result["is_valid"] = False
            result["error"] = str(e)
            result["elapsed_time"] = time.time() - start_time
            return result
