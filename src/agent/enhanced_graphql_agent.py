"""
Enhanced GraphQL Agent with Gemini's Recommendations

Implements:
1. Few-shot prompting with examples
2. Intent-first validation approach
3. Schema-aware RAG with JIT assembly
4. Safety boundaries for sensitive fields
"""

import re
import json
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
from pydantic import BaseModel, Field

from graphql import parse, validate, build_schema, GraphQLSyntaxError


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


# --- Intent Schema ---

class QueryIntent(BaseModel):
    """Structured intent for GraphQL query generation"""
    operation: str = Field(..., description="Operation type: query, mutation, subscription")
    entity: str = Field(..., description="Main entity to query (e.g., 'indexers', 'delegations')")
    fields: List[str] = Field(default_factory=list, description="Fields to select")
    filters: Optional[Dict[str, Any]] = Field(None, description="Filter conditions")
    order_by: Optional[List[Dict[str, str]]] = Field(None, description="Ordering criteria")
    limit: Optional[int] = Field(None, description="Maximum results to return")
    block_height: Optional[int] = Field(None, description="Block height for time-travel")
    aggregation: Optional[str] = Field(None, description="Aggregation type: count, sum, avg")
    is_valid: bool = Field(True, description="Whether intent is valid")
    error: Optional[str] = Field(None, description="Validation error if any")


# --- Safety Configuration ---

SENSITIVE_FIELDS = {
    # Fields that should never be queried
    "blacklist": ["privateKey", "secret", "password", "token", "apiKey"],
    # Fields that require special handling
    "restricted": ["balance", "wallet", "reward"],
}

SAFE_ENTITIES = [
    "indexers", "delegations", "delegators", "eras", "deployments",
    "projects", "rewards", "transfers", "stake"
]


# --- Enhanced Prompts ---

INTENT_EXTRACTION_PROMPT = """You are an expert at extracting structured intent from natural language questions about blockchain data.

Analyze the question and extract the query intent as JSON.

Output ONLY a valid JSON object with this structure:
{{
  "operation": "query",
  "entity": "<main entity name, plural form>",
  "fields": ["<field1>", "<field2>"],
  "filters": {{"<field>": {{"<operator>": <value>}}}},
  "order_by": [{{"<field>": "ASC|DESC"}}],
  "limit": <number or null>,
  "block_height": <number or null>,
  "aggregation": "<count|sum|avg|null>"
}}

Examples:
- "How many X?" → aggregation: "count"
- "Which X has the highest Y?" → order_by: [{{"Y": "DESC"}}], limit: 1
- "What is the total Y of X?" → fields: ["Y"], entity: "X"
- "What was X at block 5000000?" → block_height: 5000000

Question: {question}

Available entities: {entities}

JSON Intent:"""


QUERY_GENERATION_PROMPT = """You are an expert GraphQL query generator for blockchain data (SubQuery/PostGraphile protocol).

{few_shot_examples}

## Current Task

Generate a valid GraphQL query for:

**Question:** {question}

**Schema Context:**
{schema_context}

**Extracted Intent:**
{intent}

**Protocol Rules:**
- Use `blockHeight: "NUMBER"` parameter for time-travel queries (SubQL format)
- Connection types return `nodes` array and `totalCount`
- Use `filter: {{ field: {{ operator: value }} }}` for filtering
- Use `orderBy: [FIELD_DESC]` or `orderBy: [FIELD_ASC]` for sorting

Output ONLY the GraphQL query, no explanations or markdown code blocks.

GraphQL Query:"""


class EnhancedGraphQLAgent:
    """
    Enhanced GraphQL agent implementing Gemini's recommendations:
    1. Few-shot prompting
    2. Intent-first validation
    3. Safety boundaries
    4. Schema-aware RAG
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

        # Cache available entities
        self.available_entities = []
        self._load_available_entities()

    def _load_available_entities(self):
        """Load list of available entities from indexer"""
        try:
            self.available_entities = self.indexer.get_all_type_names()
            # Filter to queryable entities (usually plural)
            self.available_entities = [
                e for e in self.available_entities
                if not e.startswith('__') and e[0].islower()
            ]
        except Exception:
            self.available_entities = SAFE_ENTITIES

    def _check_safety(self, question: str, intent: QueryIntent) -> Tuple[bool, Optional[str]]:
        """Check if query is safe to execute"""
        # Check for blacklisted fields in question
        question_lower = question.lower()
        for field in SENSITIVE_FIELDS["blacklist"]:
            if field.lower() in question_lower:
                return False, f"Query contains sensitive field: {field}"

        # Check if entity is in safe list
        if intent.entity and intent.entity not in SAFE_ENTITIES:
            # Allow but log warning for unknown entities
            pass

        return True, None

    async def extract_intent(
        self,
        question: str,
        schema_context: str = ""
    ) -> QueryIntent:
        """Extract structured intent from natural language question"""
        prompt = INTENT_EXTRACTION_PROMPT.format(
            question=question,
            entities=", ".join(self.available_entities[:50])  # Limit to avoid token bloat
        )

        response = await self.llm.generate_async(prompt)

        try:
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                intent_dict = json.loads(json_match.group())
                return QueryIntent(**intent_dict)
            else:
                return QueryIntent(
                    operation="query",
                    entity="",
                    is_valid=False,
                    error="Could not extract intent from response"
                )
        except Exception as e:
            return QueryIntent(
                operation="query",
                entity="",
                is_valid=False,
                error=f"Intent parsing error: {str(e)}"
            )

    def validate_intent(self, intent: QueryIntent) -> Tuple[bool, Optional[str]]:
        """Validate intent against schema"""
        if not intent.entity:
            return False, "No entity specified"

        if intent.entity not in self.available_entities:
            # Try to find similar entity
            similar = [e for e in self.available_entities if intent.entity.lower() in e.lower()]
            if similar:
                return False, f"Unknown entity '{intent.entity}'. Did you mean: {similar[:3]}?"
            return False, f"Unknown entity: {intent.entity}"

        return True, None

    async def generate_query(
        self,
        question: str,
        schema_context: str,
        intent: QueryIntent,
        protocol: str = "subql"
    ) -> str:
        """Generate GraphQL query from intent"""
        prompt = QUERY_GENERATION_PROMPT.format(
            few_shot_examples=FEW_SHOT_EXAMPLES,
            question=question,
            schema_context=schema_context,
            intent=intent.model_dump_json(indent=2),
        )

        response = await self.llm.generate_async(prompt)
        query = self._extract_query(response)

        return query

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

    def build_minimal_schema(
        self,
        entity: str,
        fields: List[str],
        full_schema_context: str
    ) -> str:
        """Build minimal schema context for specific query (JIT Assembly)"""
        lines = []

        # Extract relevant type definitions
        entity_pattern = rf'(type\s+{entity}[^\{{]*\{{[^}}]*\}})'
        matches = re.findall(entity_pattern, full_schema_context, re.IGNORECASE)

        if matches:
            lines.append("# Relevant Schema:")
            lines.append(matches[0])

        # Extract query field for this entity
        query_pattern = rf'(\w*{entity}\w*\([^)]*\)\s*:\s*\w+)'
        query_matches = re.findall(query_pattern, full_schema_context, re.IGNORECASE)

        if query_matches:
            lines.append("# Query Fields:")
            for qm in query_matches[:3]:  # Limit to 3
                lines.append(f"  {qm}")

        # Add filter type if relevant
        if fields:
            lines.append("# Requested Fields:")
            lines.append(f"  Fields: {', '.join(fields)}")

        return "\n".join(lines)

    async def answer_question(
        self,
        question: str,
        endpoint: str,
        protocol: str = "subql",
        block_height: Optional[int] = None,
        schema_context_k: int = 5
    ) -> Dict[str, Any]:
        """
        Main entry point - implements full pipeline:
        1. Extract intent from question
        2. Validate intent
        3. Check safety
        4. Retrieve relevant schema
        5. Generate query
        6. Execute query (if endpoint provided)
        7. Generate answer
        """
        import time
        start_time = time.time()

        result = {
            "answer": "",
            "query": None,
            "intent": None,
            "is_valid": True,
            "error": None,
            "elapsed_time": 0.0
        }

        try:
            # Step 1: Extract intent
            intent = await self.extract_intent(question)
            result["intent"] = intent.model_dump()

            if not intent.is_valid:
                result["is_valid"] = False
                result["error"] = intent.error
                result["elapsed_time"] = time.time() - start_time
                return result

            # Step 2: Validate intent
            is_valid, error = self.validate_intent(intent)
            if not is_valid:
                result["is_valid"] = False
                result["error"] = error
                result["elapsed_time"] = time.time() - start_time
                return result

            # Step 3: Check safety
            is_safe, safety_error = self._check_safety(question, intent)
            if not is_safe:
                result["is_valid"] = False
                result["error"] = f"Safety violation: {safety_error}"
                result["elapsed_time"] = time.time() - start_time
                return result

            # Override block_height if provided
            if block_height:
                intent.block_height = block_height

            # Step 4: Retrieve relevant schema
            full_schema = self.indexer.get_schema_context(
                query=f"{intent.entity} {' '.join(intent.fields)}",
                top_k=schema_context_k
            )

            # Step 5: Build minimal schema (JIT Assembly)
            minimal_schema = self.build_minimal_schema(
                entity=intent.entity,
                fields=intent.fields,
                full_schema_context=full_schema
            )

            # Step 6: Generate query
            query = await self.generate_query(
                question=question,
                schema_context=minimal_schema,
                intent=intent,
                protocol=protocol
            )
            result["query"] = query

            # Step 7: Validate generated query
            if self.validator:
                is_valid, error = self.validator.validate(query)
                if not is_valid:
                    result["is_valid"] = False
                    result["error"] = f"Generated query invalid: {error}"
                    result["elapsed_time"] = time.time() - start_time
                    return result

            # Step 8: Execute query if endpoint provided
            if endpoint:
                import httpx
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(
                        endpoint,
                        json={"query": query},
                        headers={"Content-Type": "application/json"}
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
