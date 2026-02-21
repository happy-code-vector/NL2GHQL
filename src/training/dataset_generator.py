"""
Training Dataset Generator for Hermes Miner

Generates synthetic training data for fine-tuning the NL2GraphQL model.
"""

import json
import re
import random
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from pathlib import Path


@dataclass
class TrainingExample:
    """A single training example"""
    instruction: str  # The natural language question
    input: str  # Schema context
    output: str  # The GraphQL query


class QuestionTemplate:
    """Template for generating questions"""

    # Count question templates
    COUNT_TEMPLATES = [
        "How many {entity} are currently {condition}?",
        "What is the total number of {entity}?",
        "Count all {entity} in the system.",
        "What is the count of {entity} with {attribute}?",
    ]

    # Superlative question templates
    SUPERLATIVE_TEMPLATES = [
        "Which {entity} has the highest {attribute}?",
        "What is the maximum {attribute} among all {entity}?",
        "Which {entity} has the lowest {attribute}?",
        "What is the minimum {attribute} for {entity}?",
    ]

    # Specific entity question templates
    ENTITY_TEMPLATES = [
        "What is the {attribute} of {entity} {id}?",
        "Show me the {attribute} for {entity} {id}.",
        "Get the {attribute} value for {entity} with ID {id}.",
    ]

    # Sum/aggregate question templates
    AGGREGATE_TEMPLATES = [
        "What is the total {attribute} across all {entity}?",
        "What is the sum of {attribute} for all {entity}?",
        "What is the average {attribute} among {entity}?",
    ]


class QueryTemplate:
    """Template for generating GraphQL queries"""

    # Basic query template
    BASIC_TEMPLATE = """{operation} {{
  {field}({params}) {{
    {selections}
  }}
}}"""

    # Connection query template (for paginated results)
    CONNECTION_TEMPLATE = """{operation} {{
  {field}(first: {limit}{params}) {{
    nodes {{
      {selections}
    }}
    totalCount
  }}
}}"""

    # Single entity query template
    ENTITY_TEMPLATE = """{operation} {{
  {field}(id: "{id}"{params}) {{
    {selections}
  }}
}}"""


class DatasetGenerator:
    """Generates synthetic training data from GraphQL schemas"""

    def __init__(self, schema_path: Optional[str] = None):
        self.schema_path = schema_path
        self.types: Dict[str, Dict] = {}
        self.queries: Dict[str, Dict] = {}

    def load_schema(self, schema_path: str) -> None:
        """Load and parse a GraphQL schema file"""
        with open(schema_path, 'r', encoding='utf-8') as f:
            content = f.read()

        self._parse_schema(content)

    def _parse_schema(self, content: str) -> None:
        """Parse schema content to extract types and queries"""
        # Parse types
        type_pattern = r'type\s+(\w+)\s*(?:implements\s+[\w&\s]+)?\s*\{([^}]+)\}'
        for match in re.finditer(type_pattern, content, re.DOTALL):
            type_name = match.group(1)
            fields_body = match.group(2)

            fields = self._parse_fields(fields_body)

            if type_name in ['Query', 'Mutation', 'Subscription']:
                self.queries[type_name] = {
                    'name': type_name,
                    'fields': fields
                }
            else:
                self.types[type_name] = {
                    'name': type_name,
                    'fields': fields
                }

    def _parse_fields(self, body: str) -> List[Dict]:
        """Parse field definitions from a type body"""
        fields = []
        field_pattern = r'(\w+)\s*(?:\([^)]*\))?\s*:\s*([\[\]\w!]+)'

        for match in re.finditer(field_pattern, body):
            field_name = match.group(1)
            field_type = match.group(2)

            # Check if it's a connection type
            is_connection = field_name.endswith('s') or '[' in field_type

            fields.append({
                'name': field_name,
                'type': field_type,
                'is_connection': is_connection
            })

        return fields

    def generate_examples(
        self,
        num_examples: int = 1000,
        include_block_height: bool = True
    ) -> List[TrainingExample]:
        """Generate training examples"""
        examples = []

        # Generate count queries
        examples.extend(self._generate_count_examples(num_examples // 4))

        # Generate superlative queries
        examples.extend(self._generate_superlative_examples(num_examples // 4))

        # Generate entity-specific queries
        examples.extend(self._generate_entity_examples(num_examples // 4))

        # Generate aggregate queries
        examples.extend(self._generate_aggregate_examples(num_examples // 4))

        # Shuffle
        random.shuffle(examples)

        return examples[:num_examples]

    def _generate_count_examples(self, count: int) -> List[TrainingExample]:
        """Generate count-type training examples"""
        examples = []

        for query_type, query_info in self.queries.items():
            for field in query_info.get('fields', []):
                if not field['is_connection']:
                    continue

                field_name = field['name']
                entity_name = field_name.rstrip('s').replace('_', ' ')

                # Generate questions
                questions = [
                    f"How many {entity_name} are there?",
                    f"What is the total count of {entity_name}?",
                    f"Count all {entity_name} in the network.",
                ]

                # Generate queries
                base_query = f"""query {{
  {field_name}(first: 10) {{
    totalCount
    nodes {{
      id
    }}
  }}
}}"""

                for question in questions:
                    examples.append(TrainingExample(
                        instruction=question,
                        input=self._get_schema_context(field_name),
                        output=base_query
                    ))

                    if len(examples) >= count:
                        return examples[:count]

        return examples

    def _generate_superlative_examples(self, count: int) -> List[TrainingExample]:
        """Generate superlative-type training examples"""
        examples = []

        for type_name, type_info in self.types.items():
            numeric_fields = [
                f for f in type_info.get('fields', [])
                if any(t in f['type'].lower() for t in ['int', 'float', 'bigint', 'decimal'])
            ]

            for field in numeric_fields:
                field_name = field['name']
                type_lower = type_name.lower()

                # Find the query field for this type
                query_field = f"{type_lower}s"

                # Highest question
                questions = [
                    f"Which {type_lower} has the highest {field_name}?",
                    f"What is the maximum {field_name} among all {type_lower}?",
                ]

                query = f"""query {{
  {query_field}(first: 1, orderBy: {field_name}_DESC) {{
    nodes {{
      id
      {field_name}
    }}
  }}
}}"""

                for question in questions:
                    examples.append(TrainingExample(
                        instruction=question,
                        input=self._get_schema_context(type_name),
                        output=query
                    ))

                    if len(examples) >= count:
                        return examples[:count]

        return examples

    def _generate_entity_examples(self, count: int) -> List[TrainingExample]:
        """Generate entity-specific training examples"""
        examples = []

        for type_name, type_info in self.types.items():
            type_lower = type_name.lower()

            for field in type_info.get('fields', []):
                if field['name'] == 'id':
                    continue

                field_name = field['name']

                # Generate with example ID
                example_id = f"0x{''.join(random.choices('0123456789abcdef', k=40))}"

                questions = [
                    f"What is the {field_name} of {type_lower} {example_id}?",
                    f"Get the {field_name} for {type_lower} with ID {example_id}.",
                ]

                query = f"""query {{
  {type_lower}(id: "{example_id}") {{
    id
    {field_name}
  }}
}}"""

                for question in questions:
                    examples.append(TrainingExample(
                        instruction=question,
                        input=self._get_schema_context(type_name),
                        output=query
                    ))

                    if len(examples) >= count:
                        return examples[:count]

        return examples

    def _generate_aggregate_examples(self, count: int) -> List[TrainingExample]:
        """Generate aggregate-type training examples"""
        examples = []

        for type_name, type_info in self.types.items():
            numeric_fields = [
                f for f in type_info.get('fields', [])
                if any(t in f['type'].lower() for t in ['int', 'float', 'bigint', 'decimal'])
            ]

            for field in numeric_fields:
                field_name = field['name']
                type_lower = type_name.lower()
                query_field = f"{type_lower}s"

                questions = [
                    f"What is the total {field_name} across all {type_lower}?",
                    f"What is the sum of {field_name} for all {type_lower}?",
                ]

                query = f"""query {{
  {query_field}(first: 1000) {{
    nodes {{
      {field_name}
    }}
  }}
}}"""

                for question in questions:
                    examples.append(TrainingExample(
                        instruction=question,
                        input=self._get_schema_context(type_name),
                        output=query
                    ))

                    if len(examples) >= count:
                        return examples[:count]

        return examples

    def _get_schema_context(self, type_name: str) -> str:
        """Get schema context for a type"""
        if type_name in self.types:
            type_info = self.types[type_name]
            fields = "\n  ".join([
                f"{f['name']}: {f['type']}"
                for f in type_info.get('fields', [])[:5]
            ])
            return f"type {type_name} {{\n  {fields}\n}}"
        return ""

    def save_to_jsonl(
        self,
        examples: List[TrainingExample],
        output_path: str
    ) -> None:
        """Save training examples to JSONL format"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in examples:
                data = {
                    "instruction": example.instruction,
                    "input": example.input,
                    "output": example.output
                }
                f.write(json.dumps(data) + '\n')

    def load_from_jsonl(self, input_path: str) -> List[TrainingExample]:
        """Load training examples from JSONL format"""
        examples = []

        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                examples.append(TrainingExample(
                    instruction=data['instruction'],
                    input=data['input'],
                    output=data['output']
                ))

        return examples


class HermesDatasetGenerator(DatasetGenerator):
    """Dataset generator specifically for Hermes subnet schemas"""

    # Hermes-specific question templates
    HERMES_TEMPLATES = {
        "indexer": [
            "What is the total stake of indexer {id}?",
            "How many indexers are currently active?",
            "Which indexer has the highest self stake?",
            "What is the commission rate of indexer {id}?",
        ],
        "era": [
            "What is the start time of era {id}?",
            "How long is the current era period?",
            "What is the total reward for era {id}?",
        ],
        "delegation": [
            "How many delegations does indexer {id} have?",
            "What is the total delegated stake?",
            "Which delegator has the most stake?",
        ]
    }

    def generate_hermes_examples(
        self,
        num_examples: int = 500,
        block_heights: Optional[List[int]] = None
    ) -> List[TrainingExample]:
        """Generate Hermes-specific training examples"""
        examples = []

        # Add block height variations if specified
        if block_heights:
            for block_height in block_heights:
                block_examples = self._generate_block_height_examples(
                    num_examples // len(block_heights),
                    block_height
                )
                examples.extend(block_examples)

        # Add standard examples
        examples.extend(self.generate_examples(num_examples - len(examples)))

        random.shuffle(examples)
        return examples[:num_examples]

    def _generate_block_height_examples(
        self,
        count: int,
        block_height: int
    ) -> List[TrainingExample]:
        """Generate examples with block height parameter"""
        examples = []

        # SubQL format
        subql_query = f"""query {{
  indexers(first: 10, blockHeight: "{block_height}") {{
    nodes {{
      id
      totalStake
      selfStake
    }}
  }}
}}"""

        examples.append(TrainingExample(
            instruction=f"What was the total stake of indexers at block {block_height}?",
            input="type Indexer {\n  id: ID!\n  totalStake: BigInt!\n  selfStake: BigInt!\n}",
            output=subql_query
        ))

        # The Graph format
        the_graph_query = f"""query {{
  swaps(first: 10, block: {{number: {block_height}}}) {{
    id
    amount0In
    amount1In
  }}
}}"""

        examples.append(TrainingExample(
            instruction=f"What were the swap volumes at block {block_height}?",
            input="type Swap {\n  id: ID!\n  amount0In: BigDecimal!\n  amount1In: BigDecimal!\n}",
            output=the_graph_query
        ))

        return examples


def main():
    """CLI entry point for dataset generation"""
    import argparse

    parser = argparse.ArgumentParser(description="Generate training dataset")
    parser.add_argument("--schema", required=True, help="Path to GraphQL schema file")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument("--count", type=int, default=1000, help="Number of examples to generate")
    parser.add_argument("--block-heights", nargs='+', type=int, help="Block heights to include")

    args = parser.parse_args()

    generator = HermesDatasetGenerator()
    generator.load_schema(args.schema)

    if args.block_heights:
        examples = generator.generate_hermes_examples(
            args.count,
            block_heights=args.block_heights
        )
    else:
        examples = generator.generate_examples(args.count)

    generator.save_to_jsonl(examples, args.output)

    print(f"Generated {len(examples)} training examples to {args.output}")


if __name__ == "__main__":
    main()
