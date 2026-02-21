"""
Extract and convert GraphQL introspection schema to usable format.

This script reads the large introspection JSON and converts it to:
1. A simplified JSON with type definitions
2. GraphQL SDL format
"""

import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path


def load_introspection(file_path: str) -> Dict:
    """Load GraphQL introspection result from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle wrapped introspection result
    if 'data' in data and '__schema' in data['data']:
        return data['data']['__schema']
    elif '__schema' in data:
        return data['__schema']
    return data


def type_ref_to_string(type_ref: Dict) -> str:
    """Convert a type reference to string format"""
    kind = type_ref.get('kind')
    name = type_ref.get('name')
    of_type = type_ref.get('ofType')

    if kind == 'NON_NULL':
        inner = type_ref_to_string(of_type)
        return f"{inner}!"
    elif kind == 'LIST':
        inner = type_ref_to_string(of_type)
        return f"[{inner}]"
    else:
        return name or 'Unknown'


def extract_types(schema: Dict) -> Dict[str, Dict]:
    """Extract type definitions from schema"""
    types = {}

    for type_def in schema.get('types', []):
        name = type_def.get('name')

        # Skip introspection types
        if name.startswith('__'):
            continue

        kind = type_def.get('kind')

        type_info = {
            'name': name,
            'kind': kind,
            'description': type_def.get('description', ''),
        }

        if kind == 'OBJECT' or kind == 'INTERFACE':
            type_info['fields'] = extract_fields(type_def.get('fields', []))

        elif kind == 'INPUT_OBJECT':
            type_info['inputFields'] = extract_input_fields(type_def.get('inputFields', []))

        elif kind == 'ENUM':
            type_info['enumValues'] = [
                {'name': v.get('name'), 'description': v.get('description', '')}
                for v in type_def.get('enumValues', [])
            ]

        elif kind == 'UNION':
            type_info['possibleTypes'] = [
                t.get('name') for t in type_def.get('possibleTypes', [])
            ]

        elif kind == 'SCALAR':
            type_info['description'] = type_def.get('description', f"Scalar type: {name}")

        types[name] = type_info

    return types


def extract_fields(fields: List[Dict]) -> List[Dict]:
    """Extract field definitions"""
    result = []

    for field in fields:
        field_info = {
            'name': field.get('name'),
            'type': type_ref_to_string(field.get('type', {})),
            'description': field.get('description', ''),
            'args': []
        }

        for arg in field.get('args', []):
            arg_info = {
                'name': arg.get('name'),
                'type': type_ref_to_string(arg.get('type', {})),
                'description': arg.get('description', ''),
                'defaultValue': arg.get('defaultValue')
            }
            field_info['args'].append(arg_info)

        result.append(field_info)

    return result


def extract_input_fields(fields: List[Dict]) -> List[Dict]:
    """Extract input field definitions"""
    result = []

    for field in fields:
        field_info = {
            'name': field.get('name'),
            'type': type_ref_to_string(field.get('type', {})),
            'description': field.get('description', ''),
            'defaultValue': field.get('defaultValue')
        }
        result.append(field_info)

    return result


def generate_sdl(types: Dict[str, Dict]) -> str:
    """Generate GraphQL SDL from type definitions"""
    lines = []

    # Order: Scalars, Enums, Inputs, Objects, Interfaces, Unions
    def type_order(item):
        name, t = item
        kind = t.get('kind')
        order = {'SCALAR': 0, 'ENUM': 1, 'INPUT_OBJECT': 2, 'OBJECT': 3, 'INTERFACE': 4, 'UNION': 5}
        return (order.get(kind, 6), name)

    for name, type_info in sorted(types.items(), key=type_order):
        kind = type_info.get('kind')
        description = type_info.get('description', '')

        if description:
            lines.append(f'"""{description}"""')

        if kind == 'SCALAR':
            lines.append(f"scalar {name}")
            lines.append("")

        elif kind == 'ENUM':
            lines.append(f"enum {name} {{")
            for value in type_info.get('enumValues', []):
                desc = value.get('description')
                if desc:
                    lines.append(f"  \"\"\"{desc}\"\"\"")
                lines.append(f"  {value['name']}")
            lines.append("}")
            lines.append("")

        elif kind == 'INPUT_OBJECT':
            lines.append(f"input {name} {{")
            for field in type_info.get('inputFields', []):
                field_desc = field.get('description', '')
                if field_desc:
                    lines.append(f"  # {field_desc}")
                default = field.get('defaultValue')
                default_str = f" = {default}" if default else ""
                lines.append(f"  {field['name']}: {field['type']}{default_str}")
            lines.append("}")
            lines.append("")

        elif kind in ('OBJECT', 'INTERFACE'):
            keyword = 'type' if kind == 'OBJECT' else 'interface'
            lines.append(f"{keyword} {name} {{")

            for field in type_info.get('fields', []):
                field_desc = field.get('description', '')
                if field_desc:
                    # Truncate long descriptions
                    if len(field_desc) > 100:
                        field_desc = field_desc[:97] + '...'
                    lines.append(f"  # {field_desc}")

                args = field.get('args', [])
                if args:
                    args_str = ', '.join([
                        f"{a['name']}: {a['type']}" +
                        (f" = {a['defaultValue']}" if a.get('defaultValue') else "")
                        for a in args
                    ])
                    lines.append(f"  {field['name']}({args_str}): {field['type']}")
                else:
                    lines.append(f"  {field['name']}: {field['type']}")

            lines.append("}")
            lines.append("")

        elif kind == 'UNION':
            possible = ' | '.join(type_info.get('possibleTypes', []))
            lines.append(f"union {name} = {possible}")
            lines.append("")

    return '\n'.join(lines)


def get_query_types(types: Dict[str, Dict]) -> List[str]:
    """Get list of queryable entity types (for SubQL/PostGraphile)"""
    query_type = types.get('Query', {})
    fields = query_type.get('fields', [])

    entities = []
    for field in fields:
        name = field['name']
        # Skip introspection fields
        if name in ('query', 'nodeId', 'node'):
            continue
        # Look for connection types (plural)
        field_type = field['type']
        if 'Connection' in field_type or field_type.endswith('Connection'):
            # Extract entity name from connection
            entity_name = field_type.replace('Connection', '')
            entities.append({
                'query_field': name,
                'entity_type': entity_name,
                'description': field.get('description', '')
            })

    return entities


def main():
    """Main function to extract schema"""
    import argparse

    parser = argparse.ArgumentParser(description="Extract GraphQL schema")
    parser.add_argument("--input", default="schema.json", help="Input introspection JSON")
    parser.add_argument("--output", default="data/schemas/schema_subnet.json", help="Output JSON file")
    parser.add_argument("--sdl", default="data/schemas/schema_subnet.graphql", help="Output SDL file")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Resolve paths
    base_dir = Path(__file__).parent.parent
    input_path = base_dir / args.input
    output_path = base_dir / args.output
    sdl_path = base_dir / args.sdl

    print(f"Loading introspection from: {input_path}")
    schema = load_introspection(input_path)

    print("Extracting type definitions...")
    types = extract_types(schema)

    # Get queryable entities
    entities = get_query_types(types)

    # Build simplified schema
    simplified = {
        'queryType': schema.get('queryType', {}).get('name'),
        'mutationType': schema.get('mutationType', {}).get('name') if schema.get('mutationType') else None,
        'types': types,
        'entities': entities,
        'statistics': {
            'total_types': len(types),
            'object_types': len([t for t in types.values() if t['kind'] == 'OBJECT']),
            'input_types': len([t for t in types.values() if t['kind'] == 'INPUT_OBJECT']),
            'enum_types': len([t for t in types.values() if t['kind'] == 'ENUM']),
            'scalar_types': len([t for t in types.values() if t['kind'] == 'SCALAR']),
            'queryable_entities': len(entities)
        }
    }

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save simplified JSON
    print(f"Saving simplified schema to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(simplified, f, indent=2)

    # Generate and save SDL
    print(f"Generating SDL...")
    sdl = generate_sdl(types)

    print(f"Saving SDL to: {sdl_path}")
    with open(sdl_path, 'w', encoding='utf-8') as f:
        f.write(sdl)

    # Print statistics
    stats = simplified['statistics']
    print("\n" + "=" * 60)
    print("Schema Statistics")
    print("=" * 60)
    print(f"Total types: {stats['total_types']}")
    print(f"Object types: {stats['object_types']}")
    print(f"Input types: {stats['input_types']}")
    print(f"Enum types: {stats['enum_types']}")
    print(f"Scalar types: {stats['scalar_types']}")
    print(f"Queryable entities: {stats['queryable_entities']}")

    if args.verbose:
        print("\nQueryable Entities:")
        for entity in entities[:20]:  # Show first 20
            print(f"  - {entity['query_field']} -> {entity['entity_type']}")
        if len(entities) > 20:
            print(f"  ... and {len(entities) - 20} more")

    print("\nDone!")
    return simplified


if __name__ == "__main__":
    main()
