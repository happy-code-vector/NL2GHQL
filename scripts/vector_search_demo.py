"""
Real Example: Vector Search with Hermes Schema

This demonstrates how vector search works with actual Hermes subnet data.
"""

# What gets indexed for an Indexer type:
INDEXER_DOCUMENT = """
Description: A network operator that indexes blockchain data and serves queries.
Has stake, commission rate, and delegation information.

Type: type Indexer

Fields:
  id: ID! - Indexer wallet address
  totalStake: BigInt! - Total tokens staked (self + delegated)
  selfStake: BigInt! - Indexer's own staked tokens
  commission: JSON! - Commission rate for rewards
  active: Boolean! - Whether indexer is active
  controller: String - Controller address
  delegations: DelegationsConnection - Delegations to this indexer
  rewards: RewardsConnection - Rewards earned
  deployments: DeploymentsConnection - Project deployments

Definition:
type Indexer {
  id: ID!
  totalStake: BigInt!
  selfStake: BigInt!
  commission: JSON!
  active: Boolean!
  controller: String
  delegations(first: Int, filter: DelegationFilter): DelegationsConnection!
  rewards(first: Int): RewardsConnection!
  deployments(first: Int): DeploymentsConnection!
}
"""

# What gets indexed for a Delegation type:
DELEGATION_DOCUMENT = """
Description: Represents a delegation from a delegator to an indexer.
Tracks the amount delegated and era information.

Type: type Delegation

Fields:
  id: ID! - Composite ID (delegator + indexer)
  delegatorId: String! - Delegator wallet address
  indexerId: String! - Indexer wallet address
  amount: BigInt! - Amount of tokens delegated
  exitEra: Int - Era when delegation exits
  createdBlock: Int - Block when created

Definition:
type Delegation {
  id: ID!
  delegatorId: String!
  indexerId: String!
  amount: BigInt!
  exitEra: Int
  createdBlock: Int
}
"""

# What gets indexed for Era type:
ERA_DOCUMENT = """
Description: A time period for reward distribution in the network.
Tracks era start, end, and total rewards.

Type: type Era

Fields:
  id: ID! - Era identifier
  startTime: DateTime! - When era started
  endTime: DateTime - When era ended
  totalRewards: BigInt! - Total rewards for this era

Definition:
type Era {
  id: ID!
  startTime: DateTime!
  endTime: DateTime
  totalRewards: BigInt!
}
"""


def simulate_vector_search(query: str):
    """
    Simulate how vector search compares the query against indexed documents.
    """
    from sentence_transformers import SentenceTransformer
    import numpy as np

    # Load embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Documents to search
    documents = {
        "Indexer": INDEXER_DOCUMENT,
        "Delegation": DELEGATION_DOCUMENT,
        "Era": ERA_DOCUMENT
    }

    # Encode query
    query_embedding = model.encode(query)

    print("=" * 70)
    print(f"QUERY: '{query}'")
    print("=" * 70)

    print("\n1. QUERY EMBEDDING (first 10 dims):")
    print(f"   {query_embedding[:10].round(3)}")

    print("\n2. COMPARING AGAINST INDEXED DOCUMENTS:")
    print("-" * 70)

    results = []
    for name, doc in documents.items():
        # Encode document
        doc_embedding = model.encode(doc)

        # Cosine similarity
        similarity = np.dot(query_embedding, doc_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
        )

        results.append((name, similarity, doc))

        print(f"\n   {name}:")
        print(f"   Similarity Score: {similarity:.4f}")
        print(f"   Matched Text Preview:")
        # Show relevant parts of the document
        lines = doc.strip().split('\n')
        for line in lines[:5]:
            print(f"      {line[:60]}...")

    # Sort by similarity
    results.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 70)
    print("3. RANKED RESULTS (Highest Similarity First):")
    print("=" * 70)

    for i, (name, score, doc) in enumerate(results, 1):
        print(f"\n   #{i} {name} (score: {score:.4f})")

    # Return top result
    top = results[0]
    print("\n" + "=" * 70)
    print(f"BEST MATCH: {top[0]}")
    print("=" * 70)
    print(top[2][:500] + "...")

    return results


def main():
    # Real test queries from the dataset
    queries = [
        "What is the total stake of indexer",
        "How many delegations",
        "Which indexer has the highest rewards",
        "What is the commission rate",
        "Total rewards earned in era 42200",
    ]

    print("\n" + "=" * 70)
    print("VECTOR SEARCH DEMO WITH REAL HERMES SCHEMA")
    print("=" * 70)

    for query in queries[:3]:  # Test first 3
        simulate_vector_search(query)
        print("\n\n")


if __name__ == "__main__":
    main()
