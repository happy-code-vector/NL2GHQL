"""
Bittensor Miner for Hermes Subnet (Subnet 82)

This miner integrates with the Hermes subnet to:
1. Register on the subnet
2. Respond to synthetic challenges from validators
3. Handle organic queries from users
"""

import os
import sys
import asyncio
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import bittensor as bt
from config.settings import config


@dataclass
class MinerConfig:
    """Miner configuration"""
    # Network settings
    network: str = "finney"
    subnet_uid: int = 82

    # Wallet settings
    wallet_name: str = "miner"
    hotkey_name: str = "default"

    # Performance settings
    max_concurrent_requests: int = 10
    request_timeout: float = 30.0


class HermesMiner:
    """
    Bittensor miner for Hermes Subnet (Subnet 82)

    This miner:
    1. Answers synthetic challenges from validators
    2. Handles organic user queries
    3. Reports capacity and health
    """

    def __init__(
        self,
        config: MinerConfig,
        agent_service_url: str = "http://localhost:8000"
    ):
        self.config = config
        self.agent_service_url = agent_service_url

        # Initialize Bittensor components
        self.subtensor = None
        self.wallet = None
        self.metagraph = None
        self.axon = None

        # Project configurations (from Hermes subnet)
        self.projects: Dict[str, Dict] = {}

        # Performance tracking
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_response_time": 0.0
        }

    async def setup(self):
        """Initialize the miner"""
        bt.logging.info("=" * 60)
        bt.logging.info("Hermes Miner Setup")
        bt.logging.info("=" * 60)

        # 1. Setup wallet
        bt.logging.info("[1/4] Setting up wallet...")
        self.wallet = bt.wallet(
            name=self.config.wallet_name,
            hotkey=self.config.hotkey_name
        )
        bt.logging.info(f"Wallet: {self.wallet.name}/{self.wallet.hotkey_str}")

        # 2. Connect to subtensor
        bt.logging.info("[2/4] Connecting to subtensor...")
        self.subtensor = bt.subtensor(network=self.config.network)
        bt.logging.info(f"Connected to: {self.config.network}")

        # 3. Get metagraph
        bt.logging.info("[3/4] Getting metagraph...")
        self.metagraph = self.subtensor.metagraph(self.config.subnet_uid)
        bt.logging.info(f"Subnet: {self.config.subnet_uid}")
        bt.logging.info(f"Validators: {len(self.metagraph.validator_permit)}")

        # 4. Setup axon
        bt.logging.info("[4/4] Setting up axon...")
        self.axon = bt.axon(wallet=self.wallet)

        # Attach synapse handlers
        self._attach_handlers()

        bt.logging.info("Setup complete!")

    def _attach_handlers(self):
        """Attach synapse handlers for different request types"""

        # Import Hermes synapse types
        try:
            from hermes.protocol import (
                SyntheticNonStreamSynapse,
                OrganicNonStreamSynapse,
                OrganicStreamSynapse,
                CapacitySynapse
            )

            # Register handlers
            self.axon.attach(
                forward_fn=self.handle_synthetic_challenge,
                synapse=SyntheticNonStreamSynapse
            )
            self.axon.attach(
                forward_fn=self.handle_organic_query,
                synapse=OrganicNonStreamSynapse
            )
            self.axon.attach(
                forward_fn=self.handle_organic_stream,
                synapse=OrganicStreamSynapse
            )
            self.axon.attach(
                forward_fn=self.handle_capacity_check,
                synapse=CapacitySynapse
            )

        except ImportError:
            bt.logging.warning(
                "Hermes protocol not found. Using mock handlers."
            )
            # Use mock handlers for development
            self.axon.attach(forward_fn=self._mock_handler)

    async def handle_synthetic_challenge(self, synapse) -> Any:
        """
        Handle synthetic challenge from validator.

        The synapse contains:
        - question: Natural language question
        - cid_hash: Project identifier
        - block_height: Block height for time-travel query
        """
        start_time = time.time()

        try:
            bt.logging.debug(f"Received synthetic challenge: {synapse.question[:50]}...")

            # Get project configuration
            project_config = self.projects.get(synapse.cid_hash, {})
            endpoint = project_config.get("endpoint")
            protocol = project_config.get("protocol", "subql")

            if not endpoint:
                bt.logging.warning(f"No endpoint for project: {synapse.cid_hash}")
                synapse.answer = "Unable to answer: Project not configured"
                return synapse

            # Call our agent service
            import httpx

            async with httpx.AsyncClient(timeout=self.config.request_timeout) as client:
                response = await client.post(
                    f"{self.agent_service_url}/answer",
                    json={
                        "question": synapse.question,
                        "endpoint": endpoint,
                        "project": synapse.cid_hash,
                        "block_height": synapse.block_height,
                        "protocol": protocol
                    }
                )
                result = response.json()

            # Update synapse with answer
            synapse.answer = result.get("answer", "")
            synapse.query = result.get("query", "")
            synapse.elapsed_time = time.time() - start_time

            # Update stats
            self.stats["total_requests"] += 1
            self.stats["successful_requests"] += 1

            bt.logging.debug(f"Challenge answered in {synapse.elapsed_time:.2f}s")

            return synapse

        except Exception as e:
            bt.logging.error(f"Error handling challenge: {e}")
            self.stats["total_requests"] += 1
            self.stats["failed_requests"] += 1

            synapse.answer = f"Error: {str(e)}"
            synapse.elapsed_time = time.time() - start_time
            return synapse

    async def handle_organic_query(self, synapse) -> Any:
        """
        Handle organic (real user) query.

        The synapse contains OpenAI-compatible chat messages.
        """
        try:
            bt.logging.info("Received organic query")

            # Extract the question from messages
            messages = getattr(synapse, "messages", [])
            if not messages:
                synapse.completion = "No question provided"
                return synapse

            # Get the last user message
            question = None
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    question = msg.get("content", "")
                    break

            if not question:
                synapse.completion = "No question found in messages"
                return synapse

            # Get project from metadata
            project_id = getattr(synapse, "project_id", None)
            project_config = self.projects.get(project_id, {})

            # Call agent service
            import httpx

            async with httpx.AsyncClient(timeout=self.config.request_timeout) as client:
                response = await client.post(
                    f"{self.agent_service_url}/answer",
                    json={
                        "question": question,
                        "endpoint": project_config.get("endpoint"),
                        "project": project_id,
                        "protocol": project_config.get("protocol", "subql")
                    }
                )
                result = response.json()

            synapse.completion = result.get("answer", "")
            return synapse

        except Exception as e:
            bt.logging.error(f"Error handling organic query: {e}")
            synapse.completion = f"Error: {str(e)}"
            return synapse

    async def handle_organic_stream(self, synapse):
        """Handle streaming organic query"""
        # For now, delegate to non-streaming handler
        synapse = await self.handle_organic_query(synapse)

        # Yield as a single chunk
        yield synapse

    async def handle_capacity_check(self, synapse) -> Any:
        """Handle capacity/health check from validator"""
        synapse.capacity = self._get_capacity_info()
        synapse.projects = list(self.projects.keys())
        synapse.healthy = True
        return synapse

    def _get_capacity_info(self) -> Dict[str, Any]:
        """Get capacity information"""
        return {
            "max_concurrent": self.config.max_concurrent_requests,
            "current_load": 0,  # TODO: Track actual load
            "avg_response_time": self.stats["avg_response_time"],
            "success_rate": (
                self.stats["successful_requests"] / max(1, self.stats["total_requests"])
            )
        }

    async def _mock_handler(self, synapse):
        """Mock handler for development"""
        synapse.answer = "Mock response - protocol not loaded"
        return synapse

    def load_projects(self, projects_config: Dict[str, Dict]):
        """Load project configurations"""
        self.projects = projects_config
        bt.logging.info(f"Loaded {len(self.projects)} projects")

    async def run(self):
        """Run the miner"""
        await self.setup()

        # Start the axon
        bt.logging.info("Starting axon...")
        self.axon.serve(netuid=self.config.subnet_uid, subtensor=self.subtensor)
        self.axon.start()

        bt.logging.info("=" * 60)
        bt.logging.info("Hermes Miner Running!")
        bt.logging.info(f"UID: {self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)}")
        bt.logging.info("=" * 60)

        # Main loop
        try:
            while True:
                # Update metagraph periodically
                self.metagraph = self.subtensor.metagraph(self.config.subnet_uid)

                # Log stats every 100 blocks
                bt.logging.info(
                    f"Stats: {self.stats['successful_requests']}/{self.stats['total_requests']} "
                    f"successful ({self.stats['avg_response_time']:.2f}s avg)"
                )

                # Wait for next block
                await asyncio.sleep(12)  # ~12 seconds per block

        except KeyboardInterrupt:
            bt.logging.info("Shutting down...")
            self.axon.stop()
            self.subtensor.close()


def load_projects_from_file(file_path: str) -> Dict[str, Dict]:
    """Load project configurations from JSON file"""
    import json

    with open(file_path, 'r') as f:
        return json.load(f)


async def main():
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Hermes Bittensor Miner")
    parser.add_argument("--network", default="finney", help="Bittensor network")
    parser.add_argument("--wallet-name", default="miner", help="Wallet name")
    parser.add_argument("--hotkey", default="default", help="Hotkey name")
    parser.add_argument("--agent-url", default="http://localhost:8000", help="Agent service URL")
    parser.add_argument("--projects", default=None, help="Path to projects config JSON")

    args = parser.parse_args()

    # Create config
    config = MinerConfig(
        network=args.network,
        wallet_name=args.wallet_name,
        hotkey_name=args.hotkey
    )

    # Create miner
    miner = HermesMiner(
        config=config,
        agent_service_url=args.agent_url
    )

    # Load projects if provided
    if args.projects:
        projects = load_projects_from_file(args.projects)
        miner.load_projects(projects)

    # Run miner
    await miner.run()


if __name__ == "__main__":
    asyncio.run(main())
