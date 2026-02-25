"""
Multi-Provider LLM Client

Supports:
- Google Gemini (recommended for NL2GraphQL)
- OpenAI (GPT-4o, GPT-4-turbo, etc.)
- vLLM (Local models via OpenAI-compatible API)
- Mock (for testing without API calls)
"""

import os
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from enum import Enum


class LLMProvider(Enum):
    GEMINI = "gemini"
    OPENAI = "openai"
    VLLM = "vllm"
    MOCK = "mock"


@dataclass
class LLMConfig:
    """LLM configuration"""
    provider: LLMProvider = LLMProvider.GEMINI
    model: str = "gemini-2.5-flash"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout: float = 60.0

    # vLLM specific
    gpu_memory_utilization: float = 0.85

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Load configuration from environment variables"""
        provider_str = os.getenv("LLM_PROVIDER", "gemini").lower()

        provider_map = {
            "gemini": LLMProvider.GEMINI,
            "google": LLMProvider.GEMINI,
            "openai": LLMProvider.OPENAI,
            "gpt": LLMProvider.OPENAI,
            "vllm": LLMProvider.VLLM,
            "mock": LLMProvider.MOCK,
        }

        provider = provider_map.get(provider_str, LLMProvider.GEMINI)

        # Get API key based on provider
        if provider == LLMProvider.GEMINI:
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            base_url = None
        elif provider == LLMProvider.OPENAI:
            api_key = os.getenv("OPENAI_API_KEY")
            model = os.getenv("OPENAI_MODEL", "gpt-4o")
            base_url = os.getenv("OPENAI_BASE_URL")
        elif provider == LLMProvider.VLLM:
            api_key = os.getenv("VLLM_API_KEY", "dummy")
            model = os.getenv("VLLM_MODEL", "models/hermes_miner")
            base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        else:
            api_key = None
            model = "mock"
            base_url = None

        return cls(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048")),
        )


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients"""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text from prompt"""
        pass

    @abstractmethod
    async def generate_async(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Async generate text from prompt"""
        pass


class GeminiClient(BaseLLMClient):
    """Google Gemini API client"""

    def __init__(self, config: LLMConfig):
        from langchain_google_genai import ChatGoogleGenerativeAI

        self.config = config
        self.client = ChatGoogleGenerativeAI(
            model=config.model,
            google_api_key=config.api_key,
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
        )

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using Gemini API"""
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        response = self.client.invoke(messages)
        return response.content

    async def generate_async(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Async generate using Gemini"""
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        response = await self.client.ainvoke(messages)
        return response.content


class OpenAIClient(BaseLLMClient):
    """OpenAI API client"""

    def __init__(self, config: LLMConfig):
        from openai import OpenAI

        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using OpenAI API"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return response.choices[0].message.content

    async def generate_async(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Async generate using thread pool"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, prompt, system_prompt)


class VLLMClient(BaseLLMClient):
    """vLLM client (OpenAI-compatible server) for local models"""

    def __init__(self, config: LLMConfig):
        from openai import OpenAI

        self.config = config
        self.client = OpenAI(
            api_key=config.api_key or "dummy",
            base_url=config.base_url or "http://localhost:8000/v1"
        )

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using vLLM server"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            return response.choices[0].message.content
        except Exception:
            # Fall back to completions API
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            response = self.client.completions.create(
                model=self.config.model,
                prompt=full_prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            return response.choices[0].text

    async def generate_async(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Async generate using thread pool"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, prompt, system_prompt)


class MockLLMClient(BaseLLMClient):
    """Mock client for testing without API calls"""

    def __init__(self, config: LLMConfig = None):
        self.config = config or LLMConfig(provider=LLMProvider.MOCK)

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Return mock response based on prompt content"""
        # Simulate query generation
        if "GraphQL" in prompt or "query" in prompt.lower():
            return self._mock_query_response(prompt)

        # Simulate answer generation
        if "Answer" in prompt or "answer" in prompt.lower():
            return "Mock answer: The result is 42."

        return "Mock response"

    async def generate_async(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Async mock generate"""
        await asyncio.sleep(0.1)  # Simulate latency
        return self.generate(prompt, system_prompt)

    def _mock_query_response(self, prompt: str) -> str:
        """Generate mock GraphQL query"""
        if "highest" in prompt.lower() or "maximum" in prompt.lower():
            return '''query {
  indexers(first: 1, orderBy: [TOTAL_STAKE_DESC]) {
    nodes {
      id
      totalStake
    }
  }
}'''
        elif "block" in prompt.lower():
            return '''query {
  indexers(blockHeight: "5000000", first: 10) {
    nodes {
      id
      totalStake
    }
  }
}'''
        else:
            return '''query {
  indexers(first: 10) {
    nodes {
      id
      totalStake
      selfStake
    }
    totalCount
  }
}'''


def create_llm_client(config: LLMConfig = None) -> BaseLLMClient:
    """Factory function to create LLM client"""
    if config is None:
        config = LLMConfig.from_env()

    if config.provider == LLMProvider.GEMINI:
        if not config.api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY is required for Gemini provider")
        return GeminiClient(config)

    elif config.provider == LLMProvider.OPENAI:
        if not config.api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI provider")
        return OpenAIClient(config)

    elif config.provider == LLMProvider.VLLM:
        return VLLMClient(config)

    else:
        return MockLLMClient(config)


# Convenience function
def get_llm(
    provider: str = None,
    api_key: str = None,
    model: str = None,
    base_url: str = None,
    temperature: float = None,
) -> BaseLLMClient:
    """
    Get LLM client with optional overrides.

    Usage:
        # Use environment variables (recommended)
        llm = get_llm()

        # Use Gemini
        llm = get_llm(provider="gemini", api_key="AIza...")

        # Use OpenAI
        llm = get_llm(provider="openai", api_key="sk-...")

        # Use vLLM (local model)
        llm = get_llm(provider="vllm", base_url="http://localhost:8000/v1")

        # Use mock for testing
        llm = get_llm(provider="mock")
    """
    config = LLMConfig.from_env()

    # Override with provided values
    if provider:
        provider_map = {
            "gemini": LLMProvider.GEMINI,
            "google": LLMProvider.GEMINI,
            "openai": LLMProvider.OPENAI,
            "gpt": LLMProvider.OPENAI,
            "vllm": LLMProvider.VLLM,
            "mock": LLMProvider.MOCK,
        }
        config.provider = provider_map.get(provider.lower(), LLMProvider.GEMINI)

    if api_key:
        config.api_key = api_key
    if model:
        config.model = model
    if base_url:
        config.base_url = base_url
    if temperature is not None:
        config.temperature = temperature

    return create_llm_client(config)
