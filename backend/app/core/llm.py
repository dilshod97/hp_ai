"""Ollama LLM bilan ishlash — gpt-oss reasoning + streaming.

Model nomi runtime_config dan olinadi — UI'dan o'zgartirilishi mumkin
qayta startsiz.
"""
import json
from typing import AsyncIterator, Literal

import httpx

from app.config import settings
from app.core.runtime_config import runtime_config

ReasoningLevel = Literal["low", "medium", "high"]


class LLM:
    def __init__(self) -> None:
        self.base_url = settings.OLLAMA_BASE_URL
        self.client = httpx.AsyncClient(timeout=600.0)

    @property
    def model(self) -> str:
        return runtime_config.get("llm_model", settings.LLM_MODEL)

    def _build_options(self, prompt_len: int = 0) -> dict:
        # Context window'ni promptga qarab tanlash (juda kam — kontekst sigʻmaydi, juda koʻp — VRAM)
        # Standart: 8K (yengil), katta prompt: 16K (koʻp hujjat uchun)
        configured = runtime_config.get("num_ctx", 0)
        if configured:
            num_ctx = configured
        elif prompt_len > 12000:
            num_ctx = 32768
        elif prompt_len > 5000:
            num_ctx = 16384
        else:
            num_ctx = 8192
        return {
            "temperature": runtime_config.get("llm_temperature", settings.LLM_TEMPERATURE),
            "num_predict": runtime_config.get("llm_max_tokens", settings.LLM_MAX_TOKENS),
            "num_ctx": num_ctx,
            "num_batch": 512,
        }

    def _augment_system(
        self, system: str | None, reasoning: ReasoningLevel | None
    ) -> str | None:
        reasoning = reasoning or runtime_config.get("reasoning", "medium")
        if not reasoning:
            return system
        marker = f"Reasoning: {reasoning}"
        if not system:
            return marker
        return f"{system}\n\n{marker}"

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        reasoning: ReasoningLevel | None = None,
        json_mode: bool = False,
    ) -> str:
        prompt_len = len(prompt) + (len(system) if system else 0)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "24h",
            "options": self._build_options(prompt_len=prompt_len),
        }
        sys_msg = self._augment_system(system, reasoning)
        if sys_msg:
            payload["system"] = sys_msg
        if json_mode:
            payload["format"] = "json"

        resp = await self.client.post(f"{self.base_url}/api/generate", json=payload)
        resp.raise_for_status()
        return resp.json().get("response", "")

    async def stream(
        self,
        prompt: str,
        system: str | None = None,
        reasoning: ReasoningLevel | None = None,
    ) -> AsyncIterator[str]:
        prompt_len = len(prompt) + (len(system) if system else 0)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "keep_alive": "24h",
            "options": self._build_options(prompt_len=prompt_len),
        }
        sys_msg = self._augment_system(system, reasoning)
        if sys_msg:
            payload["system"] = sys_msg

        async with self.client.stream(
            "POST", f"{self.base_url}/api/generate", json=payload
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "response" in obj:
                    yield obj["response"]
                if obj.get("done"):
                    break

    async def close(self) -> None:
        await self.client.aclose()


llm = LLM()
