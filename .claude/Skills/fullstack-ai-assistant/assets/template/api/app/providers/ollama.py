"""Local models served by Ollama via the official Python SDK."""

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx
from ollama import AsyncClient, ResponseError

from app.providers.base import Provider, ProviderError
from app.schemas import ChatMessage, ModelInfo

logger = logging.getLogger(__name__)


def _format_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    formatted: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m.content, str):
            formatted.append({"role": m.role, "content": m.content})
            continue
        text_parts: list[str] = []
        images: list[str] = []
        for block in m.content:
            if block.type == "text" and block.text:
                text_parts.append(block.text)
            elif block.type == "image" and block.data:
                images.append(block.data)
        msg: dict[str, Any] = {"role": m.role, "content": "".join(text_parts)}
        if images:
            msg["images"] = images
        formatted.append(msg)
    return formatted


def _extract_error_message(err_str: str) -> str:
    try:
        data = json.loads(err_str)
        if isinstance(data, dict):
            err_obj = data.get("error")
            if isinstance(err_obj, dict) and "message" in err_obj:
                return str(err_obj["message"])
            if "message" in data:
                return str(data["message"])
    except Exception:
        pass
    return err_str


class OllamaProvider(Provider):
    id = "ollama"
    label = "Ollama (local)"
    local = True

    def __init__(self, host: str, enabled: bool = True, timeout: float = 120.0) -> None:
        self._host = host
        self._enabled = enabled
        self._client = AsyncClient(host=host, timeout=timeout)

    @property
    def configured(self) -> bool:
        return self._enabled and bool(self._host)

    async def list_models(self) -> list[ModelInfo]:
        try:
            response = await self._client.list()
        except (httpx.HTTPError, ConnectionError, ResponseError) as exc:
            raise ProviderError(f"Ollama is not reachable at {self._host}") from exc

        models: list[ModelInfo] = []
        for item in response.models:
            if not item.model:
                continue
            details = item.details
            # Embedding-only models cannot chat, so keep them out of the dropdown.
            families = (details.families or []) if details else []
            if any("bert" in f for f in families) or "embed" in item.model:
                continue
            models.append(
                ModelInfo(
                    id=item.model,
                    name=item.model.removesuffix(":latest"),
                    provider=self.id,
                    local=True,
                    size_bytes=item.size,
                    parameter_size=details.parameter_size if details else None,
                    family=details.family if details else None,
                )
            )
        return sorted(models, key=lambda m: m.name)

    async def stream_chat(
        self,
        model: str,
        messages: list[ChatMessage],
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        options = {"temperature": temperature} if temperature is not None else None
        ollama_messages = _format_messages(messages)

        try:
            stream = await self._client.chat(
                model=model,
                messages=ollama_messages,
                stream=True,
                options=options,
            )
            async for chunk in stream:
                if chunk.message and chunk.message.content:
                    yield chunk.message.content
        except ResponseError as exc:
            err_msg = _extract_error_message(str(exc.error))
            raise ProviderError(f"Ollama: {err_msg}") from exc
        except (httpx.HTTPError, ConnectionError) as exc:
            raise ProviderError(f"Ollama is not reachable at {self._host}") from exc
        except Exception as exc:
            if isinstance(exc, ProviderError):
                raise
            raise ProviderError(f"Ollama error: {exc}") from exc
