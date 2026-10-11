"""Unified LiteLLM client for chat, embeddings, and other LLM APIs.

LiteLLM already supports ``num_retries`` on completion/responses and has
``*_with_retries`` helpers that use tenacity. Coverage is uneven for
embedding, image, audio, rerank, and moderation. This service therefore
applies one tenacity policy to every operation so retries, backoff, and
error wrapping behave the same regardless of request type.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from typing import Any, Literal

import numpy as np
import tenacity

from lightrag_wrapper.config import LightRAGConfig, get_config
from lightrag_wrapper.exceptions import LLMServiceError
from lightrag_wrapper.log import get_logger

logger = get_logger(__name__)

RetryStrategy = Literal["exponential_backoff_retry", "constant_retry"]

# Kwargs LightRAG injects that must not be forwarded to LiteLLM.
_LIGHTRAG_COMPLETION_DROP = frozenset(
    {
        "hashing_kv",
        "token_tracker",
        "openai_client_configs",
        "use_azure",
        "azure_deployment",
        "image_inputs",
        "host",
        "keyword_extraction",
        "entity_extraction",
        "enable_cot",
        "_priority",
    }
)
_LIGHTRAG_EMBED_DROP = frozenset(
    {
        "token_tracker",
        "client_configs",
        "use_azure",
        "azure_deployment",
        "host",
        "embed_model",
        "context",
        "query_prefix",
        "document_prefix",
        "max_token_size",
        "embedding_dim",
        "_priority",
    }
)

# Public operation names → async LiteLLM entry points.
# Callers may also pass a raw litellm attribute name via ``invoke``.
_OPERATION_ALIASES: dict[str, str] = {
    "completion": "acompletion",
    "chat": "acompletion",
    "chat_completion": "acompletion",
    "acompletion": "acompletion",
    "text_completion": "atext_completion",
    "atext_completion": "atext_completion",
    "embedding": "aembedding",
    "embeddings": "aembedding",
    "aembedding": "aembedding",
    "image_generation": "aimage_generation",
    "aimage_generation": "aimage_generation",
    "image_edit": "aimage_edit",
    "aimage_edit": "aimage_edit",
    "image_variation": "aimage_variation",
    "aimage_variation": "aimage_variation",
    "transcription": "atranscription",
    "atranscription": "atranscription",
    "speech": "aspeech",
    "aspeech": "aspeech",
    "rerank": "arerank",
    "arerank": "arerank",
    "moderation": "amoderation",
    "amoderation": "amoderation",
    "responses": "aresponses",
    "aresponses": "aresponses",
}

# Exceptions that are worth retrying (LiteLLM + transport).
_RETRYABLE_TYPE_NAMES = frozenset(
    {
        "APIConnectionError",
        "APIError",
        "InternalServerError",
        "RateLimitError",
        "ServiceUnavailableError",
        "Timeout",
        "TimeoutError",
    }
)


def _import_litellm() -> Any:
    try:
        import litellm
    except ImportError as exc:
        raise LLMServiceError(
            "litellm is required for LLMService. Install it from requirements.txt."
        ) from exc
    return litellm


def resolve_litellm_model(binding: str, model: str) -> str:
    """Map wrapper binding + model name to a LiteLLM model id."""
    cleaned = model.strip()
    if not cleaned:
        raise LLMServiceError("model must be a non-empty string")
    if "/" in cleaned:
        return cleaned

    provider = binding.strip().lower()
    if provider in {"openai", "openai_compatible", "openai-compatible"}:
        return f"openai/{cleaned}"
    if provider in {"azure_openai", "azure"}:
        return f"azure/{cleaned}"
    if provider == "ollama":
        return f"ollama/{cleaned}"
    return cleaned


def _is_retryable(exc: BaseException) -> bool:
    name = type(exc).__name__
    if name in _RETRYABLE_TYPE_NAMES:
        return True
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and (status == 429 or status >= 500):
        return True
    return False


def _status_code(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    return status if isinstance(status, int) else None


def _embedding_vectors(response: Any) -> list[list[float]]:
    """Normalize a LiteLLM embedding response into a list of vectors."""
    if isinstance(response, np.ndarray):
        return response.tolist()
    if isinstance(response, list):
        if not response:
            return []
        if isinstance(response[0], (int, float)):
            return [list(response)]  # type: ignore[arg-type]
        return [list(item) for item in response]

    data = getattr(response, "data", None)
    if data is None and isinstance(response, dict):
        data = response.get("data")
    if not data:
        return []

    vectors: list[list[float]] = []
    for item in data:
        embedding = getattr(item, "embedding", None)
        if embedding is None and isinstance(item, dict):
            embedding = item.get("embedding")
        if embedding is None:
            continue
        vectors.append(list(embedding))
    return vectors


class LLMService:
    """Delegate any LLM request to a configured provider via LiteLLM."""

    def __init__(
        self,
        config: LightRAGConfig | None = None,
        *,
        model: str | None = None,
        embedding_model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        api_version: str | None = None,
        timeout: float | None = None,
        num_retries: int | None = None,
        retry_strategy: RetryStrategy | None = None,
        retry_min_wait: float | None = None,
        retry_max_wait: float | None = None,
        drop_params: bool | None = None,
        binding: str | None = None,
        embedding_binding: str | None = None,
        embedding_api_key: str | None = None,
        embedding_api_base: str | None = None,
    ) -> None:
        self._config = config
        self._binding = binding or (config.llm_binding if config else "openai")
        self._embedding_binding = embedding_binding or (
            config.embedding_binding if config else self._binding
        )
        self._model = model or (config.llm_model if config else "gpt-4o-mini")
        self._embedding_model = embedding_model or (
            config.embedding_model if config else self._model
        )
        self._api_key = api_key
        if self._api_key is None and config is not None:
            self._api_key = config.llm_binding_api_key
        self._api_base = api_base
        if self._api_base is None and config is not None:
            self._api_base = config.llm_binding_host
        self._api_version = api_version
        if self._api_version is None and config is not None:
            self._api_version = config.azure_openai_api_version
        self._timeout = (
            float(timeout)
            if timeout is not None
            else float(config.llm_timeout if config is not None else 240)
        )
        self._num_retries = (
            num_retries
            if num_retries is not None
            else (config.llm_num_retries if config is not None else 3)
        )
        self._retry_strategy: RetryStrategy = (
            retry_strategy
            if retry_strategy is not None
            else (
                config.llm_retry_strategy
                if config is not None
                else "exponential_backoff_retry"
            )
        )
        self._retry_min_wait = (
            retry_min_wait
            if retry_min_wait is not None
            else (config.llm_retry_min_wait if config is not None else 1.0)
        )
        self._retry_max_wait = (
            retry_max_wait
            if retry_max_wait is not None
            else (config.llm_retry_max_wait if config is not None else 10.0)
        )
        self._drop_params = (
            drop_params
            if drop_params is not None
            else (config.llm_drop_params if config is not None else True)
        )
        self._embedding_api_key = embedding_api_key
        if self._embedding_api_key is None and config is not None:
            self._embedding_api_key = config.resolved_embedding_api_key
        if self._embedding_api_key is None:
            self._embedding_api_key = self._api_key
        self._embedding_api_base = embedding_api_base
        if self._embedding_api_base is None and config is not None:
            self._embedding_api_base = config.embedding_binding_host
        if self._embedding_api_base is None:
            self._embedding_api_base = self._api_base

        self._litellm: Any | None = None

    def _litellm_mod(self) -> Any:
        if self._litellm is None:
            litellm = _import_litellm()
            if self._drop_params:
                litellm.drop_params = True
            self._litellm = litellm
        return self._litellm

    @classmethod
    async def from_env(cls) -> LLMService:
        return cls(await get_config())

    @property
    def model(self) -> str:
        return resolve_litellm_model(self._binding, self._model)

    @property
    def embedding_model(self) -> str:
        return resolve_litellm_model(self._embedding_binding, self._embedding_model)

    # ------------------------------------------------------------------
    # High-level helpers
    # ------------------------------------------------------------------

    async def complete(
        self,
        messages: Sequence[Mapping[str, Any]] | None = None,
        *,
        prompt: str | None = None,
        system: str | None = None,
        model: str | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Chat completion (``litellm.acompletion``)."""
        resolved = list(messages) if messages is not None else []
        if not resolved:
            if prompt is None:
                raise LLMServiceError(
                    "complete() requires messages=... or prompt=...",
                    operation="completion",
                )
            if system:
                resolved.append({"role": "system", "content": system})
            resolved.append({"role": "user", "content": prompt})
        return await self.invoke(
            "completion",
            messages=resolved,
            model=model,
            stream=stream,
            **kwargs,
        )

    async def complete_text(
        self,
        prompt: str,
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Legacy text completion (``litellm.atext_completion``)."""
        return await self.invoke(
            "text_completion",
            prompt=prompt,
            model=model,
            **kwargs,
        )

    async def embed(
        self,
        input: str | Sequence[str],
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Create embeddings (``litellm.aembedding``)."""
        return await self.invoke(
            "embedding",
            input=input,
            model=model or self.embedding_model,
            api_key=kwargs.pop("api_key", self._embedding_api_key),
            api_base=kwargs.pop("api_base", self._embedding_api_base),
            **kwargs,
        )

    async def generate_image(self, prompt: str, *, model: str | None = None, **kwargs: Any) -> Any:
        return await self.invoke("image_generation", prompt=prompt, model=model, **kwargs)

    async def edit_image(self, *, model: str | None = None, **kwargs: Any) -> Any:
        return await self.invoke("image_edit", model=model, **kwargs)

    async def vary_image(self, *, model: str | None = None, **kwargs: Any) -> Any:
        return await self.invoke("image_variation", model=model, **kwargs)

    async def transcribe(self, file: Any, *, model: str | None = None, **kwargs: Any) -> Any:
        return await self.invoke("transcription", file=file, model=model, **kwargs)

    async def speech(
        self,
        input: str,
        voice: str,
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> Any:
        return await self.invoke(
            "speech",
            input=input,
            voice=voice,
            model=model,
            **kwargs,
        )

    async def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> Any:
        return await self.invoke(
            "rerank",
            query=query,
            documents=list(documents),
            model=model,
            **kwargs,
        )

    async def moderate(self, input: str | Sequence[str], *, model: str | None = None, **kwargs: Any) -> Any:
        return await self.invoke("moderation", input=input, model=model, **kwargs)

    async def responses(self, *, model: str | None = None, **kwargs: Any) -> Any:
        """OpenAI Responses API (``litellm.aresponses``)."""
        return await self.invoke("responses", model=model, **kwargs)

    # ------------------------------------------------------------------
    # LightRAG SDK adapters (plain str / ndarray return types)
    # ------------------------------------------------------------------

    async def complete_prompt(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        history_messages: Sequence[Mapping[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str | AsyncIterator[str]:
        """LightRAG-compatible chat call returning plain text (or a text stream)."""
        enable_cot = bool(kwargs.get("enable_cot", False))
        call_kwargs = self._normalize_lightrag_completion_kwargs(kwargs)
        stream = bool(call_kwargs.get("stream"))

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            messages.extend(dict(item) for item in history_messages)
        messages.append({"role": "user", "content": prompt})

        response = await self.complete(messages=messages, **call_kwargs)
        if stream:
            return self._iter_completion_text(response, enable_cot=enable_cot)
        return self._extract_completion_text(response, enable_cot=enable_cot)

    async def embed_texts(
        self,
        texts: Sequence[str],
        *,
        embedding_dim: int | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """LightRAG-compatible embedding call returning a float32 ndarray."""
        call_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in _LIGHTRAG_EMBED_DROP
        }
        if embedding_dim is not None:
            call_kwargs.setdefault("dimensions", embedding_dim)

        response = await self.embed(list(texts), **call_kwargs)
        vectors = _embedding_vectors(response)
        if not vectors:
            return np.zeros((0, embedding_dim or 0), dtype=np.float32)
        return np.array(vectors, dtype=np.float32)

    def as_lightrag_llm_func(self) -> Callable[..., Awaitable[Any]]:
        """Return an ``llm_model_func`` suitable for the LightRAG SDK."""

        async def llm_model_func(
            prompt: str,
            system_prompt: str | None = None,
            history_messages: list | None = None,
            **kwargs: Any,
        ) -> Any:
            return await self.complete_prompt(
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                **kwargs,
            )

        return llm_model_func

    def as_lightrag_embedding_func(
        self,
        *,
        embedding_dim: int,
        max_token_size: int | None = None,
        model_name: str | None = None,
        send_dimensions: bool = False,
    ) -> Any:
        """Return an ``EmbeddingFunc`` suitable for the LightRAG SDK."""
        try:
            from lightrag.utils import EmbeddingFunc
        except ImportError as exc:
            raise LLMServiceError(
                "lightrag-hku is required to build a LightRAG embedding func."
            ) from exc

        async def _embed(texts: list[str], **kwargs: Any) -> np.ndarray:
            dim = kwargs.pop("embedding_dim", None)
            return await self.embed_texts(
                texts,
                embedding_dim=dim if send_dimensions else None,
                **kwargs,
            )

        return EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=max_token_size,
            func=_embed,
            model_name=model_name or self._embedding_model,
            send_dimensions=send_dimensions,
        )

    async def invoke(self, operation: str, /, **kwargs: Any) -> Any:
        """Dispatch any supported LLM request by operation name.

        ``operation`` may be a friendly alias (``completion``, ``embedding``,
        …) or a raw LiteLLM async attribute (``acompletion``, ``aembedding``,
        …). Extra kwargs are forwarded unchanged so callers can send any
        provider-specific parameter LiteLLM accepts.
        """
        litellm = self._litellm_mod()
        attr_name = _OPERATION_ALIASES.get(operation.strip().lower(), operation.strip())
        fn = getattr(litellm, attr_name, None)
        if fn is None or not callable(fn):
            # Some APIs live on submodules (e.g. aresponses).
            fn = self._resolve_litellm_callable(litellm, attr_name)
        if fn is None:
            supported = ", ".join(sorted(set(_OPERATION_ALIASES)))
            raise LLMServiceError(
                f"Unknown LLM operation {operation!r}. "
                f"Use one of: {supported}, or any litellm async callable name.",
                operation=operation,
            )

        call_kwargs = self._prepare_kwargs(attr_name, kwargs)
        return await self._execute(attr_name, fn, call_kwargs)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _normalize_lightrag_completion_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        prepared = dict(kwargs)
        keyword_extraction = bool(prepared.pop("keyword_extraction", False))
        entity_extraction = bool(prepared.pop("entity_extraction", False))
        prepared.pop("enable_cot", None)

        for key in list(prepared):
            if key in _LIGHTRAG_COMPLETION_DROP:
                prepared.pop(key, None)

        if (
            (keyword_extraction or entity_extraction)
            and prepared.get("response_format") is None
        ):
            prepared["response_format"] = {"type": "json_object"}

        return prepared

    def _extract_completion_text(self, response: Any, *, enable_cot: bool) -> str:
        if isinstance(response, str):
            return response

        choices = getattr(response, "choices", None) or []
        if not choices:
            return str(response)

        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None) if message is not None else None
        reasoning = ""
        if message is not None:
            reasoning = getattr(message, "reasoning_content", None) or ""

        text = content or ""
        if enable_cot and reasoning and reasoning.strip() and not text.strip():
            text = f"<think>{reasoning}</think>"

        finish_reason = getattr(choices[0], "finish_reason", None)
        if finish_reason == "length" and text:
            try:
                from lightrag.utils import TruncatedResponse

                return TruncatedResponse(text)
            except Exception:
                return text
        return text

    async def _iter_completion_text(
        self,
        response: Any,
        *,
        enable_cot: bool,
    ) -> AsyncIterator[str]:
        cot_opened = False
        async for chunk in response:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            if delta is None:
                continue
            reasoning = getattr(delta, "reasoning_content", None) or ""
            content = getattr(delta, "content", None) or ""
            if enable_cot and reasoning:
                if not cot_opened:
                    cot_opened = True
                    yield f"<think>{reasoning}"
                else:
                    yield reasoning
            if content:
                if cot_opened:
                    cot_opened = False
                    yield f"</think>{content}"
                else:
                    yield content
        if cot_opened:
            yield "</think>"

    def _resolve_litellm_callable(self, litellm: Any, attr_name: str) -> Any | None:
        candidates: list[Any] = []
        if attr_name in {"aresponses", "responses"}:
            try:
                from litellm.responses.main import aresponses

                candidates.append(aresponses)
            except Exception:
                pass
        if attr_name in {"aimage_edit", "image_edit"}:
            try:
                from litellm.images.main import aimage_edit

                candidates.append(aimage_edit)
            except Exception:
                pass
        if attr_name in {"aimage_variation", "image_variation"}:
            try:
                from litellm.images.main import aimage_variation

                candidates.append(aimage_variation)
            except Exception:
                pass
        if attr_name in {"aimage_generation", "image_generation"}:
            try:
                from litellm.images.main import aimage_generation

                candidates.append(aimage_generation)
            except Exception:
                pass
        for candidate in candidates:
            if callable(candidate):
                return candidate
        return getattr(litellm, attr_name, None)

    def _prepare_kwargs(self, operation: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        prepared = dict(kwargs)

        if "model" not in prepared or prepared["model"] is None:
            if operation == "aembedding":
                prepared["model"] = self.embedding_model
            else:
                prepared["model"] = self.model
        elif isinstance(prepared["model"], str) and "/" not in prepared["model"]:
            binding = (
                self._embedding_binding if operation == "aembedding" else self._binding
            )
            prepared["model"] = resolve_litellm_model(binding, prepared["model"])

        prepared.setdefault("timeout", self._timeout)
        prepared.setdefault("drop_params", self._drop_params)

        # Avoid double-retry: we own the tenacity loop below.
        prepared["num_retries"] = 0
        prepared["max_retries"] = 0

        if "api_key" not in prepared:
            if operation == "aembedding":
                prepared["api_key"] = self._embedding_api_key
            else:
                prepared["api_key"] = self._api_key
        if prepared.get("api_key") is None:
            prepared.pop("api_key", None)

        if "api_base" not in prepared and "base_url" not in prepared:
            base = (
                self._embedding_api_base
                if operation == "aembedding"
                else self._api_base
            )
            if base:
                prepared["api_base"] = base

        if self._api_version and "api_version" not in prepared:
            prepared["api_version"] = self._api_version

        return prepared

    async def _execute(
        self,
        operation: str,
        fn: Callable[..., Any],
        kwargs: dict[str, Any],
    ) -> Any:
        attempts = max(1, self._num_retries)

        if self._retry_strategy == "exponential_backoff_retry":
            wait = tenacity.wait_exponential(
                multiplier=max(self._retry_min_wait, 0.1),
                min=self._retry_min_wait,
                max=max(self._retry_max_wait, self._retry_min_wait),
            )
        else:
            wait = tenacity.wait_fixed(0)

        retryer = tenacity.AsyncRetrying(
            stop=tenacity.stop_after_attempt(attempts),
            wait=wait,
            retry=tenacity.retry_if_exception(_is_retryable),
            reraise=True,
            before_sleep=self._log_retry,
        )

        try:
            async for attempt in retryer:
                with attempt:
                    result = fn(**kwargs)
                    if isinstance(result, Awaitable):
                        return await result
                    return result
        except LLMServiceError:
            raise
        except Exception as exc:
            raise LLMServiceError(
                f"LLM {operation} failed after {attempts} attempt(s): {exc}",
                status_code=_status_code(exc),
                operation=operation,
                retryable=_is_retryable(exc),
            ) from exc

        raise LLMServiceError(
            f"LLM {operation} failed with no result",
            operation=operation,
        )

    def _log_retry(self, retry_state: tenacity.RetryCallState) -> None:
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        logger.warning(
            "Retrying LLM call",
            attempt=retry_state.attempt_number,
            max_attempts=max(1, self._num_retries),
            error_type=type(exc).__name__ if exc else "unknown",
            error=str(exc) if exc else None,
        )


_llm_service: LLMService | None = None


async def get_llm_service(config: LightRAGConfig | None = None) -> LLMService:
    """Return a process-wide LLMService (created on first use)."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService(config or await get_config())
    return _llm_service
