"""Errors raised by the LightRAG adapter and LLM service."""


class LightRAGAdapterError(RuntimeError):
    """Failure talking to embedded LightRAG or the standalone server."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class LLMServiceError(RuntimeError):
    """Failure calling an upstream LLM via LiteLLM."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        operation: str | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.operation = operation
        self.retryable = retryable
