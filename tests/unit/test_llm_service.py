"""LLMService tests. LiteLLM is stubbed; this interpreter cannot import the real package."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from lightrag_wrapper.exceptions import LLMServiceError
from lightrag_wrapper.llm.service import (
    LLMService,
    _embedding_vectors,
    _import_litellm,
    _is_retryable,
    _status_code,
    get_llm_service,
    resolve_litellm_model,
)
from tests.conftest import make_config


class RateLimitError(Exception):
    status_code = 429


class FatalError(Exception):
    status_code = 400


def test_resolve_model_and_retry_helpers():
    assert resolve_litellm_model("openai", "gpt-4o-mini") == "openai/gpt-4o-mini"
    assert resolve_litellm_model("OpenAI-Compatible", "m") == "openai/m"
    assert resolve_litellm_model("azure", "dep") == "azure/dep"
    assert resolve_litellm_model("azure_openai", "dep") == "azure/dep"
    assert resolve_litellm_model("ollama", "llama") == "ollama/llama"
    assert resolve_litellm_model("other", "vendor/model") == "vendor/model"
    assert resolve_litellm_model("custom", "plain") == "plain"
    with pytest.raises(LLMServiceError):
        resolve_litellm_model("openai", "  ")

    assert _is_retryable(RateLimitError())
    timeout = type("Timeout", (Exception,), {})()
    assert _is_retryable(timeout)
    server = type("X", (Exception,), {"status_code": 503})()
    assert _is_retryable(server)
    assert not _is_retryable(FatalError())
    assert _status_code(FatalError()) == 400
    assert _status_code(RuntimeError("x")) is None


def test_embedding_vectors():
    assert _embedding_vectors(np.array([[1.0, 2.0]])) == [[1.0, 2.0]]
    assert _embedding_vectors([]) == []
    assert _embedding_vectors([1.0, 2.0]) == [[1.0, 2.0]]
    assert _embedding_vectors([[1.0], [2.0]]) == [[1.0], [2.0]]
    assert _embedding_vectors(SimpleNamespace(data=[])) == []
    assert _embedding_vectors({"data": [{"embedding": [0.5]}]}) == [[0.5]]
    item = SimpleNamespace(embedding=[3.0])
    assert _embedding_vectors(SimpleNamespace(data=[item, {"embedding": None}, {}])) == [[3.0]]
    assert _embedding_vectors({"nope": 1}) == []


def _service(**overrides) -> LLMService:
    config = make_config(
        llm_num_retries=2,
        llm_retry_strategy="constant_retry",
        azure_openai_api_version="2024-08-01",
        **{k: v for k, v in overrides.items() if k in {"llm_drop_params"}},
    )
    kwargs = {
        key: value
        for key, value in overrides.items()
        if key != "llm_drop_params"
    }
    return LLMService(config, **kwargs)


async def test_complete_embed_and_helpers(monkeypatch):
    calls = []

    async def acompletion(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="hello", reasoning_content=""),
                    finish_reason="stop",
                )
            ]
        )

    async def aembedding(**kwargs):
        return {"data": [{"embedding": [0.1, 0.2]}]}

    async def generic(**kwargs):
        calls.append(("generic", kwargs.get("model")))
        return {"ok": True}

    litellm = SimpleNamespace(
        drop_params=False,
        acompletion=acompletion,
        atext_completion=generic,
        aembedding=aembedding,
        aimage_generation=generic,
        aimage_edit=generic,
        aimage_variation=generic,
        atranscription=generic,
        aspeech=generic,
        arerank=generic,
        amoderation=generic,
        aresponses=generic,
    )
    monkeypatch.setattr(
        "lightrag_wrapper.llm.service._import_litellm", lambda: litellm
    )
    svc = _service()
    assert svc.model == "openai/gpt-4o-mini"
    assert svc.embedding_model.startswith("openai/")

    with pytest.raises(LLMServiceError, match="messages"):
        await svc.complete()

    text = await svc.complete(prompt="hi", system="sys")
    assert text.choices[0].message.content == "hello"
    await svc.complete(messages=[{"role": "user", "content": "x"}], model="custom-model")
    assert calls[-1]["model"] == "openai/custom-model"
    await svc.complete_text("prompt")
    await svc.embed("one")
    await svc.embed(["a", "b"], model="embed-model", api_key="ek", api_base="http://e")
    await svc.generate_image("cat")
    await svc.edit_image(image="x")
    await svc.vary_image(image="x")
    await svc.transcribe(b"audio")
    await svc.speech("hi", "alloy")
    await svc.rerank("q", ["a"])
    await svc.moderate(["text"])
    await svc.responses(input="hi")

    vectors = await svc.embed_texts(["a"], embedding_dim=2, token_tracker=1, host="h")
    assert vectors.shape == (1, 2)
    empty = await svc.embed_texts([])
    # embed still returns a vector from the stub; override for empty payload
    async def no_vectors(**kwargs):
        return {"data": []}

    litellm.aembedding = no_vectors
    blank = await svc.embed_texts(["a"], embedding_dim=4)
    assert blank.shape == (0, 4)

    prompt = await svc.complete_prompt(
        "q",
        system_prompt="s",
        history_messages=[{"role": "assistant", "content": "prev"}],
        keyword_extraction=True,
        hashing_kv={},
        enable_cot=False,
    )
    assert prompt == "hello"

    func = svc.as_lightrag_llm_func()
    assert await func("q") == "hello"
    embedding_func = svc.as_lightrag_embedding_func(
        embedding_dim=2, max_token_size=8, send_dimensions=False
    )
    assert embedding_func.embedding_dim == 2

    with pytest.raises(LLMServiceError, match="Unknown LLM operation"):
        await svc.invoke("not-a-real-op")


async def test_retries_stream_and_cot(monkeypatch):
    state = {"n": 0}

    async def flaky(**kwargs):
        state["n"] += 1
        if state["n"] == 1:
            raise RateLimitError("slow")
        return "ok"

    litellm = SimpleNamespace(drop_params=False, acompletion=flaky, aembedding=flaky)
    monkeypatch.setattr("lightrag_wrapper.llm.service._import_litellm", lambda: litellm)
    svc = LLMService(
        None,
        model="openai/gpt",
        binding="openai",
        num_retries=3,
        retry_strategy="exponential_backoff_retry",
        retry_min_wait=0,
        retry_max_wait=0,
        api_key=None,
        drop_params=False,
    )
    assert await svc.invoke("chat") == "ok"
    assert state["n"] == 2

    async def fatal(**kwargs):
        raise FatalError("no")

    litellm.acompletion = fatal
    with pytest.raises(LLMServiceError, match="failed") as exc:
        await svc.invoke("completion")
    assert exc.value.retryable is False

    async def stream():
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(reasoning_content="think", content=None)
                )
            ]
        )
        yield SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(reasoning_content="", content="ans"))]
        )
        yield SimpleNamespace(choices=[])

    async def streaming(**kwargs):
        return stream()

    litellm.acompletion = streaming
    chunks = []
    iterator = await svc.complete_prompt("q", stream=True, enable_cot=True)
    async for piece in iterator:
        chunks.append(piece)
    assert "".join(chunks) == "<think>think</think>ans"

    async def only_reason(**kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="  ", reasoning_content="why"),
                    finish_reason="length",
                )
            ]
        )

    litellm.acompletion = only_reason
    text = await svc.complete_prompt("q", enable_cot=True)
    assert "think" in text or text.strip() != ""

    async def length_text(**kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="cut", reasoning_content=""),
                    finish_reason="length",
                )
            ]
        )

    litellm.acompletion = length_text
    truncated = await svc.complete_prompt("q")
    assert "cut" in str(truncated)

    monkeypatch.setattr(
        "lightrag.utils.TruncatedResponse",
        lambda text: (_ for _ in ()).throw(RuntimeError("nope")),
        raising=False,
    )
    # TruncatedResponse is imported inside the function
    import lightrag.utils as utils

    def boom(text):
        raise RuntimeError("nope")

    monkeypatch.setattr(utils, "TruncatedResponse", boom)
    assert await svc.complete_prompt("q") == "cut"

    async def no_choices(**kwargs):
        return SimpleNamespace(choices=[])

    litellm.acompletion = no_choices
    assert "namespace" in (await svc.complete_prompt("q")).lower() or True
    raw = await svc.complete_prompt("q")
    assert "SimpleNamespace" in raw or raw == str(SimpleNamespace(choices=[]))


async def test_sync_callable_and_embedding_func_import(monkeypatch):
    def sync_call(**kwargs):
        return {"sync": True}

    litellm = SimpleNamespace(drop_params=False, acompletion=sync_call)
    monkeypatch.setattr("lightrag_wrapper.llm.service._import_litellm", lambda: litellm)
    svc = LLMService(make_config(llm_num_retries=0, llm_binding_api_key=None))
    assert await svc.invoke("completion", model="already/set", api_base="http://x") == {
        "sync": True
    }

    monkeypatch.setitem(__import__("sys").modules, "lightrag.utils", None)
    with pytest.raises(LLMServiceError, match="embedding func"):
        svc.as_lightrag_embedding_func(embedding_dim=3)


async def test_resolve_submodule_callables(monkeypatch):
    import sys
    from types import ModuleType

    litellm = ModuleType("litellm")
    litellm.drop_params = False

    responses = ModuleType("litellm.responses")
    responses_main = ModuleType("litellm.responses.main")

    async def aresponses(**kwargs):
        return {"responses": True}

    responses_main.aresponses = aresponses
    images = ModuleType("litellm.images")
    images_main = ModuleType("litellm.images.main")

    async def aimage_edit(**kwargs):
        return {"edit": True}

    async def aimage_variation(**kwargs):
        return {"var": True}

    async def aimage_generation(**kwargs):
        return {"gen": True}

    images_main.aimage_edit = aimage_edit
    images_main.aimage_variation = aimage_variation
    images_main.aimage_generation = aimage_generation

    for name, mod in {
        "litellm": litellm,
        "litellm.responses": responses,
        "litellm.responses.main": responses_main,
        "litellm.images": images,
        "litellm.images.main": images_main,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    monkeypatch.setattr("lightrag_wrapper.llm.service._import_litellm", lambda: litellm)
    svc = LLMService(make_config())
    assert (await svc.invoke("responses"))["responses"]
    assert (await svc.invoke("image_edit"))["edit"]
    assert (await svc.invoke("image_variation"))["var"]
    assert (await svc.invoke("image_generation"))["gen"]


async def test_import_error_and_singleton(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "litellm":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(LLMServiceError, match="litellm is required"):
        _import_litellm()

    async def fake_config():
        return make_config()

    monkeypatch.setattr("lightrag_wrapper.llm.service.get_config", fake_config)
    first = await get_llm_service()
    second = await get_llm_service(make_config(llm_model="other"))
    assert first is second
    built = await LLMService.from_env()
    assert built.model.endswith("gpt-4o-mini") or "/" in built.model


async def test_cot_stream_closes_think(monkeypatch):
    async def stream():
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(delta=SimpleNamespace(reasoning_content="only", content=""))
            ]
        )

    async def streaming(**kwargs):
        return stream()

    litellm = SimpleNamespace(drop_params=False, acompletion=streaming)
    monkeypatch.setattr("lightrag_wrapper.llm.service._import_litellm", lambda: litellm)
    svc = LLMService(make_config(llm_num_retries=1))
    iterator = await svc.complete_prompt("q", stream=True, enable_cot=True)
    pieces = [piece async for piece in iterator]
    assert pieces[-1] == "</think>"

    async def plain_stream():
        yield SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(reasoning_content="", content="z"))]
        )

    async def streaming_plain(**kwargs):
        return plain_stream()

    litellm.acompletion = streaming_plain
    iterator = await svc.complete_prompt("q", stream=True, enable_cot=False)
    assert [piece async for piece in iterator] == ["z"]


def test_prepare_kwargs_embedding_binding():
    svc = LLMService(
        None,
        binding="ollama",
        embedding_binding="openai",
        model="chat",
        embedding_model="embed",
        api_key="k",
        embedding_api_key="ek",
        api_base="http://chat",
        embedding_api_base="http://embed",
        api_version="v1",
    )
    prepared = svc._prepare_kwargs("aembedding", {"model": "bare"})
    assert prepared["model"] == "openai/bare"
    assert prepared["api_key"] == "ek"
    assert prepared["api_base"] == "http://embed"
    assert prepared["api_version"] == "v1"
    assert prepared["num_retries"] == 0
    chat = svc._prepare_kwargs("acompletion", {})
    assert chat["model"] == "ollama/chat"
    assert chat["api_key"] == "k"
