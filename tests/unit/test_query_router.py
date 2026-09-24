from lightrag_wrapper.exceptions import LLMServiceError
from lightrag_wrapper.models import QueryMode
from lightrag_wrapper.router.query_mode import QueryModeRouter


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload

    async def complete_prompt(self, prompt, **kwargs):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


async def test_route_modes():
    router = QueryModeRouter(FakeLLM('{"mode": "local", "reason": "fact"}'))
    assert await router.route("  Who owns the Berlin office? ") is QueryMode.LOCAL
    assert await QueryModeRouter(FakeLLM("hybrid")).route("how are they related") is QueryMode.HYBRID
    assert await QueryModeRouter(FakeLLM('{"mode": "naive"}')).route("find the phrase") is QueryMode.NAIVE
    assert await QueryModeRouter(FakeLLM('{"mode": "global"}')).route("overview") is QueryMode.GLOBAL
    assert await QueryModeRouter(FakeLLM('{"mode": "mix"}')).route("everything") is QueryMode.MIX


async def test_route_fallbacks():
    router = QueryModeRouter(FakeLLM(LLMServiceError("down")), default_mode=QueryMode.NAIVE)
    assert await router.route("question") is QueryMode.NAIVE
    assert await router.route("  ") is QueryMode.NAIVE

    default = QueryModeRouter(FakeLLM("```json\n[1]\n```"))
    assert await default.route("q") is QueryMode.MIX
    assert await QueryModeRouter(FakeLLM('{"mode": "bypass"}')).route("q") is QueryMode.MIX
    assert await QueryModeRouter(FakeLLM('{"mode": 1}')).route("q") is QueryMode.MIX
    assert await QueryModeRouter(FakeLLM("not-a-mode")).route("q") is QueryMode.MIX
    assert await QueryModeRouter(FakeLLM("prefix {bad} suffix")).route("q") is QueryMode.MIX

    raw_object = await QueryModeRouter(FakeLLM('note {"mode": "local"} tail')).route("q")
    assert raw_object is QueryMode.LOCAL
