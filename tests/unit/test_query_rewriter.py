import pytest

from lightrag_wrapper.exceptions import LLMServiceError
from lightrag_wrapper.query_rewriter.query_rewriter import (
    QueryRewriter,
    _confidence,
    _extract_json_object,
    _keyword_list,
    _parse_rewrite,
)


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload
        self.prompts = []

    async def complete_prompt(self, prompt, **kwargs):
        self.prompts.append(prompt)
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


async def test_rewrite_success_and_history():
    raw = """```json
    {"rewritten_query": "Reasons for delays to Project Orion",
     "high_level_keywords": ["project delays", "Project delays"],
     "low_level_keywords": "Berlin, ORION-441",
     "is_context_dependent": true,
     "ambiguities": ["which delay"],
     "confidence": 1.4}
    ```"""
    llm = FakeLLM(raw)
    rewriter = QueryRewriter(llm)
    result = await rewriter.rewrite(
        " Why was it delayed? ",
        [{"role": "user", "content": "Project Orion"}, {"role": "", "content": ""}],
    )
    assert result.rewritten_query.startswith("Reasons")
    assert result.high_level_keywords == ["project delays"]
    assert "Berlin" in result.low_level_keywords
    assert result.confidence == 1.0
    assert result.is_context_dependent
    assert "Conversation history" in llm.prompts[0]

    empty = await rewriter.rewrite("   ")
    assert empty.rewritten_query == ""

    bare = FakeLLM('{"rewritten_query":"q"}')
    no_history = await QueryRewriter(bare).rewrite("q")
    assert no_history.high_level_keywords == []
    assert "(none)" in bare.prompts[0]


async def test_rewrite_fallbacks():
    failed = await QueryRewriter(FakeLLM(LLMServiceError("down"))).rewrite("original")
    assert failed.rewritten_query == "original"

    bad = await QueryRewriter(FakeLLM("not json")).rewrite("original")
    assert bad.rewritten_query == "original"

    array = await QueryRewriter(FakeLLM("[1, 2]")).rewrite("original")
    assert array.rewritten_query == "original"

    embedded = await QueryRewriter(
        FakeLLM('preface {"rewritten_query": "clean question"} tail')
    ).rewrite("original")
    assert embedded.rewritten_query == "clean question"

    missing = _parse_rewrite('{"high_level_keywords": ["a"]}', "original")
    assert missing.rewritten_query == "original"


def test_keyword_and_confidence_edges():
    long = "x" * 600
    keywords = _keyword_list([long, long.upper(), "ok", 1, "  "])
    assert len(keywords[0]) == 512
    assert keywords[1] == "ok"
    assert _keyword_list("a; b\nc") == ["a", "b", "c"]
    assert _keyword_list({"no": "list"}) == []
    assert _confidence(True) == 0.0
    assert _confidence("no") == 0.0
    assert _confidence(-2) == 0.0
    assert _extract_json_object("```\nnot\n```") is None
    assert _extract_json_object("no brace") is None
    assert _extract_json_object("prefix {not json} suffix") is None
    many = _keyword_list([f"k{i}" for i in range(80)])
    assert len(many) == 64
