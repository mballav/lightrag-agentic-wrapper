"""Query decomposer prompt filling and DAG parsing."""

import json

from lightrag_wrapper.exceptions import LLMServiceError
from lightrag_wrapper.query_decomposer.query_decomposer import (
    QueryDecomposer,
    parse_decomposition,
    render_decomposer_prompt,
)


def test_render_inserts_catalog_without_touching_examples():
    catalog = json.dumps({"documents": [{"file_path": "leave-policy.pdf"}]})
    prompt = render_decomposer_prompt(
        "How many leave days?",
        [{"role": "user", "content": "Berlin office"}],
        catalog,
    )
    assert "How many leave days?" in prompt
    assert "leave-policy.pdf" in prompt
    assert "Berlin office" in prompt
    assert '"nodeType": "retrieve"' in prompt
    assert "23. EXECUTION CONTRACT" in prompt
    assert "carry forward the project identity and Company X" in prompt
    assert "{user_query}" not in prompt
    assert "{knowledge_base_metadata}" not in prompt
    assert "{previous_messages}" not in prompt


def test_parse_decomposition_accepts_a_dag_and_rejects_cycles():
    raw = """
    [
      {"Id": 1, "nodeType": "retrieve", "subquery": "Find the policy.", "depends_on": [], "isResultNode": false},
      {"Id": 2, "nodeType": "synthesize", "subquery": "State the allowance.", "depends_on": [1], "isResultNode": true}
    ]
    """
    nodes = parse_decomposition(raw)
    assert nodes is not None
    assert [node.id for node in nodes] == [1, 2]
    assert nodes[1].depends_on == [1]

    fenced = "```json\n" + raw + "\n```"
    assert parse_decomposition(fenced) is not None

    cycle = """
    [
      {"Id": 1, "nodeType": "retrieve", "subquery": "A", "depends_on": [2], "isResultNode": false},
      {"Id": 2, "nodeType": "retrieve", "subquery": "B", "depends_on": [1], "isResultNode": false}
    ]
    """
    assert parse_decomposition(cycle) is None
    assert parse_decomposition("not json") is None
    assert parse_decomposition('{"Id": 1}') is None
    assert parse_decomposition("[]") is None
    missing = """
    [{"Id": 2, "nodeType": "retrieve", "subquery": "B", "depends_on": [9], "isResultNode": false}]
    """
    assert parse_decomposition(missing) is None
    unknown = """
    [{"Id": 1, "nodeType": "imagine", "subquery": "B", "depends_on": [], "isResultNode": false}]
    """
    assert parse_decomposition(unknown) is None


def test_parse_decomposition_rejects_incomplete_result_structure():
    non_result_leaf = """
    [
      {"Id": 1, "nodeType": "retrieve", "subquery": "Find projects.", "depends_on": [], "isResultNode": true},
      {"Id": 2, "nodeType": "retrieve", "subquery": "Find unused evidence.", "depends_on": [], "isResultNode": false}
    ]
    """
    assert parse_decomposition(non_result_leaf) is None

    no_result = """
    [
      {"Id": 1, "nodeType": "retrieve", "subquery": "Find the policy.", "depends_on": [], "isResultNode": false}
    ]
    """
    assert parse_decomposition(no_result) is None

    root_synthesis = """
    [
      {"Id": 1, "nodeType": "synthesize", "subquery": "State the allowance.", "depends_on": [], "isResultNode": true}
    ]
    """
    assert parse_decomposition(root_synthesis) is None

    valid = """
    [
      {"Id": 1, "nodeType": "retrieve", "subquery": "Find the 2024 policy.", "depends_on": [], "isResultNode": false},
      {"Id": 2, "nodeType": "retrieve", "subquery": "Find the 2026 policy.", "depends_on": [], "isResultNode": false},
      {"Id": 3, "nodeType": "compare", "subquery": "Compare the two policies.", "depends_on": [1, 2], "isResultNode": true}
    ]
    """
    nodes = parse_decomposition(valid)
    assert nodes is not None
    assert [node.id for node in nodes] == [1, 2, 3]


def test_parse_decomposition_renumbers_ids_in_depth_first_order():
    raw = """
    [
      {"Id": 1, "nodeType": "retrieve", "subquery": "Branch A root.", "depends_on": [], "isResultNode": false},
      {"Id": 2, "nodeType": "retrieve", "subquery": "Branch B root.", "depends_on": [], "isResultNode": true},
      {"Id": 3, "nodeType": "retrieve", "subquery": "Branch A middle.", "depends_on": [1], "isResultNode": false},
      {"Id": 4, "nodeType": "retrieve", "subquery": "Branch A leaf.", "depends_on": [3], "isResultNode": true}
    ]
    """
    nodes = parse_decomposition(raw)
    assert nodes is not None
    assert [node.subquery for node in nodes] == [
        "Branch A root.",
        "Branch A middle.",
        "Branch A leaf.",
        "Branch B root.",
    ]
    assert [node.id for node in nodes] == [1, 2, 3, 4]
    assert nodes[1].depends_on == [1]
    assert nodes[2].depends_on == [2]
    assert nodes[3].depends_on == []


class _CatalogAdapter:
    def __init__(self, catalog):
        self.catalog = catalog

    async def list_document_catalog(self):
        return self.catalog


class _LLM:
    def __init__(self, payload):
        self.payload = payload
        self.prompt = None

    async def complete_prompt(self, prompt, **kwargs):
        if isinstance(self.payload, Exception):
            raise self.payload
        self.prompt = prompt
        return self.payload


async def test_decompose_puts_catalog_in_the_prompt():
    catalog = {
        "catalog_complete": True,
        "documents": [{"id": "doc-1", "file_path": "leave-policy.pdf", "preview": "18 days"}],
    }
    llm = _LLM(
        '[{"Id": 1, "nodeType": "retrieve", "subquery": "Retrieve the leave policy.",'
        ' "depends_on": [], "isResultNode": true}]'
    )
    decomposer = QueryDecomposer(llm, _CatalogAdapter(catalog))
    plan = await decomposer.decompose("How many days?", [{"role": "user", "content": "leave"}])
    assert plan is not None
    assert plan.nodes[0].subquery == "Retrieve the leave policy."
    assert "leave-policy.pdf" in llm.prompt
    assert "18 days" in llm.prompt


async def test_decompose_returns_none_when_the_model_fails():
    decomposer = QueryDecomposer(_LLM(LLMServiceError("down")), _CatalogAdapter({"documents": []}))
    assert await decomposer.decompose("How many days?") is None


async def test_decompose_continues_when_the_catalog_cannot_be_loaded():
    class BrokenAdapter:
        async def list_document_catalog(self):
            from lightrag_wrapper.exceptions import LightRAGAdapterError

            raise LightRAGAdapterError("listing failed")

    llm = _LLM(
        '[{"Id": 1, "nodeType": "retrieve", "subquery": "Retrieve the leave policy.",'
        ' "depends_on": [], "isResultNode": true}]'
    )
    plan = await QueryDecomposer(llm, BrokenAdapter()).decompose("How many days?")
    assert plan is not None
    assert "could not be loaded" in llm.prompt
