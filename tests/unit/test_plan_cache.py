"""The decomposer plan table can be created and read without a live Postgres."""

from sqlalchemy import create_engine, inspect, select
from sqlalchemy.orm import Session

from lightrag_wrapper.models import QueryPlanNode
from lightrag_wrapper.query_decomposer.plan_cache import Base, DecomposerPlanRow


def test_plan_table_is_created_only_when_missing_and_roundtrips_the_query():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    Base.metadata.create_all(engine)
    assert inspect(engine).has_table("decomposer_plan_cache")

    node = QueryPlanNode(
        id=1,
        node_type="retrieve",
        subquery="Find the fee.",
        depends_on=[],
        is_result_node=True,
    )
    with Session(engine) as session:
        session.add(
            DecomposerPlanRow(
                rewritten_query="What is the fee?",
                plan=[node.model_dump(by_alias=True)],
            )
        )
        session.commit()
        row = session.scalar(
            select(DecomposerPlanRow).where(
                DecomposerPlanRow.rewritten_query == "What is the fee?"
            )
        )

    assert row is not None
    assert row.plan == [
        {
            "Id": 1,
            "nodeType": "retrieve",
            "subquery": "Find the fee.",
            "depends_on": [],
            "isResultNode": True,
        }
    ]
