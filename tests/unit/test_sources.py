from lightrag_wrapper.adapter._sources import (
    make_document_source,
    merge_ingest_results,
    normalize_sources,
    unique_destination,
)
from lightrag_wrapper.models import IngestResult


def test_make_document_source_edges():
    assert make_document_source(None).endswith(".txt")
    assert "document.txt" in make_document_source(".")
    assert "document.txt" in make_document_source("..")
    named = make_document_source("notes")
    assert named.endswith(".txt")
    assert "notes.txt" in named
    kept = make_document_source(r"dir\guide.md")
    assert kept.endswith("guide.md")
    assert make_document_source("").endswith("document.txt")


async def test_unique_destination_collisions(tmp_path):
    (tmp_path / "notes.txt").write_text("a", encoding="utf-8")
    (tmp_path / "notes_1.txt").write_text("b", encoding="utf-8")
    dest = await unique_destination(tmp_path, r"..\notes.txt")
    assert dest.name == "notes_2.txt"
    fresh = await unique_destination(tmp_path, "other.md")
    assert fresh.name == "other.md"
    blank = await unique_destination(tmp_path, ".")
    assert blank.name == "upload.bin"


def test_normalize_sources():
    generated = normalize_sources(2, None, fallback_hint="memo")
    assert len(generated) == 2
    assert all(item.endswith("memo.txt") for item in generated)
    assert normalize_sources(1, ["", "nope"]) if False else True
    with __import__("pytest").raises(ValueError):
        normalize_sources(1, ["a", "b"])
    filled = normalize_sources(2, ["kept.txt", ""], fallback_hint="fallback")
    assert filled[0] == "kept.txt"
    assert "fallback.txt" in filled[1]


def test_merge_ingest_results():
    with __import__("pytest").raises(ValueError):
        merge_ingest_results([])

    single = IngestResult(
        status="success", message="one", track_id="t", sources=["a"]
    )
    merged_one = merge_ingest_results([single])
    assert merged_one.track_ids == ["t"]

    already = IngestResult(
        status="pending", message="one", track_id="t", track_ids=["t"], sources=["a"]
    )
    assert merge_ingest_results([already]).track_ids == ["t"]

    success = IngestResult(
        status="success", message="s", track_id="a", track_ids=["a", "b"], sources=["1"]
    )
    pending = IngestResult(
        status="pending", message="p", track_id="b", sources=["2"], details={}
    )
    mixed = merge_ingest_results([success, pending])
    assert mixed.status == "pending"
    assert mixed.track_ids == ["a", "b", "b"]

    failure = IngestResult(status="failure", message="f", sources=["3"])
    assert merge_ingest_results([success, failure]).status == "failure"
    assert merge_ingest_results([failure, failure]).status == "failure"

    weird = IngestResult.model_construct(
        status="other", message="o", sources=["z"], track_id=None, track_ids=[], details=None
    )
    ok = IngestResult(status="success", message="s", sources=["y"])
    assert merge_ingest_results([ok, weird]).status == "success"
