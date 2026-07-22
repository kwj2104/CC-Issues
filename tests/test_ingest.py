"""M1 ingest: parsing, PR skipping, malformed handling, label lowercasing."""

from __future__ import annotations

from retrieval import ingest


def test_pr_and_malformed_skipped(conn):
    # 26 issue rows in fixture; 1 PR + 2 malformed excluded.
    n = conn.execute("SELECT COUNT(*) AS n FROM issues").fetchone()["n"]
    assert n == 26
    # PR row 701 must not be present.
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM issues WHERE number = 701"
    ).fetchone()["n"] == 0


def test_malformed_counted_in_meta(conn):
    val = conn.execute(
        "SELECT value FROM meta WHERE key = 'malformed_count'"
    ).fetchone()["value"]
    assert val == "2"  # missing number + missing created_at


def test_labels_lowercased_and_deduped():
    parsed = ingest.parse_issue(
        {
            "number": 9,
            "title": "T",
            "created_at": "2026-07-01T00:00:00Z",
            "labels": [{"name": "Area:Security"}, {"name": "BUG"}],
            "reactions": {"total_count": 0, "+1": 0},
            "html_url": "u",
        }
    )
    assert parsed["labels"] == ["area:security", "bug"]


def test_null_body_becomes_empty():
    parsed = ingest.parse_issue(
        {
            "number": 9,
            "title": "T",
            "body": None,
            "created_at": "2026-07-01T00:00:00Z",
            "reactions": {"total_count": 0},
            "html_url": "u",
        }
    )
    assert parsed["body"] == ""


def test_missing_reactions_object():
    parsed = ingest.parse_issue(
        {
            "number": 9,
            "title": "T",
            "created_at": "2026-07-01T00:00:00Z",
            "html_url": "u",
        }
    )
    assert parsed["reactions_total"] == 0
    assert parsed["reactions_plus1"] == 0


def test_malformed_returns_none():
    assert ingest.parse_issue({"title": "no number"}) is None
    assert ingest.parse_issue({"number": 1, "created_at": None}) is None


def test_lock_fields_captured():
    parsed = ingest.parse_issue(
        {
            "number": 9,
            "title": "T",
            "created_at": "2026-07-01T00:00:00Z",
            "locked": True,
            "active_lock_reason": "spam",
            "html_url": "u",
        }
    )
    assert parsed["locked"] == 1
    assert parsed["active_lock_reason"] == "spam"


def test_lock_fields_absent_default():
    parsed = ingest.parse_issue(
        {
            "number": 9,
            "title": "T",
            "created_at": "2026-07-01T00:00:00Z",
            "html_url": "u",
        }
    )
    assert parsed["locked"] == 0
    assert parsed["active_lock_reason"] is None
