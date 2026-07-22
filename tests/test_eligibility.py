"""rev 2: junk filter, maintainer flag, lock-reason exclusion, waterfall."""

from __future__ import annotations

import pytest

from retrieval import db, features, ingest
from retrieval.report import _exclusion_waterfall

SNAPSHOT_TS = "2026-07-22T00:00:00+00:00"


# --- is_junk boundaries (pure function) ------------------------------------


@pytest.fixture
def junk_cfg(cfg):
    return cfg["selection"]["junk_filter"]


def test_junk_body_length_boundary(junk_cfg):
    # max_clean_body_chars = 40; <= 40 qualifies, 41 does not.
    assert features.compute_is_junk("x" * 39, 0, 0, 7.0, junk_cfg) == 1
    assert features.compute_is_junk("x" * 40, 0, 0, 7.0, junk_cfg) == 1
    assert features.compute_is_junk("x" * 41, 0, 0, 7.0, junk_cfg) == 0


def test_junk_age_boundary(junk_cfg):
    assert features.compute_is_junk("x" * 10, 0, 0, 6.9, junk_cfg) == 0
    assert features.compute_is_junk("x" * 10, 0, 0, 7.0, junk_cfg) == 1


def test_junk_requires_zero_engagement(junk_cfg):
    assert features.compute_is_junk("x" * 10, 1, 0, 30.0, junk_cfg) == 0  # 1 reaction
    assert features.compute_is_junk("x" * 10, 0, 1, 30.0, junk_cfg) == 0  # 1 comment
    assert features.compute_is_junk("x" * 10, 0, 0, 30.0, junk_cfg) == 1


# --- maintainer_authored mapping -------------------------------------------


@pytest.mark.parametrize(
    "assoc,expected",
    [
        ("OWNER", 1),
        ("MEMBER", 1),
        ("COLLABORATOR", 1),
        ("CONTRIBUTOR", 0),
        ("FIRST_TIME_CONTRIBUTOR", 0),
        ("NONE", 0),
        ("", 0),
        (None, 0),
    ],
)
def test_maintainer_mapping(assoc, expected):
    assert features.compute_maintainer_authored(assoc) == expected


# --- crafted DB for lock-reason + eligibility + waterfall ------------------


def _issue(number, **kw):
    base = {
        "number": number,
        "title": f"Issue {number} title text here",
        "body": "This is a perfectly normal and adequately detailed bug report "
                "describing a real reproducible problem in depth for triage.",
        "state": "open",
        "created_at": "2026-07-20T00:00:00Z",  # recent -> in pool
        "updated_at": "2026-07-20T00:00:00Z",
        "comments": 2,
        "reactions": {"total_count": 5, "+1": 4},
        "author_association": "NONE",
        "labels": [],
        "html_url": f"https://example/{number}",
    }
    base.update(kw)
    return base


@pytest.fixture
def crafted(cfg):
    rows = [
        _issue(1),  # normal -> eligible
        _issue(2, locked=True, active_lock_reason="spam"),      # excluded
        _issue(3, locked=True, active_lock_reason="resolved"),  # excluded
        _issue(4, locked=True, active_lock_reason="off-topic"), # kept
        _issue(5, locked=True, active_lock_reason=None),        # kept
        _issue(6, created_at="2026-07-01T00:00:00Z", body="short",
               comments=0, reactions={"total_count": 0, "+1": 0}),  # junk
        _issue(7, labels=[{"name": "question"}]),               # excluded label
        _issue(8, author_association="OWNER"),                  # maintainer, kept
    ]
    conn = db.connect(":memory:")
    db.init_schema(conn)
    ingest.load_into_db(conn, rows, SNAPSHOT_TS, cfg["repo"], cfg, api_total_count=8)
    features.run_features(conn, cfg)
    yield conn
    conn.close()


def _elig(conn, number):
    return conn.execute(
        "SELECT eligible, is_junk, maintainer_authored FROM features WHERE number = ?",
        (number,),
    ).fetchone()


def test_lock_reason_exclusion(crafted):
    assert _elig(crafted, 2)["eligible"] == 0  # spam
    assert _elig(crafted, 3)["eligible"] == 0  # resolved
    assert _elig(crafted, 4)["eligible"] == 1  # off-topic not excluded
    assert _elig(crafted, 5)["eligible"] == 1  # locked but no reason


def test_junk_excluded_from_eligibility(crafted):
    row = _elig(crafted, 6)
    assert row["is_junk"] == 1
    assert row["eligible"] == 0


def test_question_label_excluded(crafted):
    assert _elig(crafted, 7)["eligible"] == 0


def test_maintainer_flag_does_not_affect_eligibility(crafted):
    row = _elig(crafted, 8)
    assert row["maintainer_authored"] == 1
    assert row["eligible"] == 1  # flagged, still eligible


def test_normal_issue_eligible(crafted):
    assert _elig(crafted, 1)["eligible"] == 1


# --- waterfall consistency -------------------------------------------------


def _pool_and_labels(conn):
    pool = conn.execute(
        """
        SELECT i.number, i.reactions_total, i.active_lock_reason, f.is_junk,
               f.eligible
        FROM issues i JOIN features f ON f.number = i.number WHERE f.in_pool = 1
        """
    ).fetchall()
    labels_by: dict[int, set] = {}
    for r in conn.execute("SELECT number, label FROM issue_labels"):
        labels_by.setdefault(r["number"], set()).add(r["label"])
    reactions_by = {r["number"]: r["reactions_total"] for r in pool}
    return pool, labels_by, reactions_by


def _waterfall(conn, cfg):
    pool, labels_by, reactions_by = _pool_and_labels(conn)
    total_open = conn.execute("SELECT COUNT(*) AS n FROM issues").fetchone()["n"]
    stale_min = cfg["selection"].get("stale_rescue_min_reactions")
    return _exclusion_waterfall(pool, labels_by, cfg, total_open, reactions_by,
                                stale_min)


def test_waterfall_monotonic_and_matches_eligible(crafted, cfg):
    steps, _ = _waterfall(crafted, cfg)
    remainings = [remaining for _, _, remaining in steps]
    assert all(b <= a for a, b in zip(remainings, remainings[1:]))
    eligible_count = crafted.execute(
        "SELECT COUNT(*) AS n FROM features WHERE eligible = 1"
    ).fetchone()["n"]
    assert steps[-1][0] == "eligible"
    assert steps[-1][2] == eligible_count


def test_waterfall_on_full_fixture(loaded, cfg):
    steps, _ = _waterfall(loaded, cfg)
    eligible_count = loaded.execute(
        "SELECT COUNT(*) AS n FROM features WHERE eligible = 1"
    ).fetchone()["n"]
    assert steps[-1][2] == eligible_count
