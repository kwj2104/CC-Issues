"""M2 features: text prep and each feature formula vs hand-computed values."""

from __future__ import annotations

import math

import pytest

from retrieval import features


# --- text prep -------------------------------------------------------------


def test_text_prep_strips_markup_and_headers():
    title = "Bug: the thing broke"
    body = (
        "Intro text.\n"
        "```\ncode block secret\n```\n"
        "<!-- template boilerplate comment -->\n"
        "### Environment\nmacOS 15\n"
        "### What happened\nit crashed ![shot](http://x/y.png) "
        "see https://example.com/foo for details\n"
    )
    p = features.prep_text(title, body, body_lead_chars=1500)
    ct = p["clean_text"]
    assert "secret" not in ct           # fenced code removed
    assert "boilerplate" not in ct      # HTML comment removed
    assert "http" not in ct             # URL removed
    assert "shot" not in ct             # image markdown removed
    assert "Environment" not in ct      # template header line removed
    assert "macOS 15" in ct             # content under header kept
    assert "crashed" in ct


def test_cluster_doc_doubles_title():
    p = features.prep_text("alpha beta", "gamma delta", body_lead_chars=1500)
    assert p["cluster_doc"].startswith(p["title_clean"] + " " + p["title_clean"] + " ")


def test_body_lead_truncation():
    body = "word " * 1000  # 5000 chars
    p = features.prep_text("t", body, body_lead_chars=1500)
    assert len(p["body_lead"]) <= 1500


# --- feature formulas (hand-computed on the fixture) -----------------------


def _feat(conn, number, col):
    return conn.execute(
        f"SELECT {col} FROM features WHERE number = ?", (number,)
    ).fetchone()[col]


def test_issue_101_core_features(loaded):
    # created 2026-07-12, snapshot 2026-07-22 -> 10 days; reactions 10, comments 4
    assert _feat(loaded, 101, "age_days") == pytest.approx(10.0)
    assert _feat(loaded, 101, "f_reactions") == pytest.approx(math.log2(11))
    assert _feat(loaded, 101, "f_comments") == pytest.approx(math.log2(5))
    assert _feat(loaded, 101, "f_velocity") == pytest.approx(math.log2(43))


def test_severity_regex_once_only(loaded):
    # 101 title/body hit "crash", "segfault", "after updat" -> bonus applied ONCE.
    assert _feat(loaded, 101, "f_severity") == pytest.approx(1.0)


def test_severity_label_sum(loaded):
    # 302: area:security (2) + regression (2), no regex match.
    assert _feat(loaded, 302, "f_severity") == pytest.approx(4.0)


def test_severity_cap(loaded):
    # 303: data-loss(3)+area:security(2)+high-priority(2)=7 -> capped at 5.
    assert _feat(loaded, 303, "f_severity") == pytest.approx(5.0)


def test_severity_regex_only(loaded):
    # 304: no severity labels, "freezes"/"hangs" match regex -> 1.0.
    assert _feat(loaded, 304, "f_severity") == pytest.approx(1.0)


def test_demand_applies(loaded):
    # 201: enhancement + reactions 20 >= 5 -> log2(20).
    assert _feat(loaded, 201, "f_demand") == pytest.approx(math.log2(20))


def test_demand_below_min_reactions(loaded):
    # 205: enhancement but reactions 3 < 5 -> 0.
    assert _feat(loaded, 205, "f_demand") == pytest.approx(0.0)


def test_demand_requires_label(loaded):
    # 101: bug label, no demand label -> 0.
    assert _feat(loaded, 101, "f_demand") == pytest.approx(0.0)


# --- pool + eligibility ----------------------------------------------------


def test_baseline_in_pool_is_always_one(loaded):
    # rev 3: no calendar window -> every open issue is in the baseline pool.
    rows = loaded.execute("SELECT in_pool FROM features").fetchall()
    assert rows and all(r["in_pool"] == 1 for r in rows)


def test_old_issue_now_in_pool_and_eligible(loaded):
    # 503 was window-excluded under rev 2; rev 3 keeps it (open, no bad label).
    assert _feat(loaded, 503, "in_pool") == 1
    assert _feat(loaded, 503, "eligible") == 1


def test_excluded_label_not_eligible(loaded):
    # 401 duplicate, 402 stale -> in pool but not eligible.
    assert _feat(loaded, 401, "eligible") == 0
    assert _feat(loaded, 402, "eligible") == 0


def test_age_floor(loaded):
    assert _feat(loaded, 101, "age_days") >= 1.0
