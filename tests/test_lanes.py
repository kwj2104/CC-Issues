"""rev 6: rate_score, lane portfolio selection, stale rescue."""

from __future__ import annotations

import math

import pytest

from retrieval import db, features, ingest, score

WEIGHTS = {"reactions": 3.0, "comments": 1.0, "velocity": 2.0,
           "severity": 2.0, "demand": 1.0, "cluster": 2.0}


# --- rate_score ------------------------------------------------------------


def test_rate_score_hand_value():
    # reactions=1, comments=1, age=30 -> each rate = log2(1+30/30)=log2(2)=1
    # 3*1 + 1*1 + 2*severity(2) + 2*log2(cluster_size 4)=2*2
    rs = features.compute_rate_score(1, 1, 30.0, 2.0, 4, WEIGHTS)
    assert rs == pytest.approx(3 * 1 + 1 * 1 + 2 * 2.0 + 2 * 2.0)  # = 12.0


def test_rate_score_zero_engagement_singleton():
    rs = features.compute_rate_score(0, 0, 10.0, 0.0, 1, WEIGHTS)
    assert rs == pytest.approx(0.0)


def test_rate_score_uses_rates_not_totals():
    # same reactions, older issue -> lower rate_score (age correction).
    young = features.compute_rate_score(30, 0, 10.0, 0.0, 1, WEIGHTS)
    old = features.compute_rate_score(30, 0, 100.0, 0.0, 1, WEIGHTS)
    assert young > old


# --- lane selection --------------------------------------------------------


def _row(num, clu, score_, rate, sev=0.0, age=10.0, cs=1, react=0, eligible=1):
    return {
        "number": num, "cluster_id": clu, "score": score_, "rate_score": rate,
        "f_severity": sev, "age_days": age, "cluster_size": cs,
        "reactions_total": react, "eligible": eligible,
    }


def _sel_cfg():
    return {
        "top_n": 4,
        "lanes": [
            {"name": "big-bets", "slots": 2, "rank_by": ["score"]},
            {"name": "emerging", "slots": 1, "filter": "age_days <= 30",
             "rank_by": ["rate_score"]},
            {"name": "severity", "slots": 1, "filter": "severity >= 2.0",
             "rank_by": ["severity", "rate_score"]},
        ],
        "spill_order": ["global-score"],
    }


def test_lane_fill_and_ordering():
    rows = [
        _row(1, 10, 100, 10),
        _row(2, 20, 90, 9),
        _row(3, 30, 80, 8, sev=3.0, age=10),
        _row(4, 40, 70, 50, age=10),   # emerging: highest rate, age<=30
        _row(5, 50, 60, 1, age=100),
    ]
    selected, lane_fill, spill_fill = score.select_lanes(rows, _sel_cfg())
    assert lane_fill == {"big-bets": 2, "emerging": 1, "severity": 1}
    assert sum(spill_fill.values()) == 0
    lane_of = {r["number"]: r["selection_lane"] for r in selected}
    assert lane_of == {1: "big-bets", 2: "big-bets", 4: "emerging", 3: "severity"}


def test_cross_lane_cluster_dedup():
    # r4 shares cluster 10 with r1 (claimed by big-bets) -> emerging must skip it.
    rows = [
        _row(1, 10, 100, 10),
        _row(2, 20, 90, 9),
        _row(3, 30, 80, 8, age=10),
        _row(4, 10, 70, 50, age=10),   # same cluster as #1
        _row(5, 50, 60, 5, age=100),
    ]
    selected, lane_fill, spill_fill = score.select_lanes(rows, _sel_cfg())
    nums = {r["number"] for r in selected}
    assert 4 not in nums                       # cluster 10 already claimed
    assert lane_fill["emerging"] == 1
    lane_of = {r["number"]: r["selection_lane"] for r in selected}
    assert lane_of[3] == "emerging"            # next-best age<=30 by rate
    # severity lane finds nothing (no sev>=2 unclaimed) -> spill fills to top_n.
    assert lane_fill["severity"] == 0
    assert sum(spill_fill.values()) == 1
    assert selected[-1]["selection_lane"] == "spill:global-score"
    assert len(selected) == 4


def test_lane_underfill_spills():
    # only 3 eligible rows, top_n 4 -> one slot spills (but nothing left to fill).
    rows = [_row(1, 10, 100, 10), _row(2, 20, 90, 9), _row(3, 30, 80, 8, age=10)]
    selected, lane_fill, spill_fill = score.select_lanes(rows, _sel_cfg())
    assert len(selected) == 3            # cannot exceed supply
    assert all(r.get("selection_lane") for r in selected)


def test_every_selected_tagged():
    rows = [_row(i, i, 100 - i, 10 - i, age=10) for i in range(1, 6)]
    selected, _, _ = score.select_lanes(rows, _sel_cfg())
    assert all(r["selection_lane"] for r in selected)
    assert len(selected) == 4


def test_ineligible_never_selected():
    rows = [_row(1, 10, 100, 10, eligible=0), _row(2, 20, 90, 9)]
    selected, _, _ = score.select_lanes(rows, _sel_cfg())
    assert {r["number"] for r in selected} == {2}


# --- stale rescue boundary -------------------------------------------------

SNAPSHOT_TS = "2026-07-22T00:00:00+00:00"


def _stale_issue(num, reactions):
    return {
        "number": num, "title": f"stale {num}",
        "body": "A detailed and adequately long body describing the stale issue "
                "so that it is not treated as junk by the junk filter at all.",
        "state": "open", "created_at": "2026-07-10T00:00:00Z",
        "updated_at": "2026-07-10T00:00:00Z", "comments": 0,
        "reactions": {"total_count": reactions, "+1": reactions},
        "author_association": "NONE",
        "labels": [{"name": "stale"}],
        "html_url": f"u/{num}",
    }


@pytest.mark.parametrize("reactions,expected", [(9, 0), (10, 1)])
def test_stale_rescue_boundary(cfg, reactions, expected):
    conn = db.connect(":memory:")
    db.init_schema(conn)
    ingest.load_into_db(conn, [_stale_issue(1, reactions)], SNAPSHOT_TS,
                        cfg["repo"], cfg, api_total_count=1)
    features.run_features(conn, cfg)
    elig = conn.execute(
        "SELECT eligible FROM features WHERE number = 1"
    ).fetchone()["eligible"]
    assert elig == expected  # 9 -> excluded, 10 -> rescued (threshold is 10)
    conn.close()
