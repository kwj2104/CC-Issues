"""M3 scoring/selection: cluster dedup, tie-breaks, determinism."""

from __future__ import annotations

import csv
import hashlib
import os

from retrieval import db, report, score
from retrieval.features import _parse_iso_epoch


def _run(loaded, cfg, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    run_id = score.run_score(loaded, cfg, out_dir=out_dir)
    return run_id


def _read(out_dir, name):
    with open(os.path.join(out_dir, name), newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_one_row_per_cluster(loaded, cfg, tmp_path):
    _run(loaded, cfg, tmp_path)
    rows = _read(tmp_path, "top_1000.csv")
    cluster_ids = [r["cluster_id"] for r in rows]
    assert len(cluster_ids) == len(set(cluster_ids))


def test_highest_scoring_member_represents_cluster(loaded, cfg, tmp_path):
    _run(loaded, cfg, tmp_path)
    numbers = {r["number"] for r in _read(tmp_path, "top_1000.csv")}
    # crash cluster 101/102/103 -> 101 has the most engagement, so it wins.
    assert "101" in numbers
    assert "102" not in numbers and "103" not in numbers


def test_no_excluded_labels_selected(loaded, cfg, tmp_path):
    _run(loaded, cfg, tmp_path)
    exclude = set(cfg["selection"]["exclude_labels"])
    stale_min = cfg["selection"]["stale_rescue_min_reactions"]
    for r in _read(tmp_path, "top_1000.csv"):
        labels = set(filter(None, r["labels"].split(";")))
        bad = labels & exclude
        # only `stale` may appear, and only on a stale-rescued row (>= threshold)
        if bad:
            assert bad == {"stale"}
            assert int(r["reactions_total"]) >= stale_min



def test_maintainer_authored_column_present(loaded, cfg, tmp_path):
    _run(loaded, cfg, tmp_path)
    for name in ("top_1000.csv", "ranked_pool.csv"):
        rows = _read(tmp_path, name)
        assert rows and "maintainer_authored" in rows[0]
        for r in rows:
            assert r["maintainer_authored"] in ("0", "1")


def test_no_junk_or_excluded_lock_selected(loaded, cfg, tmp_path):
    # Extended acceptance (section 9 item 3): no selected row may carry an
    # excluded label, an excluded lock reason, or is_junk = 1.
    _run(loaded, cfg, tmp_path)
    exclude_lock = set(cfg["selection"]["exclude_lock_reasons"])
    for r in _read(tmp_path, "top_1000.csv"):
        num = int(r["number"])
        feat = loaded.execute(
            "SELECT is_junk FROM features WHERE number = ?", (num,)
        ).fetchone()
        assert feat["is_junk"] == 0
        lock = loaded.execute(
            "SELECT active_lock_reason FROM issues WHERE number = ?", (num,)
        ).fetchone()["active_lock_reason"]
        assert (lock or "").lower() not in exclude_lock


def test_every_selected_is_open(loaded, cfg, tmp_path):
    # rev 3: acceptance is state=open (all ingested issues), not a pool predicate.
    _run(loaded, cfg, tmp_path)
    for r in _read(tmp_path, "top_1000.csv"):
        state = loaded.execute(
            "SELECT state FROM issues WHERE number = ?", (int(r["number"]),)
        ).fetchone()["state"]
        assert state == "open"


def test_ranked_pool_sorted_by_score_desc(loaded, cfg, tmp_path):
    _run(loaded, cfg, tmp_path)
    rows = _read(tmp_path, "ranked_pool.csv")
    scores = [float(r["score"]) for r in rows]
    assert scores == sorted(scores, reverse=True)


def test_determinism_byte_identical(loaded, cfg, tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    _run(loaded, cfg, str(a))
    _run(loaded, cfg, str(b))
    for name in ("top_1000.csv", "ranked_pool.csv"):
        ha = hashlib.sha256((a / name).read_bytes()).hexdigest()
        hb = hashlib.sha256((b / name).read_bytes()).hexdigest()
        assert ha == hb, f"{name} not byte-identical across runs"


def test_window_variant_selections_valid(loaded, cfg):
    # rev 3: each window variant must yield a valid selection — dedup-compliant,
    # <= top_n, every member eligible and created within the window.
    scored = score._load_scored_rows(loaded, cfg["weights"])
    top_n = cfg["selection"]["top_n"]
    snap_epoch = _parse_iso_epoch(db.get_meta(loaded, "snapshot_ts"))
    for w_days in cfg["sensitivity"]["window_variants"]:
        sel = report.select_within_window(scored, snap_epoch, w_days, cfg["selection"])
        assert len(sel) <= top_n
        cluster_ids = [r["cluster_id"] for r in sel]
        assert len(cluster_ids) == len(set(cluster_ids))  # one per cluster
        for r in sel:
            assert r["eligible"] == 1
            age = (snap_epoch - _parse_iso_epoch(r["created_at"])) / 86400.0
            assert age <= w_days


def test_scores_table_ranks_eligible_only(loaded, cfg, tmp_path):
    run_id = _run(loaded, cfg, tmp_path)
    # Ineligible issues get NULL rank; eligible get a 1-based rank.
    row_401 = loaded.execute(
        "SELECT rank, selected FROM scores WHERE run_id = ? AND number = 401",
        (run_id,),
    ).fetchone()
    assert row_401["rank"] is None
    assert row_401["selected"] == 0
    top = loaded.execute(
        "SELECT rank FROM scores WHERE run_id = ? AND selected = 1 ORDER BY rank "
        "LIMIT 1",
        (run_id,),
    ).fetchone()
    assert top["rank"] == 1
