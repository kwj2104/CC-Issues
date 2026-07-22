"""M2 clustering: union-find groups the two known duplicate families."""

from __future__ import annotations


def _cluster_of(conn, number):
    return conn.execute(
        "SELECT cluster_id, cluster_size FROM clusters WHERE number = ?", (number,)
    ).fetchone()


def test_crash_group_clustered(loaded):
    # 101, 102, 103 are near-duplicate crash-on-update reports.
    ids = {_cluster_of(loaded, n)["cluster_id"] for n in (101, 102, 103)}
    assert len(ids) == 1


def test_theme_group_clustered(loaded):
    # 201, 202, 203 are near-duplicate custom-theme requests; 401 is a
    # (label-excluded) duplicate that still joins the cluster mass.
    ids = {_cluster_of(loaded, n)["cluster_id"] for n in (201, 202, 203, 401)}
    assert len(ids) == 1
    assert _cluster_of(loaded, 201)["cluster_size"] >= 4


def test_singletons_are_own_cluster(loaded):
    # A distinct filler issue should not fold into either duplicate family.
    c = _cluster_of(loaded, 604)  # documentation issue, unique vocabulary
    assert c["cluster_size"] == 1


def test_cluster_id_is_stable_min_number(loaded):
    # cluster_id is the smallest member number (deterministic).
    assert _cluster_of(loaded, 103)["cluster_id"] == 101
    assert _cluster_of(loaded, 203)["cluster_id"] == 201


def test_every_issue_has_a_cluster(loaded):
    n_issues = loaded.execute("SELECT COUNT(*) AS n FROM issues").fetchone()["n"]
    n_clustered = loaded.execute("SELECT COUNT(*) AS n FROM clusters").fetchone()["n"]
    assert n_issues == n_clustered
