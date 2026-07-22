"""M3 — scoring, selection, and CSV emission."""

from __future__ import annotations

import csv
import math
import os
import sqlite3

from . import config as config_mod
from . import db

OUT_DIR = "out"

CSV_COLUMNS = [
    "rank", "number", "html_url", "title", "created_at", "updated_at",
    "age_days", "reactions_total", "comments", "maintainer_authored", "labels",
    "cluster_id", "cluster_size", "cluster_members",
    "score", "c_reactions", "c_comments", "c_velocity", "c_severity",
    "c_demand", "c_cluster", "run_id", "snapshot_ts",
]


def _fmt(x: float) -> str:
    """Fixed-precision float formatting for byte-identical CSV output."""
    return f"{x:.6f}"


def _components(row, weights) -> dict:
    cluster_size = row["cluster_size"] or 1
    c_cluster_feat = math.log2(cluster_size)  # 0 for singletons
    return {
        "c_reactions": weights["reactions"] * row["f_reactions"],
        "c_comments": weights["comments"] * row["f_comments"],
        "c_velocity": weights["velocity"] * row["f_velocity"],
        "c_severity": weights["severity"] * row["f_severity"],
        "c_demand": weights["demand"] * row["f_demand"],
        "c_cluster": weights["cluster"] * c_cluster_feat,
    }


def _load_scored_rows(conn: sqlite3.Connection, weights: dict) -> list[dict]:
    """Join issues+features+clusters and compute score + components for all."""
    rows = conn.execute(
        """
        SELECT i.number, i.title, i.html_url, i.created_at, i.updated_at,
               i.comments, i.reactions_total,
               f.age_days, f.f_reactions, f.f_comments, f.f_velocity,
               f.f_severity, f.f_demand, f.in_pool, f.eligible,
               f.maintainer_authored,
               c.cluster_id, c.cluster_size
        FROM issues i
        JOIN features f ON f.number = i.number
        JOIN clusters c ON c.number = i.number
        """
    ).fetchall()
    out = []
    for r in rows:
        comp = _components(r, weights)
        score = sum(comp.values())
        out.append(
            {
                "number": r["number"],
                "title": r["title"],
                "html_url": r["html_url"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "comments": r["comments"],
                "reactions_total": r["reactions_total"],
                "age_days": r["age_days"],
                "eligible": r["eligible"],
                "maintainer_authored": r["maintainer_authored"],
                "cluster_id": r["cluster_id"],
                "cluster_size": r["cluster_size"],
                "score": score,
                **comp,
            }
        )
    return out


def _rank_and_select(scored: list[dict], top_n: int) -> tuple[list[dict], list[dict]]:
    """Rank eligible by (score, reactions, number) desc; dedup clusters -> top_n.

    Returns (eligible_ranked, selected) where selected preserves ranking order
    and holds at most one issue per cluster_id.
    """
    eligible = [r for r in scored if r["eligible"]]
    eligible.sort(
        key=lambda r: (r["score"], r["reactions_total"], r["number"]),
        reverse=True,
    )
    for i, r in enumerate(eligible, start=1):
        r["rank"] = i

    selected = []
    seen_clusters = set()
    for r in eligible:
        if r["cluster_id"] in seen_clusters:
            continue
        seen_clusters.add(r["cluster_id"])
        selected.append(r)
        if len(selected) >= top_n:
            break
    return eligible, selected


def _cluster_members(conn: sqlite3.Connection) -> dict:
    members: dict[int, list] = {}
    for row in conn.execute(
        "SELECT cluster_id, number FROM clusters ORDER BY number ASC"
    ):
        members.setdefault(row["cluster_id"], []).append(row["number"])
    return members


def _labels_by_number(conn: sqlite3.Connection) -> dict:
    out: dict[int, list] = {}
    for row in conn.execute(
        "SELECT number, label FROM issue_labels ORDER BY label ASC"
    ):
        out.setdefault(row["number"], []).append(row["label"])
    return out


def _write_csv(path: str, rows: list[dict], conn, run_id, snapshot_ts,
               display_rank: bool):
    members = _cluster_members(conn)
    labels = _labels_by_number(conn)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(CSV_COLUMNS)
        for seq, r in enumerate(rows, start=1):
            member_nums = members.get(r["cluster_id"], [])[:50]
            rank_val = seq if display_rank else r["rank"]
            writer.writerow([
                rank_val,
                r["number"],
                r["html_url"],
                r["title"],
                r["created_at"],
                r["updated_at"],
                _fmt(r["age_days"]),
                r["reactions_total"],
                r["comments"],
                r["maintainer_authored"],
                ";".join(labels.get(r["number"], [])),
                r["cluster_id"],
                r["cluster_size"],
                ";".join(str(m) for m in member_nums),
                _fmt(r["score"]),
                _fmt(r["c_reactions"]),
                _fmt(r["c_comments"]),
                _fmt(r["c_velocity"]),
                _fmt(r["c_severity"]),
                _fmt(r["c_demand"]),
                _fmt(r["c_cluster"]),
                run_id,
                snapshot_ts,
            ])


def run_score(conn: sqlite3.Connection, cfg: dict, out_dir: str = OUT_DIR) -> str:
    """Score every issue, persist the run, select top_n, emit CSVs. Returns run_id."""
    snapshot_ts = db.get_meta(conn, "snapshot_ts")
    if snapshot_ts is None:
        raise RuntimeError("no snapshot_ts in meta; run ingest first")

    weights = cfg["weights"]
    run_id = config_mod.derive_run_id(snapshot_ts, cfg)

    scored = _load_scored_rows(conn, weights)
    eligible, selected = _rank_and_select(scored, cfg["selection"]["top_n"])
    selected_numbers = {r["number"] for r in selected}

    # Persist the run (idempotent replace on same run_id).
    conn.execute("DELETE FROM scores WHERE run_id = ?", (run_id,))
    conn.execute("DELETE FROM score_runs WHERE run_id = ?", (run_id,))
    conn.execute(
        "INSERT INTO score_runs (run_id, created_ts, weights_json, config_json) "
        "VALUES (?,?,?,?)",
        (
            run_id,
            snapshot_ts,  # no wall-clock after ingest -> use snapshot_ts
            config_mod.canonical_config_json({"weights": weights}),
            config_mod.canonical_config_json(cfg),
        ),
    )
    rank_by_number = {r["number"]: r.get("rank") for r in eligible}
    score_rows = []
    for r in scored:
        score_rows.append((
            run_id, r["number"], r["score"],
            r["c_reactions"], r["c_comments"], r["c_velocity"],
            r["c_severity"], r["c_demand"], r["c_cluster"],
            rank_by_number.get(r["number"]),
            1 if r["number"] in selected_numbers else 0,
        ))
    conn.executemany(
        "INSERT INTO scores (run_id, number, score, c_reactions, c_comments, "
        "c_velocity, c_severity, c_demand, c_cluster, rank, selected) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        score_rows,
    )
    conn.commit()

    _write_csv(os.path.join(out_dir, "top_1000.csv"), selected, conn, run_id,
               snapshot_ts, display_rank=True)
    _write_csv(os.path.join(out_dir, "ranked_pool.csv"), eligible, conn, run_id,
               snapshot_ts, display_rank=False)
    return run_id
