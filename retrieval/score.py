"""M3 — scoring, lane-based selection, and CSV emission."""

from __future__ import annotations

import csv
import math
import operator
import os
import re
import sqlite3

from . import config as config_mod
from . import db

OUT_DIR = "out"

CSV_COLUMNS = [
    "rank", "number", "html_url", "title", "created_at", "updated_at",
    "age_days", "reactions_total", "reactions_plus1", "comments",
    "maintainer_authored", "labels",
    "cluster_id", "cluster_size", "cluster_members",
    "score", "rate_score", "selection_lane",
    "c_reactions", "c_comments", "c_velocity", "c_severity",
    "c_demand", "c_cluster", "run_id", "snapshot_ts",
]

# Lane filter / rank vocabulary. Config field names map to row-dict keys.
_FIELD_MAP = {
    "age_days": "age_days",
    "severity": "f_severity",
    "cluster_size": "cluster_size",
    "score": "score",
    "rate_score": "rate_score",
    "reactions_total": "reactions_total",
}
_OPS = {"<=": operator.le, ">=": operator.ge, "<": operator.lt,
        ">": operator.gt, "==": operator.eq}
_FILTER_RE = re.compile(r"^\s*(\w+)\s*(<=|>=|<|>|==)\s*([-\d.]+)\s*$")


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
               i.comments, i.reactions_total, i.reactions_plus1,
               f.age_days, f.f_reactions, f.f_comments, f.f_velocity,
               f.f_severity, f.f_demand, f.rate_score, f.in_pool, f.eligible,
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
                "reactions_plus1": r["reactions_plus1"],
                "age_days": r["age_days"],
                "f_severity": r["f_severity"],
                "rate_score": r["rate_score"],
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
    """Legacy single-score selection (baseline for diagnostics/sensitivity).

    Rank eligible by (score, reactions, number) desc; keep the first member of
    each cluster up to top_n.
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


# --- lane-based portfolio selection (rev 6) --------------------------------


def compile_filter(expr):
    """Compile a lane filter string like 'age_days <= 30' to a row predicate."""
    if not expr:
        return lambda r: True
    m = _FILTER_RE.match(expr)
    if not m:
        raise ValueError(f"bad lane filter: {expr!r}")
    field, op, val = m.group(1), m.group(2), float(m.group(3))
    key, fn = _FIELD_MAP[field], _OPS[op]
    return lambda r: fn(r[key], val)


def _rank_key(rank_by):
    keys = [_FIELD_MAP[k] for k in rank_by]
    return lambda r: tuple(r[k] for k in keys) + (r["reactions_total"], r["number"])


def _spill_key(source):
    if source == "emerging-rank":
        return lambda r: (r["rate_score"], r["reactions_total"], r["number"])
    if source == "global-score":
        return lambda r: (r["score"], r["reactions_total"], r["number"])
    raise ValueError(f"unknown spill source: {source}")


def select_lanes(scored: list[dict], sel_cfg: dict):
    """Fill each lane in order (global dedup on number AND cluster_id), then
    refill unfilled slots via spill_order. Tags rows with selection_lane.

    Returns (selected_in_order, lane_fill, spill_fill).
    """
    top_n = sel_cfg["top_n"]
    eligible = [r for r in scored if r["eligible"]]
    claimed_num: set = set()
    claimed_clu: set = set()
    selected: list[dict] = []
    lane_fill: dict = {}

    def _take(candidates, slots, tag):
        taken = 0
        for r in candidates:
            if (slots is not None and taken >= slots) or len(selected) >= top_n:
                break
            if r["number"] in claimed_num or r["cluster_id"] in claimed_clu:
                continue
            claimed_num.add(r["number"])
            claimed_clu.add(r["cluster_id"])
            r["selection_lane"] = tag
            selected.append(r)
            taken += 1
        return taken

    for lane in sel_cfg["lanes"]:
        pred = compile_filter(lane.get("filter"))
        cand = [r for r in eligible if pred(r)]
        cand.sort(key=_rank_key(lane["rank_by"]), reverse=True)
        lane_fill[lane["name"]] = _take(cand, lane["slots"], lane["name"])

    spill_fill: dict = {}
    for source in sel_cfg["spill_order"]:
        if len(selected) >= top_n:
            spill_fill[source] = 0
            continue
        cand = sorted(eligible, key=_spill_key(source), reverse=True)
        spill_fill[source] = _take(cand, None, f"spill:{source}")

    return selected, lane_fill, spill_fill


# --- CSV emission ----------------------------------------------------------


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
                r["reactions_plus1"],
                r["comments"],
                r["maintainer_authored"],
                ";".join(labels.get(r["number"], [])),
                r["cluster_id"],
                r["cluster_size"],
                ";".join(str(m) for m in member_nums),
                _fmt(r["score"]),
                _fmt(r["rate_score"]),
                r.get("selection_lane") or "",
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
    """Score every issue, run the lane portfolio, persist, emit CSVs. Returns run_id."""
    snapshot_ts = db.get_meta(conn, "snapshot_ts")
    if snapshot_ts is None:
        raise RuntimeError("no snapshot_ts in meta; run ingest first")

    weights = cfg["weights"]
    run_id = config_mod.derive_run_id(snapshot_ts, cfg)

    scored = _load_scored_rows(conn, weights)

    # Global score-rank among eligible (ranked_pool order + rank column).
    eligible = [r for r in scored if r["eligible"]]
    eligible.sort(
        key=lambda r: (r["score"], r["reactions_total"], r["number"]), reverse=True
    )
    for i, r in enumerate(eligible, start=1):
        r["rank"] = i

    # Lane portfolio selection (tags rows with selection_lane in place).
    selected, lane_fill, spill_fill = select_lanes(scored, cfg["selection"])
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
    score_rows = []
    for r in scored:
        score_rows.append((
            run_id, r["number"], r["score"],
            r["c_reactions"], r["c_comments"], r["c_velocity"],
            r["c_severity"], r["c_demand"], r["c_cluster"],
            r.get("rank"),
            1 if r["number"] in selected_numbers else 0,
            r.get("selection_lane"),
        ))
    conn.executemany(
        "INSERT INTO scores (run_id, number, score, c_reactions, c_comments, "
        "c_velocity, c_severity, c_demand, c_cluster, rank, selected, "
        "selection_lane) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        score_rows,
    )
    conn.commit()

    _write_csv(os.path.join(out_dir, "top_1000.csv"), selected, conn, run_id,
               snapshot_ts, display_rank=True)
    _write_csv(os.path.join(out_dir, "ranked_pool.csv"), eligible, conn, run_id,
               snapshot_ts, display_rank=False)

    fills = ", ".join(f"{k}={v}" for k, v in lane_fill.items())
    spills = sum(spill_fill.values())
    print(f"[score] lanes: {fills}; spill={spills} (total {len(selected)})")
    return run_id
