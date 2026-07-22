"""M4 — diagnostic reports: composition, sensitivity, top20, cluster QC."""

from __future__ import annotations

import csv
import os
import random
import sqlite3

from . import config as config_mod
from . import db
from .score import _load_scored_rows, _rank_and_select

REPORTS_DIR = "reports"


# --- shared loaders --------------------------------------------------------


def _labels_by_number(conn) -> dict:
    out: dict[int, set] = {}
    for row in conn.execute("SELECT number, label FROM issue_labels"):
        out.setdefault(row["number"], set()).add(row["label"])
    return out


def _age_bucket(age_days: float) -> str:
    if age_days <= 7:
        return "<=7d"
    if age_days <= 30:
        return "8-30d"
    if age_days <= 90:
        return "31-90d"
    return ">90d"


def _classify(labels: set, demand_labels: set) -> str:
    if any("bug" in l for l in labels):
        return "bug"
    if any(l in demand_labels or "enhancement" in l or "feature" in l for l in labels):
        return "enhancement"
    return "other"


def _dist(counter: dict, keys) -> str:
    total = sum(counter.get(k, 0) for k in keys) or 1
    parts = []
    for k in keys:
        n = counter.get(k, 0)
        parts.append(f"| {k} | {n} | {n / total:.1%} |")
    return "\n".join(parts)


# --- exclusion waterfall ---------------------------------------------------


def _exclusion_waterfall(pool, labels_by, cfg, total_open) -> list[tuple]:
    """Sequential funnel: (step, removed, remaining), each step <= previous.

    Starts from the in_pool set and removes, in order, each excluded label,
    then excluded lock reasons, then junk. Final remaining == eligible count.
    """
    exclude_labels = cfg["selection"]["exclude_labels"]
    exclude_lock = set(cfg["selection"].get("exclude_lock_reasons", []))

    remaining = {r["number"] for r in pool}
    by_number = {r["number"]: r for r in pool}
    steps = [
        ("open snapshot", total_open - total_open, total_open),
        ("in_pool", total_open - len(remaining), len(remaining)),
    ]

    for label in exclude_labels:
        removed = {n for n in remaining if label in labels_by.get(n, set())}
        remaining -= removed
        steps.append((f"exclude label: {label}", len(removed), len(remaining)))

    removed = {
        n for n in remaining
        if (by_number[n]["active_lock_reason"] or "").lower() in exclude_lock
    }
    remaining -= removed
    steps.append(("exclude lock reasons", len(removed), len(remaining)))

    removed = {n for n in remaining if by_number[n]["is_junk"]}
    remaining -= removed
    steps.append(("junk filter", len(removed), len(remaining)))

    steps.append(("eligible", 0, len(remaining)))
    return steps


# --- composition -----------------------------------------------------------


def _composition(conn, cfg, run_id, out_dir) -> None:
    labels_by = _labels_by_number(conn)
    demand_labels = set(cfg["demand"]["labels"])
    created_days = cfg["window"]["created_days"]
    snapshot_ts = db.get_meta(conn, "snapshot_ts")

    pool = conn.execute(
        """
        SELECT i.number, i.created_at, i.reactions_total, i.comments,
               i.active_lock_reason, f.age_days, f.in_pool, f.eligible, f.is_junk
        FROM issues i JOIN features f ON f.number = i.number
        WHERE f.in_pool = 1
        """
    ).fetchall()
    eligible = [r for r in pool if r["eligible"]]
    selected_numbers = {
        row["number"]
        for row in conn.execute(
            "SELECT number FROM scores WHERE run_id = ? AND selected = 1", (run_id,)
        )
    }
    selected = [r for r in eligible if r["number"] in selected_numbers]

    total_open = conn.execute("SELECT COUNT(*) AS n FROM issues").fetchone()["n"]
    waterfall = _exclusion_waterfall(pool, labels_by, cfg, total_open)
    maintainer_in_top = conn.execute(
        "SELECT COUNT(*) AS n FROM scores s JOIN features f ON f.number = s.number "
        "WHERE s.run_id = ? AND s.selected = 1 AND f.maintainer_authored = 1",
        (run_id,),
    ).fetchone()["n"]

    # carve-out share: in pool but created outside the created_days window.
    from .features import _parse_iso_epoch

    snap_epoch = _parse_iso_epoch(snapshot_ts)

    def carveout_only(r):
        age_created = (snap_epoch - _parse_iso_epoch(r["created_at"])) / 86400.0
        return age_created > created_days

    pool_carveout = sum(1 for r in pool if carveout_only(r))
    selected_carveout = sum(1 for r in selected if carveout_only(r))

    def bucket_counts(rows):
        c: dict[str, int] = {}
        for r in rows:
            c[_age_bucket(r["age_days"])] = c.get(_age_bucket(r["age_days"]), 0) + 1
        return c

    def class_counts(rows):
        c: dict[str, int] = {}
        for r in rows:
            k = _classify(labels_by.get(r["number"], set()), demand_labels)
            c[k] = c.get(k, 0) + 1
        return c

    area_counts: dict[str, int] = {}
    for r in selected:
        for l in labels_by.get(r["number"], set()):
            if l.startswith("area:"):
                area_counts[l] = area_counts.get(l, 0) + 1
    top_areas = sorted(area_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:15]

    # engagement deciles over the pool; where do the selected fall?
    eng = sorted((r["reactions_total"] + r["comments"]) for r in pool)
    bounds = [eng[min(len(eng) - 1, (len(eng) * d) // 10)] for d in range(1, 10)] if eng else []

    def decile(v):
        d = 0
        for b in bounds:
            if v > b:
                d += 1
        return d

    sel_deciles: dict[int, int] = {}
    for r in selected:
        sel_deciles[decile(r["reactions_total"] + r["comments"])] = (
            sel_deciles.get(decile(r["reactions_total"] + r["comments"]), 0) + 1
        )

    age_keys = ["<=7d", "8-30d", "31-90d", ">90d"]
    class_keys = ["bug", "enhancement", "other"]

    pure_engagement_note = ""
    # crude head-purity check: fraction of top-100 whose top component is engagement
    head = conn.execute(
        """
        SELECT c_reactions, c_comments, c_velocity, c_severity, c_demand, c_cluster
        FROM scores WHERE run_id = ? AND selected = 1
        ORDER BY rank ASC LIMIT 100
        """,
        (run_id,),
    ).fetchall()
    if head:
        eng_dom = 0
        for h in head:
            comps = {
                "engagement": h["c_reactions"] + h["c_comments"],
                "velocity": h["c_velocity"],
                "severity": h["c_severity"],
                "demand": h["c_demand"],
                "cluster": h["c_cluster"],
            }
            if max(comps, key=comps.get) == "engagement":
                eng_dom += 1
        frac = eng_dom / len(head)
        if frac > 0.9:
            pure_engagement_note = (
                f"\n> **FLAG:** {frac:.0%} of the top 100 are engagement-dominated "
                f"picks; cluster/severity/velocity are barely moving the head.\n"
            )

    lines = [
        "# Composition report",
        "",
        f"- run_id: `{run_id}`",
        f"- snapshot_ts: `{snapshot_ts}`",
        f"- pool size (in_pool): **{len(pool)}**",
        f"- eligible (ranked pool): **{len(eligible)}**",
        f"- selected (top_n): **{len(selected)}**",
        f"- carve-out share of pool: **{pool_carveout}** "
        f"({pool_carveout / (len(pool) or 1):.1%})",
        f"- selected admitted **only** via carve-out: **{selected_carveout}**",
        f"- maintainer-authored rows inside top_n: **{maintainer_in_top}** "
        f"(flag only; not excluded)",
        pure_engagement_note,
        "## Exclusion waterfall",
        "",
        "| step | removed | remaining |",
        "|---|---|---|",
        *[f"| {name} | {removed} | {remaining} |"
          for name, removed, remaining in waterfall],
        "",
        "## Age buckets",
        "",
        "| bucket | full-pool | share || top_n | share |",
        "|---|---|---|---|---|---|",
    ]
    pb, sb = bucket_counts(pool), bucket_counts(selected)
    pt = sum(pb.values()) or 1
    st = sum(sb.values()) or 1
    for k in age_keys:
        lines.append(
            f"| {k} | {pb.get(k,0)} | {pb.get(k,0)/pt:.1%} || "
            f"{sb.get(k,0)} | {sb.get(k,0)/st:.1%} |"
        )

    lines += ["", "## Label mix (bug / enhancement / other)", "",
              "| class | full-pool | share || top_n | share |",
              "|---|---|---|---|---|---|"]
    pc, sc = class_counts(pool), class_counts(selected)
    pct = sum(pc.values()) or 1
    sct = sum(sc.values()) or 1
    for k in class_keys:
        lines.append(
            f"| {k} | {pc.get(k,0)} | {pc.get(k,0)/pct:.1%} || "
            f"{sc.get(k,0)} | {sc.get(k,0)/sct:.1%} |"
        )

    lines += ["", "## Top-15 `area:*` labels among selected", "",
              "| label | count |", "|---|---|"]
    for name, n in top_areas:
        lines.append(f"| {name} | {n} |")

    lines += ["", "## Engagement deciles of selected (0=lowest, 9=highest)", "",
              "| decile | selected |", "|---|---|"]
    for d in range(10):
        lines.append(f"| {d} | {sel_deciles.get(d, 0)} |")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "composition.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    # machine-readable twin
    with open(os.path.join(out_dir, "composition.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["metric", "key", "value"])
        w.writerow(["pool_size", "", len(pool)])
        w.writerow(["eligible", "", len(eligible)])
        w.writerow(["selected", "", len(selected)])
        w.writerow(["pool_carveout", "", pool_carveout])
        w.writerow(["selected_carveout", "", selected_carveout])
        w.writerow(["maintainer_in_top", "", maintainer_in_top])
        for name, removed, remaining in waterfall:
            w.writerow(["waterfall_remaining", name, remaining])
            w.writerow(["waterfall_removed", name, removed])
        for k in age_keys:
            w.writerow(["age_pool", k, pb.get(k, 0)])
            w.writerow(["age_selected", k, sb.get(k, 0)])
        for k in class_keys:
            w.writerow(["class_pool", k, pc.get(k, 0)])
            w.writerow(["class_selected", k, sc.get(k, 0)])
        for name, n in top_areas:
            w.writerow(["area_label", name, n])


# --- sensitivity -----------------------------------------------------------


def _sensitivity(conn, cfg, run_id, out_dir) -> None:
    weights = cfg["weights"]
    top_n = cfg["selection"]["top_n"]
    pert = cfg["sensitivity"]["perturbation"]

    base_scored = _load_scored_rows(conn, weights)
    _, base_selected = _rank_and_select(base_scored, top_n)
    base_set = {r["number"] for r in base_selected}

    def jaccard(a: set, b: set) -> float:
        if not a and not b:
            return 1.0
        return len(a & b) / len(a | b)

    rows = []
    for key in config_mod.WEIGHT_KEYS:
        for factor, label in ((1 - pert, "x0.5"), (1 + pert, "x1.5")):
            w = dict(weights)
            w[key] = weights[key] * factor
            scored = _load_scored_rows(conn, w)
            _, selected = _rank_and_select(scored, top_n)
            j = jaccard(base_set, {r["number"] for r in selected})
            rows.append((key, label, j))

    overlaps = [j for _, _, j in rows]
    headline_min = min(overlaps) if overlaps else 1.0
    headline_mean = sum(overlaps) / len(overlaps) if overlaps else 1.0

    lines = [
        "# Sensitivity report",
        "",
        f"- run_id: `{run_id}`",
        f"- perturbation: +/-{pert:.0%} per weight (12 variants)",
        f"- **min overlap: {headline_min:.3f}**, mean overlap: {headline_mean:.3f}",
        "",
        "| weight | variant | Jaccard vs baseline top_n |",
        "|---|---|---|",
    ]
    for key, label, j in rows:
        lines.append(f"| {key} | {label} | {j:.3f} |")
    with open(os.path.join(out_dir, "sensitivity.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# --- top20 -----------------------------------------------------------------


def _top20(conn, cfg, run_id, out_dir) -> None:
    rows = conn.execute(
        """
        SELECT s.rank, i.number, i.title, i.html_url, s.score,
               s.c_reactions, s.c_comments, s.c_velocity, s.c_severity,
               s.c_demand, s.c_cluster
        FROM scores s JOIN issues i ON i.number = s.number
        WHERE s.run_id = ? AND s.selected = 1
        ORDER BY s.rank ASC LIMIT 20
        """,
        (run_id,),
    ).fetchall()
    lines = [
        "# Top-20 preview",
        "",
        "| rank | # | title | score | react | comm | vel | sev | dem | clus |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        title = (r["title"] or "").replace("|", "\\|")[:80]
        lines.append(
            f"| {r['rank']} | [{r['number']}]({r['html_url']}) | {title} | "
            f"{r['score']:.2f} | {r['c_reactions']:.2f} | {r['c_comments']:.2f} | "
            f"{r['c_velocity']:.2f} | {r['c_severity']:.2f} | {r['c_demand']:.2f} | "
            f"{r['c_cluster']:.2f} |"
        )
    with open(os.path.join(out_dir, "top20_preview.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# --- cluster QC ------------------------------------------------------------


def _cluster_qc(conn, cfg, out_dir) -> None:
    seed = cfg["qc"]["seed"]
    sample_n = cfg["qc"]["cluster_sample"]
    multi = conn.execute(
        "SELECT DISTINCT cluster_id FROM clusters WHERE cluster_size >= 2 "
        "ORDER BY cluster_id ASC"
    ).fetchall()
    ids = [r["cluster_id"] for r in multi]
    rng = random.Random(seed)
    chosen = sorted(rng.sample(ids, min(sample_n, len(ids)))) if ids else []

    lines = ["# Cluster QC sample", "",
             f"- seed: {seed}; multi-member clusters: {len(ids)}; "
             f"sampled: {len(chosen)}", ""]
    for cid in chosen:
        members = conn.execute(
            "SELECT c.number, i.title FROM clusters c JOIN issues i "
            "ON i.number = c.number WHERE c.cluster_id = ? ORDER BY c.number ASC",
            (cid,),
        ).fetchall()
        lines.append(f"## cluster {cid} (size {len(members)})")
        for m in members:
            lines.append(f"- #{m['number']}: {m['title']}")
        lines.append("")
    with open(os.path.join(out_dir, "cluster_qc.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def run_report(conn: sqlite3.Connection, cfg: dict,
               out_dir: str = REPORTS_DIR) -> None:
    snapshot_ts = db.get_meta(conn, "snapshot_ts")
    if snapshot_ts is None:
        raise RuntimeError("no snapshot_ts in meta; run ingest first")
    run_id = config_mod.derive_run_id(snapshot_ts, cfg)
    os.makedirs(out_dir, exist_ok=True)
    _composition(conn, cfg, run_id, out_dir)
    _sensitivity(conn, cfg, run_id, out_dir)
    _top20(conn, cfg, run_id, out_dir)
    _cluster_qc(conn, cfg, out_dir)
    print(f"[report] wrote 4 reports to {out_dir}/ for run {run_id}")
