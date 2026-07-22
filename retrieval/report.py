"""M4 — diagnostic reports: composition, sensitivity, top20, cluster QC."""

from __future__ import annotations

import csv
import os
import random
import sqlite3
import statistics

from . import config as config_mod
from . import db
from .features import _parse_iso_epoch
from .score import (_load_scored_rows, _rank_and_select, compile_filter,
                    select_lanes)

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


def _exclusion_waterfall(pool, labels_by, cfg, total_open, reactions_by, stale_min):
    """Sequential funnel; returns (steps, stale_rescued).

    Each step is (name, removed, remaining), monotone non-increasing. The
    `stale` step removes only issues NOT rescued by the stale-upvote exemption
    (label `stale` stops excluding at >= stale_min reactions); the rescued count
    is returned separately. Final remaining == the features-table eligible count.
    """
    exclude_labels = cfg["selection"]["exclude_labels"]
    exclude_lock = set(cfg["selection"].get("exclude_lock_reasons", []))

    remaining = {r["number"] for r in pool}
    by_number = {r["number"]: r for r in pool}
    steps = [
        ("open snapshot", total_open - total_open, total_open),
        ("in_pool", total_open - len(remaining), len(remaining)),
    ]
    stale_rescued = 0

    for label in exclude_labels:
        if label == "stale" and stale_min is not None:
            has_stale = {n for n in remaining if "stale" in labels_by.get(n, set())}
            rescued = {n for n in has_stale if reactions_by.get(n, 0) >= stale_min}
            stale_rescued = len(rescued)
            removed = has_stale - rescued
        else:
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
    return steps, stale_rescued


# --- lane diagnostics (rev 6) ----------------------------------------------


def _eng_dominated(r) -> bool:
    """True if raw engagement (c_reactions+c_comments) is the largest score part."""
    comps = {
        "engagement": r["c_reactions"] + r["c_comments"],
        "velocity": r["c_velocity"],
        "severity": r["c_severity"],
        "demand": r["c_demand"],
        "cluster": r["c_cluster"],
    }
    return max(comps, key=comps.get) == "engagement"


def _lane_diagnostics(conn, cfg, run_id):
    """Per-lane fills/medians/eng-dominance + a lane-filter overlap matrix."""
    sel = conn.execute(
        "SELECT s.number, s.selection_lane, s.c_reactions, s.c_comments, "
        "s.c_velocity, s.c_severity, s.c_demand, s.c_cluster, "
        "f.age_days, f.rate_score, f.f_severity, c.cluster_size, i.reactions_total "
        "FROM scores s JOIN features f ON f.number = s.number "
        "JOIN clusters c ON c.number = s.number "
        "JOIN issues i ON i.number = s.number "
        "WHERE s.run_id = ? AND s.selected = 1",
        (run_id,),
    ).fetchall()
    rows = [dict(r) for r in sel]

    lane_names = [lane["name"] for lane in cfg["selection"]["lanes"]]
    spill_names = sorted({
        r["selection_lane"] for r in rows
        if r["selection_lane"] and r["selection_lane"].startswith("spill:")
    })
    order = lane_names + spill_names
    by_lane = {name: [r for r in rows if r["selection_lane"] == name]
               for name in order}

    md = ["", "## Per-lane composition", "",
          "| lane | fill | median age_days | median rate_score | median severity "
          "| eng-dominated % |",
          "|---|---|---|---|---|---|"]
    csv_rows = []
    flags = []
    for name in order:
        g = by_lane.get(name, [])
        if not g:
            md.append(f"| {name} | 0 | - | - | - | - |")
            csv_rows.append(("lane_fill", name, 0))
            continue
        ma = statistics.median([x["age_days"] for x in g])
        mr = statistics.median([x["rate_score"] for x in g])
        ms = statistics.median([x["f_severity"] for x in g])
        eng_frac = sum(1 for x in g if _eng_dominated(x)) / len(g)
        md.append(f"| {name} | {len(g)} | {ma:.1f} | {mr:.2f} | {ms:.2f} | "
                  f"{eng_frac:.0%} |")
        csv_rows.append(("lane_fill", name, len(g)))
        csv_rows.append(("lane_eng_dominated_frac", name, f"{eng_frac:.4f}"))
        # big-bets ranks by raw score, so its engagement dominance is by design;
        # a rate/severity/cluster lane going >90% engagement is the real signal.
        if name != "big-bets" and eng_frac > 0.9:
            flags.append(
                f"> **FLAG:** lane `{name}` is {eng_frac:.0%} engagement-dominated "
                f"despite ranking by rate/severity/cluster — age-correction may not "
                f"be separating it from raw engagement."
            )
    if flags:
        md += ["", *flags]

    lanes = cfg["selection"]["lanes"]
    preds = {lane["name"]: compile_filter(lane.get("filter")) for lane in lanes}
    header = " | ".join(lane["name"] for lane in lanes)
    md += ["", "## Lane overlap matrix",
           "Rows = assigned lane; columns = how many of those rows also pass each "
           "lane's filter.", "",
           f"| assigned \\\\ passes | {header} |",
           "|" + "---|" * (len(lanes) + 1)]
    for name in order:
        g = by_lane.get(name, [])
        cells = [str(sum(1 for r in g if preds[lane["name"]](r))) for lane in lanes]
        md.append(f"| {name} | " + " | ".join(cells) + " |")
    return md, csv_rows


# --- composition -----------------------------------------------------------


def _composition(conn, cfg, run_id, out_dir) -> None:
    labels_by = _labels_by_number(conn)
    demand_labels = set(cfg["demand"]["labels"])
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
    reactions_by = {r["number"]: r["reactions_total"] for r in pool}
    stale_min = cfg["selection"].get("stale_rescue_min_reactions")
    waterfall, stale_rescued = _exclusion_waterfall(
        pool, labels_by, cfg, total_open, reactions_by, stale_min
    )
    # rev 3: no window -> pool is every open issue; carve-out no longer exists.
    maintainer_in_top = conn.execute(
        "SELECT COUNT(*) AS n FROM scores s JOIN features f ON f.number = s.number "
        "WHERE s.run_id = ? AND s.selected = 1 AND f.maintainer_authored = 1",
        (run_id,),
    ).fetchone()["n"]

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

    # Engagement deciles over the pool, where do the selected fall?
    # rev 4.1: rank-based percentiles (average-rank of the tie group) instead of
    # value cut-points, so heavy ties (e.g. engagement 0) don't silently collapse
    # deciles. Equal engagement -> equal decile; the note below quantifies it.
    import bisect

    eng = sorted((r["reactions_total"] + r["comments"]) for r in pool)
    n_pool = len(eng)

    def decile(v):
        lo = bisect.bisect_left(eng, v)
        hi = bisect.bisect_right(eng, v)
        avg_rank = (lo + hi - 1) / 2.0        # 0-based average rank of the ties
        return min(9, int((avg_rank + 0.5) / n_pool * 10)) if n_pool else 0

    sel_deciles: dict[int, int] = {}
    for r in selected:
        d = decile(r["reactions_total"] + r["comments"])
        sel_deciles[d] = sel_deciles.get(d, 0) + 1

    # Quantify the tie-collapse: modal engagement value and where it maps.
    zero_share = (eng.count(0) / n_pool) if n_pool else 0.0
    modal_decile = decile(0)
    decile_note = (
        f"Rank-based deciles (ties share a decile). **{zero_share:.0%}** of the pool "
        f"has engagement 0 -> all map to decile {modal_decile}, so lower deciles are "
        f"empty by construction."
    )

    age_keys = ["<=7d", "8-30d", "31-90d", ">90d"]
    class_keys = ["bug", "enhancement", "other"]

    lines = [
        "# Composition report",
        "",
        f"- run_id: `{run_id}`",
        f"- snapshot_ts: `{snapshot_ts}`",
        f"- pool size (in_pool): **{len(pool)}**",
        f"- eligible (ranked pool): **{len(eligible)}**",
        f"- selected (top_n): **{len(selected)}**",
        f"- maintainer-authored rows inside top_n: **{maintainer_in_top}** "
        f"(flag only; not excluded)",
        "",
        "## Exclusion waterfall",
        "",
        f"- **stale rescued** (kept despite `stale`, >= {stale_min} reactions): "
        f"**{stale_rescued}**",
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
              decile_note, "",
              "| decile | selected |", "|---|---|"]
    for d in range(10):
        lines.append(f"| {d} | {sel_deciles.get(d, 0)} |")

    # rev 6: lane diagnostics + Jaccard vs the old single-score head.
    lane_md, lane_csv = _lane_diagnostics(conn, cfg, run_id)
    lines += lane_md

    scored = _load_scored_rows(conn, cfg["weights"])
    _, old_selected = _rank_and_select(scored, cfg["selection"]["top_n"])
    old_head = {r["number"] for r in old_selected}
    jac = _jaccard(old_head, selected_numbers)
    lines += ["", "## Head vs single-score baseline", "",
              f"- Jaccard(lane head, single-score head) = **{jac:.3f}** "
              f"({len(old_head & selected_numbers)} shared of {len(selected_numbers)})"]

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
        w.writerow(["maintainer_in_top", "", maintainer_in_top])
        w.writerow(["stale_rescued", "", stale_rescued])
        w.writerow(["jaccard_vs_single_score", "", f"{jac:.4f}"])
        for name, removed, remaining in waterfall:
            w.writerow(["waterfall_remaining", name, remaining])
            w.writerow(["waterfall_removed", name, removed])
        for metric, key, value in lane_csv:
            w.writerow([metric, key, value])
        for k in age_keys:
            w.writerow(["age_pool", k, pb.get(k, 0)])
            w.writerow(["age_selected", k, sb.get(k, 0)])
        for k in class_keys:
            w.writerow(["class_pool", k, pc.get(k, 0)])
            w.writerow(["class_selected", k, sc.get(k, 0)])
        for name, n in top_areas:
            w.writerow(["area_label", name, n])


# --- sensitivity -----------------------------------------------------------


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _lane_head(scored, sel_cfg) -> set:
    """Numbers of the full lane portfolio (lanes + spill + cluster dedup)."""
    selected, _, _ = select_lanes(scored, sel_cfg)
    return {r["number"] for r in selected}


def select_within_window(scored, snap_epoch, window_days, sel_cfg):
    """Full lane selection restricted to issues created within `window_days`."""
    subset = [
        r for r in scored
        if (snap_epoch - _parse_iso_epoch(r["created_at"])) / 86400.0 <= window_days
    ]
    selected, _, _ = select_lanes(subset, sel_cfg)
    return selected


def _sensitivity(conn, cfg, run_id, out_dir) -> None:
    weights = cfg["weights"]
    sel_cfg = cfg["selection"]
    pert = cfg["sensitivity"]["perturbation"]

    # rev 4.1: every variant re-runs the FULL lane portfolio and compares to the
    # current run's lane head (not the retired single-score head).
    base_scored = _load_scored_rows(conn, weights)
    base_set = _lane_head(base_scored, sel_cfg)

    # Sanity: an unperturbed re-run must reproduce the baseline head exactly.
    unpert = _lane_head(_load_scored_rows(conn, weights), sel_cfg)
    sanity = _jaccard(base_set, unpert)
    if sanity != 1.0:
        raise RuntimeError(
            f"sensitivity harness broken: unperturbed Jaccard {sanity} != 1.0"
        )

    # (a) weight perturbation
    rows = []
    for key in config_mod.WEIGHT_KEYS:
        for factor, label in ((1 - pert, "x0.5"), (1 + pert, "x1.5")):
            w = dict(weights)
            w[key] = weights[key] * factor
            head = _lane_head(_load_scored_rows(conn, w), sel_cfg)
            rows.append((key, label, _jaccard(base_set, head)))

    overlaps = [j for _, _, j in rows]
    headline_min = min(overlaps) if overlaps else 1.0
    headline_mean = sum(overlaps) / len(overlaps) if overlaps else 1.0

    # (b) creation-window variants: does any calendar window change the pick?
    snap_epoch = _parse_iso_epoch(db.get_meta(conn, "snapshot_ts"))
    win_rows = []
    for w_days in cfg["sensitivity"].get("window_variants", []):
        sel = select_within_window(base_scored, snap_epoch, w_days, sel_cfg)
        win_rows.append(
            (w_days, len(sel), _jaccard(base_set, {r["number"] for r in sel}))
        )

    lines = [
        "# Sensitivity report",
        "",
        f"- run_id: `{run_id}`",
        f"- baseline = the current run's **lane head** ({len(base_set)} rows); "
        f"every variant re-runs the full lane portfolio",
        f"- unperturbed sanity: Jaccard vs baseline = **{sanity:.3f}** (must be 1.000)",
        "",
        "## (a) Weight perturbation",
        "",
        f"- perturbation: +/-{pert:.0%} per weight (12 variants)",
        f"- **min overlap: {headline_min:.3f}**, mean overlap: {headline_mean:.3f}",
        "",
        "| weight | variant | Jaccard vs baseline lane head |",
        "|---|---|---|",
    ]
    for key, label, j in rows:
        lines.append(f"| {key} | {label} | {j:.3f} |")

    lines += [
        "",
        "## (b) Creation-window variants",
        "",
        "Baseline imposes **no** calendar window; each variant restricts the lane "
        "portfolio to issues created within N days of the snapshot. High overlap = "
        "the window choice would not have changed the selection.",
        "",
        "| window (days) | selected | Jaccard vs no-window baseline |",
        "|---|---|---|",
    ]
    for w_days, n_sel, j in win_rows:
        lines.append(f"| {w_days} | {n_sel} | {j:.3f} |")

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
