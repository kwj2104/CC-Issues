"""M2 — text preparation and per-issue feature computation."""

from __future__ import annotations

import math
import re
import sqlite3

from . import db

# --- Text prep -------------------------------------------------------------

_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_IMAGE_MD_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")

# Issue-template headers whose lines are boilerplate, not signal.
_TEMPLATE_STOPLIST = {
    "environment",
    "what happened",
    "steps to reproduce",
    "expected behavior",
    "preflight checklist",
    "version",
    "platform",
}
_HEADER_RE = re.compile(r"^#{1,6}\s*(.+?)\s*$")


def _strip_markup(text: str) -> str:
    """Remove fenced code, HTML comments, image markdown, and URLs."""
    text = _CODE_FENCE_RE.sub(" ", text)
    text = _HTML_COMMENT_RE.sub(" ", text)
    text = _IMAGE_MD_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    return text


def _drop_template_headers(text: str) -> str:
    """Drop lines that are issue-template section headers from the stoplist."""
    kept = []
    for line in text.splitlines():
        m = _HEADER_RE.match(line.strip())
        if m and m.group(1).strip().lower() in _TEMPLATE_STOPLIST:
            continue
        kept.append(line)
    return "\n".join(kept)


def _clean(text: str) -> str:
    return _drop_template_headers(_strip_markup(text or ""))


def _collapse(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def prep_text(title: str, body: str, body_lead_chars: int) -> dict:
    """Shared text prep for severity regex and clustering.

    Returns clean_text, body_lead, severity_text, and cluster_doc (title
    doubled to up-weight it), per plan sections 5 and 6.
    """
    title_clean = _collapse(_clean(title))
    clean_body = _collapse(_clean(body))          # body only, untruncated
    body_lead = clean_body[:body_lead_chars]
    clean_text = _collapse(title_clean + " " + body_lead)
    severity_text = title_clean + " " + body_lead
    cluster_doc = title_clean + " " + title_clean + " " + body_lead
    return {
        "title_clean": title_clean,
        "clean_body": clean_body,
        "body_lead": body_lead,
        "clean_text": clean_text,
        "severity_text": severity_text,
        "cluster_doc": cluster_doc,
    }


# --- Feature formulas ------------------------------------------------------


def age_days(snapshot_ts_epoch: float, created_epoch: float) -> float:
    return max(1.0, (snapshot_ts_epoch - created_epoch) / 86400.0)


def f_reactions(reactions_total: int) -> float:
    return math.log2(1 + reactions_total)


def f_comments(comments: int) -> float:
    return math.log2(1 + comments)


def f_velocity(reactions_total: int, comments: int, age: float) -> float:
    return math.log2(1 + 30.0 * (reactions_total + comments) / age)


def f_severity(severity_text: str, labels: set, sev_cfg: dict) -> float:
    total = 0.0
    for label, weight in sev_cfg["label_weights"].items():
        if label in labels:
            total += weight
    regex = re.compile(sev_cfg["regex_bank"], re.IGNORECASE)
    if regex.search(severity_text):
        total += sev_cfg["regex_bonus"]  # at most once, regardless of matches
    return min(sev_cfg["cap"], total)


def f_demand(reactions_total: int, labels: set, demand_cfg: dict) -> float:
    has_label = any(lbl in labels for lbl in demand_cfg["labels"])
    if has_label and reactions_total >= demand_cfg["min_reactions"]:
        return math.log2(reactions_total)
    return 0.0


def compute_rate_score(reactions_total: int, comments: int, age: float,
                       f_severity: float, cluster_size: int, weights: dict) -> float:
    """rev 6: age-corrected score. Engagement enters as monthly *rates*, not raw
    totals (raw reactions skew mechanically with age), and there is no demand term.

    rate_score = w.reactions*log2(1 + 30*reactions/age)
               + w.comments*log2(1 + 30*comments/age)
               + w.severity*severity
               + w.cluster*log2(cluster_size or 1)
    """
    react_rate = math.log2(1 + 30.0 * reactions_total / age)
    comm_rate = math.log2(1 + 30.0 * comments / age)
    return (
        weights["reactions"] * react_rate
        + weights["comments"] * comm_rate
        + weights["severity"] * f_severity
        + weights["cluster"] * math.log2(cluster_size or 1)
    )


_MAINTAINER_ASSOCIATIONS = {"OWNER", "MEMBER", "COLLABORATOR"}


def compute_is_junk(clean_body: str, reactions_total: int, comments: int,
                    age: float, junk_cfg: dict) -> int:
    """Abandoned empty report: ALL junk conditions must hold."""
    return int(
        len(clean_body) <= junk_cfg["max_clean_body_chars"]
        and reactions_total <= junk_cfg["max_reactions"]
        and comments <= junk_cfg["max_comments"]
        and age >= junk_cfg["min_age_days"]
    )


def compute_maintainer_authored(author_association) -> int:
    """Flag only — never affects eligibility or score."""
    return int((author_association or "").upper() in _MAINTAINER_ASSOCIATIONS)


# --- Driver ----------------------------------------------------------------


def _parse_iso_epoch(ts: str) -> float:
    from datetime import datetime, timezone

    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def run_features(conn: sqlite3.Connection, cfg: dict) -> int:
    """Compute features for every ingested issue. Returns row count."""
    snapshot_ts = db.get_meta(conn, "snapshot_ts")
    if snapshot_ts is None:
        raise RuntimeError("no snapshot_ts in meta; run ingest first")
    snap_epoch = _parse_iso_epoch(snapshot_ts)

    body_lead_chars = cfg["clustering"]["body_lead_chars"]
    exclude = set(cfg["selection"]["exclude_labels"])
    exclude_lock = set(cfg["selection"].get("exclude_lock_reasons", []))
    junk_cfg = cfg["selection"]["junk_filter"]
    stale_rescue_min = cfg["selection"].get("stale_rescue_min_reactions")

    # Preload labels per issue.
    labels_by_number: dict[int, set] = {}
    for row in conn.execute("SELECT number, label FROM issue_labels"):
        labels_by_number.setdefault(row["number"], set()).add(row["label"])

    conn.execute("DELETE FROM features")
    rows = conn.execute(
        "SELECT number, title, body, created_at, updated_at, comments, "
        "reactions_total, author_association, active_lock_reason FROM issues"
    ).fetchall()

    inserts = []
    for r in rows:
        number = r["number"]
        labels = labels_by_number.get(number, set())
        prepped = prep_text(r["title"], r["body"], body_lead_chars)
        created_epoch = _parse_iso_epoch(r["created_at"])
        age = age_days(snap_epoch, created_epoch)

        fr = f_reactions(r["reactions_total"])
        fc = f_comments(r["comments"])
        fv = f_velocity(r["reactions_total"], r["comments"], age)
        fs = f_severity(prepped["severity_text"], labels, cfg["severity"])
        fd = f_demand(r["reactions_total"], labels, cfg["demand"])
        is_junk = compute_is_junk(
            prepped["clean_body"], r["reactions_total"], r["comments"], age, junk_cfg
        )
        maintainer = compute_maintainer_authored(r["author_association"])

        # rev 3: no calendar window — every open issue is in the baseline pool.
        # Maintainers' own lifecycle labels (via exclude_labels) define staleness;
        # window_variants are explored only inside the sensitivity report.
        in_pool = 1
        # rev 6 stale rescue: `stale` stops excluding at >= N reactions,
        # mirroring the repo sweep's own STALE_UPVOTE_THRESHOLD exemption.
        bad_labels = labels & exclude
        if (stale_rescue_min is not None and "stale" in bad_labels
                and r["reactions_total"] >= stale_rescue_min):
            bad_labels = bad_labels - {"stale"}
        lock_reason = (r["active_lock_reason"] or "").lower()
        eligible = int(
            bool(in_pool)
            and not bad_labels
            and lock_reason not in exclude_lock
            and not is_junk
        )

        # rate_score needs cluster_size (known only after clustering); it is
        # filled by compute_rate_scores() and stored 0.0 as a placeholder here.
        inserts.append(
            (number, age, fr, fc, fv, fs, fd, 0.0, is_junk, maintainer,
             in_pool, eligible)
        )

    conn.executemany(
        "INSERT INTO features (number, age_days, f_reactions, f_comments, "
        "f_velocity, f_severity, f_demand, rate_score, is_junk, "
        "maintainer_authored, in_pool, eligible) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        inserts,
    )
    conn.commit()
    return len(inserts)


def compute_rate_scores(conn: sqlite3.Connection, cfg: dict) -> int:
    """Fill features.rate_score once clusters exist. Returns row count."""
    weights = cfg["weights"]
    rows = conn.execute(
        "SELECT f.number, i.reactions_total, i.comments, f.age_days, "
        "f.f_severity, c.cluster_size "
        "FROM features f JOIN issues i ON i.number = f.number "
        "JOIN clusters c ON c.number = f.number"
    ).fetchall()
    updates = [
        (
            compute_rate_score(r["reactions_total"], r["comments"], r["age_days"],
                               r["f_severity"], r["cluster_size"], weights),
            r["number"],
        )
        for r in rows
    ]
    conn.executemany("UPDATE features SET rate_score = ? WHERE number = ?", updates)
    conn.commit()
    return len(updates)
