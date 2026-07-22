"""M1 — ingest open issues from the GitHub REST API into the raw + SQLite layers."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import datetime, timezone

from . import TOOL_VERSION, db
from .config import canonical_config_json

API_ROOT = "https://api.github.com"
RAW_ROOT = os.path.join("data", "raw")
PER_PAGE = 100
MAX_RETRIES = 5
# Cursor sentinel for "start of backlog". We OMIT the `since` param on the first
# fetch rather than passing an epoch date: GitHub returns an empty array for
# `since=1970-01-01T00:00:00Z`, but with no `since` and sort=updated&asc it
# returns the genuinely oldest-updated open issues, which is what we want.
INITIAL_CURSOR = None
# Search `total_count` is approximate and runs a few percent above the actual
# open-issue list, so ±1% was unachievable on a full ingest. 3% still flags a
# genuinely broken/truncated pull (the offset-cap bug gave ~19%).
DEFAULT_COUNT_TOLERANCE = 0.03


def count_within_tolerance(row_count: int, api_total: int, tol: float):
    """Return (ok, drift). Trivially ok when the API gave no usable total."""
    if api_total <= 0:
        return True, 0.0
    drift = abs(row_count - api_total) / api_total
    return drift <= tol, drift


# --- Network fetch ---------------------------------------------------------


def _headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _respect_rate_limit(resp) -> None:
    remaining = resp.headers.get("X-RateLimit-Remaining")
    reset = resp.headers.get("X-RateLimit-Reset")
    if remaining is not None and reset is not None and int(remaining) < 10:
        sleep_for = max(0, int(reset) - int(time.time())) + 1
        print(f"[ingest] rate limit low ({remaining}); sleeping {sleep_for}s")
        time.sleep(sleep_for)


def _get(session, url: str, params: dict, token: str):
    """GET with rate-limit, 403/429, and 5xx retry handling."""
    for attempt in range(1, MAX_RETRIES + 1):
        resp = session.get(url, params=params, headers=_headers(token), timeout=60)
        if resp.status_code == 200:
            _respect_rate_limit(resp)
            return resp
        if resp.status_code in (403, 429):
            print(f"[ingest] {resp.status_code} on {url}; sleep 60s "
                  f"(attempt {attempt}/{MAX_RETRIES})")
            time.sleep(60)
            continue
        if 500 <= resp.status_code < 600:
            backoff = min(80, 5 * (2 ** (attempt - 1)))
            print(f"[ingest] {resp.status_code} on {url}; backoff {backoff}s "
                  f"(attempt {attempt}/{MAX_RETRIES})")
            time.sleep(backoff)
            continue
        resp.raise_for_status()
    raise RuntimeError(f"giving up on {url} after {MAX_RETRIES} attempts")


def _fetch_issues(session, repo: str, token: str, raw_path: str,
                  checkpoint_path: str, cursor: str, seen: set) -> str:
    """Cursor-paginate the issues endpoint by `updated_at` ascending.

    GitHub's REST **offset** pagination is capped (page*per_page beyond ~10k
    returns 422), so we page with the `since` cursor instead of `page`, which
    walks the full backlog regardless of size. `since` is inclusive, so the
    boundary rows reappear each step; `seen` (persisted across resume by
    replaying the raw file) drops them, keeping the raw layer append-only and
    every issue written exactly once. Returns the final cursor. Skips PRs.
    """
    url = f"{API_ROOT}/repos/{repo}/issues"
    while True:
        params = {
            "state": "open",
            "sort": "updated",
            "direction": "asc",
            "per_page": PER_PAGE,
        }
        if cursor is not None:  # omit `since` at start (epoch returns [] on GitHub)
            params["since"] = cursor
        resp = _get(session, url, params, token)
        items = resp.json()
        if not items:
            break
        new_count = 0
        max_updated = cursor
        with open(raw_path, "a", encoding="utf-8") as fh:
            for item in items:
                num = item.get("number")
                upd = item.get("updated_at")
                if upd is not None and (max_updated is None or upd > max_updated):
                    max_updated = upd
                if num in seen:
                    continue
                seen.add(num)
                if "pull_request" in item:
                    continue  # endpoint interleaves PRs
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")
                new_count += 1
        with open(checkpoint_path, "w", encoding="utf-8") as fh:
            json.dump(
                {"cursor": max_updated, "count": len(seen), "done": False}, fh
            )
        print(f"[ingest] since {cursor} -> {max_updated}: {len(items)} fetched, "
              f"{new_count} new (total seen {len(seen)})")
        if max_updated == cursor:
            # A full page shares the cursor timestamp; `since` cannot advance
            # without offset paging (capped). Vanishingly rare on real repos.
            print(f"[ingest] WARNING: cursor stalled at {cursor}; stopping.")
            break
        cursor = max_updated
    return cursor


def _validate_count(session, repo: str, token: str) -> int:
    """One search-API call for the authoritative open-issue count."""
    url = f"{API_ROOT}/search/issues"
    # Space-separated qualifiers; requests URL-encodes them correctly.
    resp = session.get(
        url,
        params={"q": f"repo:{repo} type:issue state:open", "per_page": 1},
        headers=_headers(token),
        timeout=60,
    )
    resp.raise_for_status()
    return int(resp.json().get("total_count", 0))


# --- Parsing + load --------------------------------------------------------


def parse_issue(item: dict) -> dict | None:
    """Normalise one raw API object into a DB row dict, or None if malformed."""
    number = item.get("number")
    if number is None or item.get("created_at") is None:
        return None
    reactions = item.get("reactions") or {}
    body = item.get("body")
    if body is None:
        body = ""
    elif isinstance(body, bytes):
        body = body.decode("utf-8", errors="replace")
    labels = []
    for lbl in item.get("labels", []) or []:
        name = lbl.get("name") if isinstance(lbl, dict) else lbl
        if name:
            labels.append(str(name).lower())
    return {
        "number": int(number),
        "title": item.get("title") or "",
        "body": body,
        "state": item.get("state") or "open",
        "created_at": item["created_at"],
        "updated_at": item.get("updated_at") or item["created_at"],
        "comments": int(item.get("comments") or 0),
        "reactions_total": int(reactions.get("total_count") or 0),
        "reactions_plus1": int(reactions.get("+1") or 0),
        "author_association": item.get("author_association"),
        "html_url": item.get("html_url") or "",
        "locked": 1 if item.get("locked") else 0,
        "active_lock_reason": item.get("active_lock_reason"),
        "labels": labels,
    }


def load_into_db(conn: sqlite3.Connection, raw_rows, snapshot_ts: str, repo: str,
                 cfg: dict, api_total_count: int | None) -> dict:
    """Load parsed raw rows into SQLite. Returns {row_count, malformed_count}."""
    db.init_schema(conn)
    conn.execute("DELETE FROM issue_labels")
    conn.execute("DELETE FROM issues")

    row_count = 0
    malformed = 0
    for item in raw_rows:
        if isinstance(item, dict) and "pull_request" in item:
            continue  # endpoint interleaves PRs; not an issue, not malformed
        parsed = parse_issue(item)
        if parsed is None:
            malformed += 1
            continue
        conn.execute(
            "INSERT OR REPLACE INTO issues (number, title, body, state, "
            "created_at, updated_at, comments, reactions_total, reactions_plus1, "
            "author_association, html_url, locked, active_lock_reason) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                parsed["number"], parsed["title"], parsed["body"], parsed["state"],
                parsed["created_at"], parsed["updated_at"], parsed["comments"],
                parsed["reactions_total"], parsed["reactions_plus1"],
                parsed["author_association"], parsed["html_url"],
                parsed["locked"], parsed["active_lock_reason"],
            ),
        )
        conn.execute("DELETE FROM issue_labels WHERE number = ?", (parsed["number"],))
        for label in set(parsed["labels"]):
            conn.execute(
                "INSERT OR IGNORE INTO issue_labels (number, label) VALUES (?, ?)",
                (parsed["number"], label),
            )
        row_count += 1

    db.set_meta(conn, "snapshot_ts", snapshot_ts)
    db.set_meta(conn, "repo", repo)
    db.set_meta(conn, "row_count", row_count)
    db.set_meta(conn, "malformed_count", malformed)
    db.set_meta(conn, "tool_version", TOOL_VERSION)
    db.set_meta(conn, "config_json", canonical_config_json(cfg))
    if api_total_count is not None:
        db.set_meta(conn, "api_total_count", api_total_count)
    conn.commit()
    return {"row_count": row_count, "malformed_count": malformed}


def _read_raw(raw_path: str):
    with open(raw_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def run_ingest(conn: sqlite3.Connection, cfg: dict) -> dict:
    """Full ingest: fetch (resumable), validate count, load into SQLite."""
    import requests

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise SystemExit("GITHUB_TOKEN environment variable is not set")

    repo = cfg["repo"]

    # Resume support: reuse an in-progress snapshot dir if a checkpoint exists.
    snapshot_ts, snapshot_date, start_cursor = _resolve_snapshot(repo)
    raw_dir = os.path.join(RAW_ROOT, snapshot_date)
    os.makedirs(raw_dir, exist_ok=True)
    raw_path = os.path.join(raw_dir, "issues.jsonl")
    checkpoint_path = os.path.join(raw_dir, "checkpoint.json")
    meta_path = os.path.join(raw_dir, "snapshot_meta.json")
    if not os.path.exists(meta_path):
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump({"snapshot_ts": snapshot_ts, "repo": repo}, fh)

    # On resume, replay the already-written raw rows so we never re-append them.
    seen: set = set()
    if os.path.exists(raw_path):
        for row in _read_raw(raw_path):
            num = row.get("number")
            if num is not None:
                seen.add(num)

    session = requests.Session()
    final_cursor = _fetch_issues(
        session, repo, token, raw_path, checkpoint_path, start_cursor, seen
    )
    with open(checkpoint_path, "w", encoding="utf-8") as fh:
        json.dump({"cursor": final_cursor, "count": len(seen), "done": True}, fh)

    api_total = _validate_count(session, repo, token)

    raw_rows = list(_read_raw(raw_path))
    result = load_into_db(conn, raw_rows, snapshot_ts, repo, cfg, api_total)

    row_count = result["row_count"]
    print(f"[ingest] ingested {row_count} issues; API total_count {api_total}; "
          f"malformed {result['malformed_count']}")
    tol = cfg.get("ingest", {}).get("count_tolerance", DEFAULT_COUNT_TOLERANCE)
    ok, drift = count_within_tolerance(row_count, api_total, tol)
    if not ok:
        raise SystemExit(
            f"row_count {row_count} vs api_total_count {api_total} exceeds "
            f"{tol:.0%} tolerance ({drift:.3%})"
        )
    if api_total > 0:
        print(f"[ingest] count drift {drift:.3%} within {tol:.0%} tolerance")
    return result


def run_migrate(conn: sqlite3.Connection, cfg: dict) -> dict:
    """Replay the loader over the frozen raw snapshot — no network.

    Used to backfill columns (e.g. locked/active_lock_reason) added after a
    snapshot was ingested. The raw JSONL already holds the full API objects, so
    the snapshot stays byte-for-byte the same snapshot; only the DB is rebuilt.
    """
    db.migrate_schema(conn)
    snapshot_ts = db.get_meta(conn, "snapshot_ts")
    if snapshot_ts is None:
        raise SystemExit("no snapshot_ts in meta; nothing to migrate (run ingest first)")

    snapshot_date = snapshot_ts[:10]
    raw_path = os.path.join(RAW_ROOT, snapshot_date, "issues.jsonl")
    if not os.path.exists(raw_path):
        raise SystemExit(f"raw snapshot not found at {raw_path}; cannot replay")

    api_total = db.get_meta(conn, "api_total_count")
    api_total = int(api_total) if api_total is not None else None

    raw_rows = list(_read_raw(raw_path))
    result = load_into_db(conn, raw_rows, snapshot_ts, cfg["repo"], cfg, api_total)
    print(f"[migrate] replayed {result['row_count']} issues from {raw_path} "
          f"(snapshot {snapshot_ts} unchanged)")
    return result


def _resolve_snapshot(repo: str) -> tuple[str, str, str]:
    """Pick or resume today's snapshot. Returns (snapshot_ts, date, start_cursor)."""
    now = datetime.now(timezone.utc)
    snapshot_date = now.strftime("%Y-%m-%d")
    raw_dir = os.path.join(RAW_ROOT, snapshot_date)
    checkpoint_path = os.path.join(raw_dir, "checkpoint.json")
    meta_path = os.path.join(raw_dir, "snapshot_meta.json")
    if os.path.exists(checkpoint_path) and os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as fh:
            snapshot_ts = json.load(fh)["snapshot_ts"]
        with open(checkpoint_path, encoding="utf-8") as fh:
            cp = json.load(fh)
        # Resume from the stored cursor (rewinds slightly; `seen` dedups the
        # overlap). A stale pre-rev-4 checkpoint has no cursor -> start over.
        return snapshot_ts, snapshot_date, cp.get("cursor", INITIAL_CURSOR)
    return now.isoformat(), snapshot_date, INITIAL_CURSOR
