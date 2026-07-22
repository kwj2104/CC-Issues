"""rev 4: `since`-cursor pagination (offline, mocked GitHub responses)."""

from __future__ import annotations

import json

from retrieval import ingest


class _FakeResponse:
    def __init__(self, payload):
        self.status_code = 200
        self.headers = {}
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


class _FakeSession:
    """Emulates GET /issues?since=... sorted by updated_at asc, capped at per_page."""

    def __init__(self, dataset):
        self.dataset = sorted(dataset, key=lambda d: d["updated_at"])
        self.calls = 0

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls += 1
        since = params.get("since")  # omitted on the first (start-of-backlog) call
        per_page = params["per_page"]
        if since is None:
            rows = self.dataset[:per_page]
        else:
            rows = [d for d in self.dataset if d["updated_at"] >= since][:per_page]
        return _FakeResponse(rows)


def _issue(num, updated, is_pr=False):
    d = {
        "number": num,
        "title": f"i{num}",
        "updated_at": updated,
        "created_at": updated,
        "html_url": f"u/{num}",
        "reactions": {"total_count": 0},
        "labels": [],
    }
    if is_pr:
        d["pull_request"] = {"url": "p"}
    return d


def test_cursor_walks_past_offset_and_dedups(tmp_path, monkeypatch):
    # Force small pages so the dataset spans several fetches.
    monkeypatch.setattr(ingest, "PER_PAGE", 3)
    dataset = [
        _issue(1, "2020-01-01T00:00:00Z"),
        _issue(2, "2020-02-01T00:00:00Z"),
        _issue(3, "2020-03-01T00:00:00Z"),
        _issue(4, "2020-03-15T00:00:00Z", is_pr=True),  # PR -> skipped
        _issue(5, "2020-04-01T00:00:00Z"),
        _issue(6, "2020-05-01T00:00:00Z"),
        _issue(7, "2020-06-01T00:00:00Z"),
        _issue(8, "2020-06-01T00:00:00Z"),  # shares ts with #7 (boundary)
    ]
    raw_path = tmp_path / "issues.jsonl"
    cp_path = tmp_path / "checkpoint.json"
    seen: set = set()

    final = ingest._fetch_issues(
        _FakeSession(dataset), "o/r", "tok", str(raw_path), str(cp_path),
        ingest.INITIAL_CURSOR, seen,
    )

    numbers = [json.loads(l)["number"] for l in raw_path.read_text().splitlines()]
    # every open issue captured exactly once; PR #4 excluded; no duplicates.
    assert sorted(numbers) == [1, 2, 3, 5, 6, 7, 8]
    assert len(numbers) == len(set(numbers))
    assert final == "2020-06-01T00:00:00Z"
    # checkpoint records the cursor for resume.
    assert json.loads(cp_path.read_text())["cursor"] == "2020-06-01T00:00:00Z"


def test_cursor_resume_skips_already_seen(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "PER_PAGE", 3)
    dataset = [_issue(n, f"2021-0{n}-01T00:00:00Z") for n in range(1, 5)]
    raw_path = tmp_path / "issues.jsonl"
    cp_path = tmp_path / "checkpoint.json"

    # simulate a prior partial run that already wrote #1 and #2
    raw_path.write_text(
        json.dumps(dataset[0]) + "\n" + json.dumps(dataset[1]) + "\n"
    )
    seen = {1, 2}
    ingest._fetch_issues(
        _FakeSession(dataset), "o/r", "tok", str(raw_path), str(cp_path),
        ingest.INITIAL_CURSOR, seen,
    )
    numbers = [json.loads(l)["number"] for l in raw_path.read_text().splitlines()]
    assert sorted(numbers) == [1, 2, 3, 4]  # 1/2 not re-appended
    assert numbers.count(1) == 1 and numbers.count(2) == 1


def test_empty_repo_terminates(tmp_path):
    raw_path = tmp_path / "issues.jsonl"
    cp_path = tmp_path / "checkpoint.json"
    raw_path.write_text("")
    final = ingest._fetch_issues(
        _FakeSession([]), "o/r", "tok", str(raw_path), str(cp_path),
        ingest.INITIAL_CURSOR, set(),
    )
    assert final == ingest.INITIAL_CURSOR
    assert raw_path.read_text() == ""
