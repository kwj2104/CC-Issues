# plan.md — GitHub Issue Opportunity Retrieval (Stage 1) — rev 2

> **rev 2 delta (eligibility filtering upgrade).** Adds a `question` label
> exclusion, lock-reason exclusion, a junk filter for abandoned empty reports,
> and a maintainer-authored flag (flag only — never affects eligibility or
> score). Touches §3, §4, §5, §7, §8, §9. A no-network `migrate` command
> replays the loader over the frozen raw snapshot to backfill new columns.

**Purpose:** implement a deterministic, no-LLM retrieval pipeline that ingests the open-issue backlog of `anthropics/claude-code`, scores every issue for "opportunity" using metadata + text features computed in pure code, and emits the **top 1,000 distinct problems** for downstream deep review. This document is a complete implementation spec for a coding agent.

**Non-goals:** no LLM calls anywhere; no issue summarization or semantic tagging (that's Stage 2, out of scope); no writes to GitHub; no UI.

---

## 1. Tech stack & constraints

- Python ≥ 3.11. Dependencies: `requests`, `pandas`, `scikit-learn`, `scipy`, `pyyaml`. Stdlib `sqlite3` for storage. No other runtime dependencies. (Optional: `datasketch` behind a config flag — see §6; not required for v1.)
- Auth: GitHub personal access token from env var `GITHUB_TOKEN` (public-repo read scope). Fail fast with a clear message if unset.
- Determinism is a hard requirement: given the same raw snapshot and config, every output file must be **byte-identical** across runs. No wall-clock reads after ingest (snapshot timestamp is stored once), no unseeded randomness (the only RNG use is cluster-QC sampling, seeded from config).
- Network calls: GitHub REST API only, and only during `ingest`.
- Runtime budget: ingest ≤ ~20 min (API-bound); everything after ingest ≤ 5 min and ≤ 2 GB RAM on a laptop.

## 2. Repo layout & CLI

```
retrieval/
  __main__.py          # CLI dispatch
  config.py            # config load + validation + run_id derivation
  ingest.py            # M1
  features.py          # M2 (incl. text prep)
  cluster.py           # M2 (duplicate clustering)
  score.py             # M3
  report.py            # M4
  db.py                # schema DDL + connection helpers
config.yaml            # all tunables (defaults in §3)
data/
  raw/<snapshot_date>/issues.jsonl      # append-only raw layer
  retrieval.db                          # SQLite working layer
out/
  top_1000.csv
  ranked_pool.csv
reports/
  composition.md  sensitivity.md  cluster_qc.md  top20_preview.md
tests/
  fixtures/issues_fixture.jsonl         # ~30 handcrafted rows
  test_ingest.py  test_features.py  test_cluster.py  test_score.py
```

CLI (argparse):

```
python -m retrieval ingest   [--config config.yaml] [--db data/retrieval.db]
python -m retrieval migrate  [--config ...] [--db ...]   # replay loader over frozen raw snapshot
python -m retrieval features [--config ...] [--db ...]
python -m retrieval score    [--config ...] [--db ...]
python -m retrieval report   [--config ...] [--db ...]
python -m retrieval all      [--config ...] [--db ...]
```

Each stage is independently re-runnable: `migrate`, `features`, `score`, `report` read only the DB (and, for `migrate`, the raw JSONL layer) — never the network. `score` with a changed config creates a **new run** (see §7) rather than overwriting.

`migrate` (added in rev 2) applies additive schema changes (ALTER TABLE) and re-runs the SQLite loader over `data/raw/<snapshot_date>/issues.jsonl` to backfill columns added after ingest. It **never touches the network** — the frozen snapshot stays the same snapshot; only the DB is rebuilt.

## 3. Config (`config.yaml` defaults)

```yaml
repo: anthropics/claude-code
window:
  created_days: 90          # pool: created within N days of snapshot_ts
  carveout_updated_days: 45 # OR older but updated within N days
  carveout_min_reactions: 25 # OR older but holding >= N reactions
weights:
  reactions: 3.0
  comments: 1.0
  velocity: 2.0
  severity: 2.0
  demand: 1.0
  cluster: 2.0
severity:
  cap: 5.0
  label_weights:            # matched against lowercased label names, exact
    data-loss: 3.0
    area:security: 2.0
    regression: 2.0
    high-priority: 2.0
    perf:reliability: 1.5
    med-priority: 1.0
    "has repro": 0.5
  regex_bonus: 1.0          # added at most once if any pattern matches title+body_lead
  regex_bank: "crash|hang|freeze|data.?loss|corrupt|segfault|CVE-\\d{4}|after updat"
demand:
  labels: [enhancement, "feature request", feature]
  min_reactions: 5
clustering:
  similarity_threshold: 0.6
  ngram_range: [1, 2]
  min_df: 2
  max_df: 0.5
  body_lead_chars: 1500
selection:
  top_n: 1000
  exclude_labels: [duplicate, invalid, stale, autoclose, question]
  exclude_lock_reasons: [spam, resolved]
  junk_filter:              # excluded only if ALL conditions hold
    max_clean_body_chars: 40
    max_reactions: 0
    max_comments: 0
    min_age_days: 7
sensitivity:
  perturbation: 0.5         # ±50% per weight
qc:
  cluster_sample: 20
  seed: 42
```

## 4. M1 — Ingest

**Endpoint:** `GET https://api.github.com/repos/{repo}/issues?state=open&per_page=100&page=N` with headers `Authorization: Bearer $GITHUB_TOKEN`, `Accept: application/vnd.github+json`, `X-GitHub-Api-Version: 2022-11-28`.

Rules:

1. Paginate until an empty page. **Skip any row containing a `pull_request` key** (the endpoint interleaves PRs).
2. Append each kept item as one JSON line (verbatim API object) to `data/raw/<YYYY-MM-DD>/issues.jsonl`. The raw layer is append-only and never edited. Maintain a `checkpoint.json` (last completed page) alongside it so an interrupted ingest resumes rather than restarts; a completed ingest finalizes the snapshot.
3. Rate limits: read `X-RateLimit-Remaining`/`X-RateLimit-Reset`; if remaining < 10, sleep until reset. On 403/429 sleep 60s and retry (max 5); on 5xx exponential backoff 5s→80s (max 5); on persistent failure exit nonzero with the page number.
4. After pagination, make **one** validation call to `GET /search/issues?q=repo:{repo}+type:issue+state:open&per_page=1` and compare `total_count` with rows ingested. Tolerance ±1% (state changes mid-ingest). Outside tolerance → exit nonzero with both numbers.
5. Load into SQLite (schema below): parse timestamps as UTC ISO-8601 strings; `body` may be null → store `''`; lowercase all label names; `reactions_total` = `reactions.total_count`, `reactions_plus1` = `reactions["+1"]`; capture `locked` (absent → 0) and `active_lock_reason` (absent → NULL). Store `snapshot_ts` (UTC now, once) and config snapshot in `meta`.

**Migration (rev 2):** columns added after the initial release (`issues.locked`, `issues.active_lock_reason`, `features.is_junk`, `features.maintainer_authored`) are patched into pre-existing DBs idempotently via `ALTER TABLE ADD COLUMN`. Because the raw JSONL layer already stores the full API objects, `migrate` **replays the loader from `data/raw/<date>/issues.jsonl`** (preserving the stored `snapshot_ts`/`api_total_count`) rather than re-fetching — the frozen snapshot must stay the same snapshot.

**Schema (DDL, `db.py`):**

```sql
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
-- keys: snapshot_ts, repo, row_count, api_total_count, tool_version

CREATE TABLE IF NOT EXISTS issues (
  number          INTEGER PRIMARY KEY,
  title           TEXT NOT NULL,
  body            TEXT NOT NULL DEFAULT '',
  state           TEXT NOT NULL,
  created_at      TEXT NOT NULL,   -- ISO8601 UTC
  updated_at      TEXT NOT NULL,
  comments        INTEGER NOT NULL,
  reactions_total INTEGER NOT NULL,
  reactions_plus1 INTEGER NOT NULL,
  author_association TEXT,
  html_url        TEXT NOT NULL,
  locked          INTEGER NOT NULL DEFAULT 0,   -- rev 2
  active_lock_reason TEXT                        -- rev 2 (e.g. spam, resolved, off-topic)
);

CREATE TABLE IF NOT EXISTS issue_labels (
  number INTEGER NOT NULL REFERENCES issues(number),
  label  TEXT NOT NULL,            -- lowercased
  PRIMARY KEY (number, label)
);
CREATE INDEX IF NOT EXISTS idx_labels_label ON issue_labels(label);

CREATE TABLE IF NOT EXISTS features (
  number INTEGER PRIMARY KEY REFERENCES issues(number),
  age_days REAL NOT NULL,          -- (snapshot_ts - created_at)/86400, floor 1.0
  f_reactions REAL NOT NULL,       -- log2(1+reactions_total)
  f_comments  REAL NOT NULL,       -- log2(1+comments)
  f_velocity  REAL NOT NULL,       -- log2(1 + 30*(reactions_total+comments)/age_days)
  f_severity  REAL NOT NULL,       -- per §5, capped
  f_demand    REAL NOT NULL,       -- per §5
  is_junk            INTEGER NOT NULL,  -- rev 2: abandoned empty report
  maintainer_authored INTEGER NOT NULL, -- rev 2: flag only, never affects score/eligibility
  in_pool  INTEGER NOT NULL,       -- window/carve-out predicate
  eligible INTEGER NOT NULL        -- rev 2: in_pool AND labels AND lock-reason AND NOT junk
);

CREATE TABLE IF NOT EXISTS clusters (
  number INTEGER PRIMARY KEY REFERENCES issues(number),
  cluster_id INTEGER NOT NULL,     -- singletons get their own id
  cluster_size INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS score_runs (
  run_id TEXT PRIMARY KEY,         -- sha256(snapshot_ts + canonical_config_json)[:12]
  created_ts TEXT NOT NULL,
  weights_json TEXT NOT NULL,
  config_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scores (
  run_id TEXT NOT NULL REFERENCES score_runs(run_id),
  number INTEGER NOT NULL REFERENCES issues(number),
  score REAL NOT NULL,
  c_reactions REAL NOT NULL, c_comments REAL NOT NULL, c_velocity REAL NOT NULL,
  c_severity REAL NOT NULL, c_demand REAL NOT NULL, c_cluster REAL NOT NULL,
  rank INTEGER,                    -- rank among eligible, 1-based; NULL if ineligible
  selected INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (run_id, number)
);
```

**Pool filter** (computed into `features.in_pool`, so the window is a query, not a code path):

```sql
julianday(meta.snapshot_ts) - julianday(issues.created_at) <= 90
OR julianday(meta.snapshot_ts) - julianday(issues.updated_at) <= 45
OR issues.reactions_total >= 25
```

## 5. M2 — Features

All formulas exact; `age_days = max(1.0, (snapshot_ts - created_at)/86400)` (floor guards divide-by-zero and clock skew).

- `f_reactions = log2(1 + reactions_total)`
- `f_comments = log2(1 + comments)`
- `f_velocity = log2(1 + 30 * (reactions_total + comments) / age_days)`
- `f_severity = min(cap, Σ label_weights[matching labels] + (regex_bonus if regex matches else 0))` — regex is case-insensitive, applied to `title + " " + body_lead` (see text prep), added **at most once** regardless of match count.
- `f_demand = log2(reactions_total)` if the issue has any demand label AND `reactions_total ≥ min_reactions`, else `0`.

**Two computed flags (rev 2):**

- `is_junk = 1` iff ALL hold: `len(clean_body) ≤ junk_filter.max_clean_body_chars (40)` AND `reactions_total ≤ max_reactions (0)` AND `comments ≤ max_comments (0)` AND `age_days ≥ min_age_days (7)`. `clean_body` is the **body portion** after the text-prep pipeline below (template/code/URL stripping), **without the title** and untruncated. Junk = an abandoned empty report: no body, no engagement, a week old.
- `maintainer_authored = 1` iff `author_association ∈ {OWNER, MEMBER, COLLABORATOR}`. This is a **flag only** — it never affects eligibility or score; megathreads/tracking issues are containers flagged for downstream treatment, never discarded here.

**Text prep** (shared by severity regex and clustering; implement in `features.py`, unit-test it):

1. Take `title + "\n" + body`.
2. Remove fenced code blocks (```` ``` … ``` ````), HTML comments (`<!-- … -->`, which is where issue-template boilerplate lives), URLs, and image markdown.
3. Remove issue-template headers (lines starting with `###` whose text is in a small stoplist: "Environment", "What happened", "Steps to reproduce", "Expected behavior", "Preflight checklist", "Version", "Platform").
4. Collapse whitespace; truncate body part to `body_lead_chars` (1500). Result: `clean_text`.
5. For clustering, the document is `title + " " + title + " " + body_lead` (title doubled to up-weight it).

## 6. M2 — Duplicate clustering

Clustering runs over **all ingested open issues** (not just the pool), so duplicate mass outside the pool still credits its cluster.

1. Vectorize documents (from §5.5) with `TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.5, stop_words='english', lowercase=True)`.
2. Pairwise cosine similarity, **chunked** (e.g., 1,000-row blocks of `X @ X.T`) to respect the 2 GB budget; collect pairs with similarity ≥ `similarity_threshold`.
3. Union-find over the pairs → connected components. Every issue gets a `cluster_id` (singletons included) and `cluster_size`.
4. `c_cluster` score component (computed in M3) = `log2(cluster_size)` (0 for singletons).
5. QC artifact: sample `qc.cluster_sample` clusters of size ≥ 2 (RNG seeded with `qc.seed`), dump their member titles to `reports/cluster_qc.md` for human review of the threshold.

(`datasketch` MinHash-LSH may replace step 2 behind a config flag if the chunked matmul proves too slow — same interface: emit pairs ≥ threshold.)

## 7. M3 — Score & select

```
score = w.reactions*f_reactions + w.comments*f_comments + w.velocity*f_velocity
      + w.severity*f_severity + w.demand*f_demand + w.cluster*log2(cluster_size or 1)
```

- Compute for **every** issue; store per-component contributions (`c_*` columns) for auditability.
- `run_id = sha256(snapshot_ts + canonical_json(config))[:12]`; insert into `score_runs`. Same snapshot + same config → same `run_id` → re-run replaces that run idempotently. Changed weights → new run, old runs preserved (this is what makes sensitivity analysis and weight experiments comparable). The rev 2 config change alters the canonical config JSON → a **new `run_id`**; old runs stay in `score_runs`/`scores` (delete nothing), and reports use the new baseline run.
- **Eligibility (rev 2):** `eligible = in_pool AND (no label ∈ exclude_labels) AND (active_lock_reason ∉ exclude_lock_reasons, compared lowercased) AND NOT is_junk`. Ineligible rows still receive scores and cluster membership and still appear in `ranked_pool.csv` and diagnostics — they simply cannot occupy a top-1,000 slot.
- **Selection:** among `eligible` issues, rank by `score DESC`, tie-break `reactions_total DESC`, then `number DESC` (full determinism). Walk the ranking keeping only the **first (highest-scoring) eligible member of each cluster_id**; stop at `top_n`. Mark `selected=1`. Ineligible cluster members still contribute to `cluster_size` but never represent the cluster.
- Outputs:
  - `out/top_1000.csv` — columns: `rank, number, html_url, title, created_at, updated_at, age_days, reactions_total, comments, maintainer_authored, labels` (`;`-joined), `cluster_id, cluster_size, cluster_members` (`;`-joined numbers, capped at 50), `score, c_reactions, c_comments, c_velocity, c_severity, c_demand, c_cluster, run_id, snapshot_ts`. (`maintainer_authored` added in rev 2.)
  - `out/ranked_pool.csv` — same columns for every eligible issue.

## 8. M4 — Diagnostics (`report`)

1. `reports/composition.md` (+ a machine-readable `.csv` twin): pool size and carve-out share; top-1,000 vs full-pool distributions of age buckets (≤7d, 8–30d, 31–90d, >90d), bug/enhancement/other label mix, top-15 `area:*` labels, engagement deciles; count of selected issues admitted **only** via carve-out. **(rev 2)** an **exclusion waterfall** with counts at every step: open snapshot → in_pool → after each `exclude_label` (count per label) → after lock-reason filter → after junk filter → eligible (each step's remaining ≤ the previous; final remaining = the features-table eligible count); plus the count of `maintainer_authored` rows inside the top 1,000.
2. `reports/sensitivity.md`: for each of the 6 weights, re-rank with that weight ×0.5 and ×1.5 (12 variants; selection re-run in memory, no DB writes) → table of Jaccard overlaps between each variant's top-1,000 and the baseline's. Include min/mean overlap headline.
3. `reports/top20_preview.md`: rank, number, title, score components, link — the human sanity check before Stage 2 commits.
4. `reports/cluster_qc.md`: per §6.5.

## 9. Acceptance criteria

1. `python -m retrieval all` completes against the live repo with only `GITHUB_TOKEN` set; ingest ≤ ~20 min, post-ingest ≤ 5 min, ≤ 2 GB RAM.
2. Ingested row count within ±1% of the search-API `total_count`; both values stored in `meta` and printed.
3. `top_1000.csv` has exactly `top_n` rows; all `number`s unique; **at most one row per `cluster_id`**; every row satisfies the pool predicate. **(rev 2)** no selected row carries an excluded label, an excluded lock reason, or `is_junk = 1`.
4. Determinism: running `score` + `report` twice on the same DB and config produces byte-identical `top_1000.csv` (verify by hash in a test).
5. Unit tests: text prep (fixture strings → expected clean output); each feature formula against hand-computed values on the ~30-row fixture; union-find clustering on a fixture with two known duplicate groups; selection logic incl. tie-breaks and cluster dedup; severity cap and regex once-only bonus. **(rev 2)** `is_junk` boundaries (clean_body 39/40/41 chars; age 6.9 vs 7.0; exactly 1 reaction or 1 comment is not junk); `maintainer_authored` mapping for each `author_association`; lock-reason exclusion (spam/resolved excluded; other/NULL kept); waterfall counts sum consistently (each step ≤ previous; eligible matches the features table); the determinism hash covers the new CSV column.
6. All 12 sensitivity variants generated; `composition.md` renders with real numbers.
7. `grep`-provable: no network access outside `ingest.py`; no LLM/API-model imports anywhere.

## 10. Edge cases the implementation must handle

- `body` null or non-UTF8 → treat as empty / decode with `errors="replace"`.
- Issues with zero age (created seconds before snapshot) → `age_days` floor 1.0.
- Label names containing spaces and colons (`has repro`, `area:security`) — always compare lowercased, exact string.
- The list endpoint occasionally returns transferred/deleted issues as 404 stubs mid-pagination — skip malformed rows, count them, report in `meta`.
- Pagination termination: empty page, not `Link` header parsing (simpler and sufficient).
- Reactions object missing (rare) → all reaction fields 0.
- Duplicate `number` across pages (issue state changed mid-ingest, page shift) → last write wins in SQLite (`INSERT OR REPLACE`), raw JSONL keeps both lines.
- A cluster whose highest-scoring member is ineligible (e.g., labeled `duplicate`) — the cluster is represented by its highest-scoring *eligible* member; ineligible members still contribute to `cluster_size`.

## 11. Context for the implementer (facts, as of 2026-07-22)

- Open issues ≈ 12k; ~322 new issues/day; ~87% of open backlog is ≤ 90 days old; ~1,590 issues are older, of which ~900 pass the carve-out. Expect a pool of ~11.3k and an eligible set slightly smaller after label exclusions.
- ~96 distinct labels exist; a triage bot applies `area:*`/`platform:*`/type labels with ~1-day lag, so very recent issues are sparsely labeled — the regex severity path and velocity term exist precisely to compensate.
- 1,274 open issues have >10 reactions: engagement alone nearly fills the top 1,000, so cluster/severity/velocity terms decide the meaningful margins. If diagnostics show the head is >90% pure-engagement picks, flag it in `composition.md` rather than silently accepting.
