# GitHub Issue Opportunity Retrieval — Stage 1

A **deterministic, no-LLM** retrieval pipeline that ingests the open-issue backlog
of `anthropics/claude-code`, scores every issue for "opportunity" from metadata +
text features computed in pure Python, and emits the **top 1,000 distinct problems**
as a declared lane portfolio for downstream (Stage 2) deep review.

- **No LLM calls anywhere** — every signal is a closed-form feature or a TF-IDF
  similarity. (`grep`-provable: no model imports; no network outside `ingest`.)
- **Deterministic** — same raw snapshot + same config ⇒ byte-identical outputs.
  The snapshot timestamp is captured once at ingest; nothing after reads the clock.
- **Auditable** — every score decomposes into stored per-component contributions,
  and every selected row is tagged with the lane that picked it.

The authoritative spec (with revision history) is [`plan.md`](plan.md); this README
explains how the mechanism actually works.

---

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
export GITHUB_TOKEN=<token with public-repo read scope>   # only needed for `ingest`
```

## Usage

```bash
python -m retrieval ingest     # M1: fetch open issues  -> data/raw + SQLite
python -m retrieval migrate    # replay loader over the frozen raw snapshot (no network)
python -m retrieval features   # M2: features + rate_score + duplicate clustering
python -m retrieval score      # M3: score + lane selection -> out/*.csv
python -m retrieval report     # M4: diagnostics -> reports/*.md
python -m retrieval all        # ingest -> features -> score -> report
```

Flags: `--config config.yaml`, `--db data/retrieval.db`. Every stage after `ingest`
reads **only** the SQLite DB — no network. All tunables live in
[`config.yaml`](config.yaml).

---

## How it works

The pipeline is four stages over three storage layers. Data flows one way; each
stage is independently re-runnable from the DB.

```
GitHub REST API
      │  (M1 ingest — the only network step)
      ▼
data/raw/<date>/issues.jsonl      append-only raw layer (verbatim API objects)
      │  load
      ▼
data/retrieval.db (SQLite)        issues, issue_labels, features, clusters,
      │                           score_runs, scores  — the working layer
      │  M2 features + clustering
      │  M3 score + lane selection
      ▼
out/top_1000.csv, out/ranked_pool.csv      the deliverables
reports/*.md                                the diagnostics
```

### M1 — Ingest (`ingest.py`)

Pulls every **open issue** (pull requests are interleaved by the endpoint and
skipped via their `pull_request` key).

- **Cursor pagination.** GitHub's REST *offset* pagination caps at ~10,000 items
  (`page=100` → HTTP 422), and this repo has ~12k open issues. So we page with the
  **`since` cursor** (`sort=updated&direction=asc`): start with no `since`, then
  advance it to the max `updated_at` seen each page. `since` is inclusive, so the
  boundary row reappears; a `seen` set (rebuilt by replaying the raw file on resume)
  drops duplicates, so each issue is written exactly once and the walk clears the
  10k wall at any repo size.
- **Append-only raw layer.** Each kept object is one JSON line in
  `data/raw/<snapshot_date>/issues.jsonl`, never edited. A `checkpoint.json`
  (`{cursor, count, done}`) makes an interrupted ingest resume rather than restart.
- **Resilience.** Rate-limit aware (sleeps to reset when remaining < 10); retries
  403/429 and 5xx with backoff.
- **Count validation.** One search-API call gives the authoritative open-issue
  `total_count`; ingest fails if the loaded count drifts beyond
  `ingest.count_tolerance` (default **±3%** — GitHub's search count is approximate
  and runs ~2% high, but a truncated pull is ~19%, so this still catches breakage).
- **Load.** Timestamps as UTC ISO-8601; null body → `''`; label names lowercased;
  `locked`/`active_lock_reason` captured; `snapshot_ts` (UTC, once) stored in `meta`.

`migrate` re-runs the loader over the already-downloaded raw JSONL to backfill
columns added after a snapshot — **never re-fetching**, so the frozen snapshot is
preserved.

### M2 — Features, rate_score, clustering (`features.py`, `cluster.py`)

**Text prep** (shared by severity regex + clustering): from `title + body`, strip
fenced code blocks, HTML comments, URLs, image markdown, and issue-template
boilerplate headers; collapse whitespace; truncate the body to `body_lead_chars`.

**Per-issue features** (`age_days = max(1, (snapshot − created)/86400)`):

| feature | formula |
|---|---|
| `f_reactions` | `log2(1 + reactions_total)` |
| `f_comments` | `log2(1 + comments)` |
| `f_velocity` | `log2(1 + 30·(reactions+comments)/age_days)` |
| `f_severity` | `min(cap, Σ label_weights + regex_bonus)` — regex on title+body, added at most once |
| `f_demand` | `log2(reactions_total)` if a demand label AND `reactions ≥ min`, else 0 |

**`rate_score`** (age-corrected, filled after clustering since it needs cluster size):

```
rate_score = w.reactions·log2(1 + 30·reactions/age_days)
           + w.comments ·log2(1 + 30·comments /age_days)
           + w.severity ·severity
           + w.cluster  ·log2(cluster_size or 1)
```

Engagement enters as **monthly rates**, not raw totals, and there is no demand term.
Rationale: raw reaction counts grow mechanically with age (pool median engagement is
0 at ≤30 days vs 19 at >90 days), so a raw-engagement ranking is really an age
ranking. `rate_score` is what the freshness-sensitive lanes rank by.

**Duplicate clustering.** TF-IDF (1–2-grams, English stop-words) over all issue
documents (title doubled to up-weight it), chunked cosine similarity ≥
`similarity_threshold`, then **union-find** into connected components. Every issue
gets a `cluster_id` (singletons included) and `cluster_size`; the cluster id is the
smallest member number (deterministic). Duplicate mass is what the `volume` lane and
the `cluster` score term reward.

**Eligibility** (who may occupy a top-1,000 slot):

```
eligible = in_pool
         AND no label ∈ exclude_labels     (duplicate, invalid, stale, autoclose, question)
         AND active_lock_reason ∉ exclude_lock_reasons   (spam, resolved)
         AND NOT is_junk
```

- `in_pool = 1` for every open issue — there is **no calendar window**. The repo's
  own lifecycle automation (stale at 14d inactivity unless ≥10 👍, autoclose 14d
  later) already defines a live issue, so we inherit that policy through
  `exclude_labels` instead of imposing a window.
- **Junk filter** (`is_junk`): abandoned empty reports — clean body ≤ 40 chars AND 0
  reactions AND 0 comments AND ≥ 7 days old (all four must hold).
- **Stale rescue**: the `stale` label stops excluding at
  `stale_rescue_min_reactions` (10) reactions, mirroring the repo sweep's own
  upvote exemption.
- `maintainer_authored` (OWNER/MEMBER/COLLABORATOR) is a **flag only** — megathreads
  and tracking issues are marked for Stage 2, never dropped, and it never affects
  eligibility or score.

Ineligible issues still get scored and clustered and still appear in
`ranked_pool.csv` — they just can't take a slot.

### M3 — Score + lane selection (`score.py`)

The **base `score`** (used by the `big-bets` lane and the ranked pool) is the
weighted sum of the raw-total features, with per-component contributions (`c_*`)
stored for auditability:

```
score = w.reactions·f_reactions + w.comments·f_comments + w.velocity·f_velocity
      + w.severity ·f_severity  + w.demand ·f_demand   + w.cluster ·log2(cluster_size or 1)
```

Selection is a **declared 4-lane portfolio** rather than a flat top-N, so the head
isn't just one ranking (raw engagement) wearing four hats. Lanes are processed in
order, each takes rows whose `number` **and** `cluster_id` are still unclaimed
(global one-row-per-cluster across the whole head), ranking by its keys with
universal tie-breaks `reactions_total DESC, number DESC`:

| lane | slots | filter | ranks by | intent |
|---|---|---|---|---|
| `big-bets` | 350 | — | `score` | accumulated raw demand |
| `emerging` | 350 | `age_days ≤ 30` | `rate_score` | fresh, fast-moving |
| `severity` | 200 | `severity ≥ 2.0` | `severity, rate_score` | reliability / security |
| `volume` | 100 | `cluster_size ≥ 2` | `cluster_size, rate_score` | duplicate mass |

Lane slots sum to `top_n`. If a lane underfills, remaining slots are refilled by
`spill_order` (`emerging-rank` = remaining eligible by `rate_score`; then
`global-score` = by `score`); **spill rows skip lane filters but never eligibility**.
Every selected row is tagged `selection_lane` (a lane name or `spill:<source>`).

**Runs are content-addressed.** `run_id = sha256(snapshot_ts + canonical_config)[:12]`.
Changing weights or lanes → a new `run_id`; old runs are preserved in
`score_runs`/`scores`, so weight experiments and sensitivity analysis stay
comparable. Same snapshot + same config re-runs idempotently.

### M4 — Diagnostics (`report.py`)

- **`composition.md`** — the exclusion waterfall (per-label removal counts + a
  stale-rescued line, ending exactly at the eligible count), age / label /
  `area:*` / engagement-decile distributions, **per-lane composition** (fills,
  median age/rate_score/severity, and per-lane engagement-dominance %), a **lane
  overlap matrix** (how many of each lane's picks also pass the other lanes'
  filters), and the Jaccard of the lane head vs the retired single-score head.
  Engagement deciles are **rank-based** (heavy ties share a decile) with a note
  quantifying the zero-engagement collapse.
- **`sensitivity.md`** — robustness. (a) Each of the 6 weights ×0.5 / ×1.5 (12
  variants), each **re-running the full lane portfolio** and reporting Jaccard vs
  the baseline lane head; an unperturbed re-run must score exactly 1.0 (asserted).
  (b) Creation-window variants (90 / 180 days) show how much a calendar window
  *would* have changed the pick.
- **`top20_preview.md`** — the human sanity check before Stage 2.
- **`cluster_qc.md`** — a seeded sample of multi-member clusters to eyeball the
  similarity threshold.

---

## Outputs

- **`out/top_1000.csv`** — the selected opportunities, one row per duplicate
  cluster, each carrying its `selection_lane`, `score`, `rate_score`, the six `c_*`
  component contributions, cluster members, labels, and `maintainer_authored`.
- **`out/ranked_pool.csv`** — every eligible issue, ranked by `score` (same columns).
- **`reports/{composition,sensitivity,top20_preview,cluster_qc}.md`** (+
  `composition.csv` machine twin).

## Determinism

Given a fixed raw snapshot and config, `features → score → report` produces
byte-identical CSVs. Guarantees: the snapshot timestamp is stored once at ingest and
never re-read; the only RNG (cluster-QC sampling) is seeded from config; ranking uses
total tie-breaks (`… , reactions_total DESC, number DESC`); floats are emitted at
fixed precision. A test asserts `top_1000.csv` is byte-identical across reruns.

## Tests

```bash
.venv/bin/python -m pytest
```

Covers text prep, every feature formula against hand-computed values, `rate_score`,
union-find clustering on known duplicate groups, junk/stale-rescue/lock-reason
eligibility, lane fill/ordering/spill/dedup, the exclusion waterfall, the
sensitivity harness, cursor pagination (mocked), and output determinism.
