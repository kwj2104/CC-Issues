# GitHub Issue Opportunity Retrieval — Stage 1

A deterministic, **no-LLM** retrieval pipeline that ingests the open-issue
backlog of `anthropics/claude-code`, scores every issue for "opportunity" using
metadata + text features computed in pure code, and emits the **top 1,000
distinct problems** for downstream deep review.

See [`plan.md`](plan.md) for the full specification.

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
export GITHUB_TOKEN=<a token with public-repo read scope>
```

## Usage

```bash
python -m retrieval ingest     # M1: fetch open issues -> data/raw + SQLite
python -m retrieval migrate    # replay loader over frozen raw snapshot (no network)
python -m retrieval features   # M2: features + duplicate clustering
python -m retrieval score      # M3: score, select top_n -> out/*.csv
python -m retrieval report     # M4: diagnostics -> reports/*.md
python -m retrieval all        # everything, in order
```

`migrate` applies additive schema changes and re-runs the loader over the
already-downloaded `data/raw/<date>/issues.jsonl` to backfill columns added
after a snapshot was ingested — it never re-fetches from the network, so the
frozen snapshot stays identical.

Every stage after `ingest` reads only the SQLite DB — no network. Re-running
`score` with a changed `config.yaml` creates a **new run** (keyed by
`run_id = sha256(snapshot_ts + config)[:12]`) rather than overwriting the old
one, so weight experiments and sensitivity analysis stay comparable.

Common flags: `--config config.yaml` and `--db data/retrieval.db`.

## Outputs

- `out/top_1000.csv` — the selected opportunities (one row per duplicate cluster).
- `out/ranked_pool.csv` — every eligible issue, ranked.
- `reports/{composition,sensitivity,top20_preview,cluster_qc}.md` — diagnostics.

## Tests

```bash
.venv/bin/python -m pytest
```

All tunables live in [`config.yaml`](config.yaml).
