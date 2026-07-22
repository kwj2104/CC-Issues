"""CLI dispatch for the retrieval pipeline."""

from __future__ import annotations

import argparse
import sys

from . import cluster, db, features, ingest, report, score
from .config import load_config

DEFAULT_CONFIG = "config.yaml"
DEFAULT_DB = "data/retrieval.db"


def _common(sub):
    sub.add_argument("--config", default=DEFAULT_CONFIG)
    sub.add_argument("--db", default=DEFAULT_DB)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="retrieval")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("ingest", "migrate", "features", "score", "report", "all"):
        _common(sub.add_parser(name))
    return p


def cmd_ingest(conn, cfg):
    ingest.run_ingest(conn, cfg)


def cmd_migrate(conn, cfg):
    ingest.run_migrate(conn, cfg)


def cmd_features(conn, cfg):
    n = features.run_features(conn, cfg)
    k = cluster.run_clustering(conn, cfg)
    print(f"[features] computed features for {n} issues; {k} clusters")


def cmd_score(conn, cfg):
    run_id = score.run_score(conn, cfg)
    print(f"[score] wrote out/top_1000.csv and out/ranked_pool.csv for run {run_id}")


def cmd_report(conn, cfg):
    report.run_report(conn, cfg)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)

    import os

    os.makedirs(os.path.dirname(args.db) or ".", exist_ok=True)
    conn = db.connect(args.db)
    db.init_schema(conn)
    try:
        if args.command == "ingest":
            cmd_ingest(conn, cfg)
        elif args.command == "migrate":
            cmd_migrate(conn, cfg)
        elif args.command == "features":
            cmd_features(conn, cfg)
        elif args.command == "score":
            cmd_score(conn, cfg)
        elif args.command == "report":
            cmd_report(conn, cfg)
        elif args.command == "all":
            cmd_ingest(conn, cfg)
            cmd_features(conn, cfg)
            cmd_score(conn, cfg)
            cmd_report(conn, cfg)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
