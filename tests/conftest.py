"""Shared test fixtures: an in-memory DB loaded from the JSONL fixture."""

from __future__ import annotations

import json
import os

import pytest

from retrieval import db, ingest
from retrieval.config import load_config

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "issues_fixture.jsonl")
# Fixed snapshot so age-based features are deterministic and hand-computable.
SNAPSHOT_TS = "2026-07-22T00:00:00+00:00"


def _read_fixture():
    with open(FIXTURE, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


@pytest.fixture
def cfg():
    root = os.path.dirname(os.path.dirname(__file__))
    return load_config(os.path.join(root, "config.yaml"))


@pytest.fixture
def conn(cfg):
    c = db.connect(":memory:")
    db.init_schema(c)
    ingest.load_into_db(
        c, list(_read_fixture()), SNAPSHOT_TS, cfg["repo"], cfg, api_total_count=26
    )
    yield c
    c.close()


@pytest.fixture
def loaded(conn, cfg):
    """DB with features + clusters computed."""
    from retrieval import cluster, features

    features.run_features(conn, cfg)
    cluster.run_clustering(conn, cfg)
    return conn
