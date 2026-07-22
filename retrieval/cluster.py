"""M2 — duplicate clustering over all ingested open issues."""

from __future__ import annotations

import sqlite3

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from . import features


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def _similar_pairs(matrix, threshold: float, block: int = 1000):
    """Yield (i, j) index pairs with i < j and cosine similarity >= threshold.

    TF-IDF rows are L2-normalised, so X @ X.T is cosine similarity. Computed
    in row blocks to bound memory.
    """
    n = matrix.shape[0]
    for start in range(0, n, block):
        end = min(start + block, n)
        sims = (matrix[start:end] @ matrix.T).toarray()
        rows, cols = np.nonzero(sims >= threshold)
        for r, c in zip(rows.tolist(), cols.tolist()):
            i = start + r
            if i < c:  # keep i < j, drop diagonal and mirror
                yield i, c


def run_clustering(conn: sqlite3.Connection, cfg: dict) -> int:
    """Cluster all issues by title+body similarity. Returns cluster count."""
    ccfg = cfg["clustering"]
    body_lead_chars = ccfg["body_lead_chars"]

    rows = conn.execute(
        "SELECT number, title, body FROM issues ORDER BY number ASC"
    ).fetchall()
    numbers = [r["number"] for r in rows]
    docs = [
        features.prep_text(r["title"], r["body"], body_lead_chars)["cluster_doc"]
        for r in rows
    ]
    n = len(numbers)

    uf = _UnionFind(n)
    if n >= 1:
        vec = TfidfVectorizer(
            ngram_range=tuple(ccfg["ngram_range"]),
            min_df=ccfg["min_df"],
            max_df=ccfg["max_df"],
            stop_words="english",
            lowercase=True,
        )
        try:
            matrix = vec.fit_transform(docs)
            if matrix.shape[1] > 0:
                for i, j in _similar_pairs(matrix, ccfg["similarity_threshold"]):
                    uf.union(i, j)
        except ValueError:
            # Empty vocabulary (e.g. all-stopword corpus) -> all singletons.
            pass

    # Root index -> stable cluster_id (smallest issue number in component).
    members: dict[int, list] = {}
    for idx in range(n):
        members.setdefault(uf.find(idx), []).append(idx)

    conn.execute("DELETE FROM clusters")
    inserts = []
    for root, idxs in members.items():
        cluster_id = min(numbers[i] for i in idxs)  # deterministic id
        size = len(idxs)
        for i in idxs:
            inserts.append((numbers[i], cluster_id, size))

    conn.executemany(
        "INSERT INTO clusters (number, cluster_id, cluster_size) VALUES (?,?,?)",
        inserts,
    )
    conn.commit()
    return len(members)
