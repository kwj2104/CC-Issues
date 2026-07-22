"""Config load, validation, and run_id derivation."""

from __future__ import annotations

import hashlib
import json

import yaml

REQUIRED_TOP_KEYS = {
    "repo",
    "weights",
    "severity",
    "demand",
    "clustering",
    "selection",
    "sensitivity",
    "qc",
}

WEIGHT_KEYS = ["reactions", "comments", "velocity", "severity", "demand", "cluster"]


def load_config(path: str) -> dict:
    """Load and validate the YAML config, returning a plain dict."""
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    if not isinstance(cfg, dict):
        raise ValueError("config must be a mapping")
    missing = REQUIRED_TOP_KEYS - set(cfg)
    if missing:
        raise ValueError(f"config missing required keys: {sorted(missing)}")

    weights = cfg["weights"]
    for k in WEIGHT_KEYS:
        if k not in weights:
            raise ValueError(f"weights.{k} is required")
        if not isinstance(weights[k], (int, float)):
            raise ValueError(f"weights.{k} must be numeric")

    sel = cfg["selection"]
    if sel["top_n"] <= 0:
        raise ValueError("selection.top_n must be positive")
    lanes = sel.get("lanes", [])
    if not lanes:
        raise ValueError("selection.lanes is required")
    total_slots = 0
    for lane in lanes:
        for key in ("name", "slots", "rank_by"):
            if key not in lane:
                raise ValueError(f"lane missing '{key}': {lane}")
        if not isinstance(lane["rank_by"], list) or not lane["rank_by"]:
            raise ValueError(f"lane.rank_by must be a non-empty list: {lane}")
        total_slots += lane["slots"]
    if total_slots != sel["top_n"]:
        raise ValueError(
            f"lane slots sum to {total_slots}, must equal top_n {sel['top_n']}"
        )
    if "spill_order" not in sel:
        raise ValueError("selection.spill_order is required")
    thr = cfg["clustering"]["similarity_threshold"]
    if not (0.0 < thr <= 1.0):
        raise ValueError("clustering.similarity_threshold must be in (0, 1]")
    # Labels stored lowercased everywhere; normalise config lists to match.
    lock_reasons = cfg["selection"].setdefault("exclude_lock_reasons", [])
    for lst in (
        cfg["selection"]["exclude_labels"],
        cfg["demand"]["labels"],
        lock_reasons,
    ):
        for i, name in enumerate(lst):
            lst[i] = str(name).lower()
    cfg["severity"]["label_weights"] = {
        str(k).lower(): float(v)
        for k, v in cfg["severity"]["label_weights"].items()
    }


def canonical_config_json(cfg: dict) -> str:
    """Deterministic serialization of the config for hashing/storage."""
    return json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def derive_run_id(snapshot_ts: str, cfg: dict) -> str:
    """run_id = sha256(snapshot_ts + canonical_config_json)[:12]."""
    payload = snapshot_ts + canonical_config_json(cfg)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
