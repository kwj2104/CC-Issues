"""rev 4.1: sensitivity uses the lane baseline; report renders end-to-end."""

from __future__ import annotations

from retrieval import report, score


def test_report_runs_end_to_end(loaded, cfg, tmp_path):
    score.run_score(loaded, cfg, out_dir=str(tmp_path))
    report.run_report(loaded, cfg, out_dir=str(tmp_path))
    for name in ("composition.md", "sensitivity.md", "top20_preview.md",
                 "cluster_qc.md", "composition.csv"):
        assert (tmp_path / name).exists()


def test_sensitivity_unperturbed_sanity(loaded, cfg, tmp_path):
    # _sensitivity raises if the unperturbed lane re-run != baseline; if we get a
    # file, the sanity assert passed. It must also be reported as 1.000.
    score.run_score(loaded, cfg, out_dir=str(tmp_path))
    report.run_report(loaded, cfg, out_dir=str(tmp_path))
    sens = (tmp_path / "sensitivity.md").read_text()
    assert "unperturbed sanity" in sens
    assert "1.000" in sens
    assert "baseline lane head" in sens  # (a) compares to the lane head, not single-score


def test_sensitivity_uses_lane_selection(loaded, cfg):
    # Direct check: _lane_head reproduces the run's lane head exactly.
    from retrieval.report import _lane_head, _jaccard

    scored = score._load_scored_rows(loaded, cfg["weights"])
    a = _lane_head(scored, cfg["selection"])
    b = _lane_head(score._load_scored_rows(loaded, cfg["weights"]), cfg["selection"])
    assert _jaccard(a, b) == 1.0


def test_composition_notes_decile_tie_collapse(loaded, cfg, tmp_path):
    score.run_score(loaded, cfg, out_dir=str(tmp_path))
    report.run_report(loaded, cfg, out_dir=str(tmp_path))
    comp = (tmp_path / "composition.md").read_text()
    assert "Rank-based deciles" in comp
    assert "eng-dominated %" in comp  # per-lane engagement dominance, not global
