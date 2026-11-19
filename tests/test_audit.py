import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import audit  # noqa: E402


def small_df():
    # Group X: 40 employees, 20 rated 4+; Group Y: 40 employees, 10 rated 4+
    rows = ([("X", 5, 1)] * 10 + [("X", 4, 1)] * 10 + [("X", 3, 0)] * 20
            + [("Y", 4, 1)] * 10 + [("Y", 3, 1)] * 10 + [("Y", 2, 0)] * 20)
    return pd.DataFrame(rows, columns=["team_group", "rating", "met_objectives"])


def test_scale_and_cutoff():
    p = audit.prepare(small_df(), ["team_group"], cutoff=4)
    assert p.scale == [2, 3, 4, 5]
    assert p.has_truth
    r = audit.analyse(p, "team_group")
    g = r.groups.set_index("group")
    assert g.loc["X", "selection_rate"] == pytest.approx(0.5)
    assert g.loc["Y", "selection_rate"] == pytest.approx(0.25)
    assert r.reference == "X"
    assert r.worst_di == pytest.approx(0.5)
    assert r.status == "bad"
    assert r.dpd == pytest.approx(0.25)
    # met objectives: X 20/20 rated high, Y 10/20
    assert g.loc["X", "tpr"] == pytest.approx(1.0)
    assert g.loc["Y", "tpr"] == pytest.approx(0.5)
    assert r.eod == pytest.approx(0.5)
    assert r.mean_gap == pytest.approx((5 * 10 + 4 * 10 + 3 * 20) / 40 - (4 * 10 + 3 * 10 + 2 * 20) / 40)


def test_cutoff_changes_outcome():
    p = audit.prepare(small_df(), ["team_group"], cutoff=3)
    g = audit.analyse(p, "team_group").groups.set_index("group")
    assert g.loc["X", "selection_rate"] == pytest.approx(1.0)
    assert g.loc["Y", "selection_rate"] == pytest.approx(0.5)


def test_binary_rating_column():
    df = small_df().assign(rating=lambda d: (d.rating >= 4).astype(int))
    p = audit.prepare(df, ["team_group"])
    assert p.scale is None and p.cutoff is None
    assert audit.analyse(p, "team_group").worst_di == pytest.approx(0.5)


def test_errors():
    with pytest.raises(ValueError, match="rating"):
        audit.prepare(pd.DataFrame({"gender": ["a", "b"], "score": [1, 2]}))
    with pytest.raises(ValueError, match="cut-off"):
        audit.prepare(small_df(), ["team_group"], cutoff=9)


def test_sample_flags_injected_bias():
    df = audit.generate_sample()
    assert audit.candidate_dims(df) == ["gender", "ethnicity", "age_band"]
    p = audit.prepare(df)
    results = [audit.analyse(p, d) for d in p.dims]
    assert all(r.status == "bad" for r in results)
    recs = audit.recommendations(results, 0.8)
    assert recs[0].priority == "High"
    report = audit.report_markdown(results, p, 0.8, "sample")
    assert "Performance Ratings" in report and "Remediation recommendations" in report
