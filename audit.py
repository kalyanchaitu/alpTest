"""Fairness metrics and remediation report for AI-suggested performance ratings.

Pure pandas/numpy (no Streamlit), so it can be tested and reused on its own, e.g. in a notebook.

Expected input (one row per employee)
-------------------------------------
ai_score / ai_rating                  : score (0-100) or band suggested by the AI tool         (required: one of them)
management_score / management_rating  : score or band given by management (fair benchmark)     (optional, enables
                                                                                                 equalised odds)
1-3 demographic columns with 2-12 distinct values each, e.g. gender, age_group, region.

Older files with a 0/1 `decision` (AI outcome) and `qualified` (ground truth) column still work.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

MIN_N = 30            # groups smaller than this are shown but excluded from summary figures
HIGH_CUTOFF = 80      # score at or above which a rating counts as a top rating (Exceeds or Outstanding)

BANDS = ["Outstanding", "Exceeds Expectations", "Meets Expectations", "Needs Improvement", "Unsatisfactory"]
BAND_CUTS = [(90, "Outstanding"), (80, "Exceeds Expectations"), (70, "Meets Expectations"), (60, "Needs Improvement")]
TOP_BANDS = {
    "Exceeds or Outstanding": ("Outstanding", "Exceeds Expectations"),
    "Outstanding only": ("Outstanding",),
}

AI_SCORE = {"ai_score", "ai_suggested_score"}
AI_RATING = {"ai_rating", "ai_suggested_rating", "ai_band"}
MGMT_SCORE = {"management_score", "manager_score", "mgmt_score"}
MGMT_RATING = {"management_rating", "manager_rating", "mgmt_rating"}
AI_FLAG = {"ai_high_rating", "decision", "ai_decision", "selected", "ai_recommended"}
MGMT_FLAG = {"mgmt_high_rating", "qualified", "actual", "ground_truth", "label"}
OUTCOME_COLS = AI_SCORE | AI_RATING | MGMT_SCORE | MGMT_RATING | AI_FLAG | MGMT_FLAG
PREFERRED_DIMS = ["gender", "age_group", "region"]   # decision D-003
TRUTHY = {"1", "yes", "y", "true", "high", "top"}

DIM_PROXIES = {
    "gender": "leave or career gaps, part-time status, working-hours data, or wording in manager comments",
    "age_group": "tenure, years of experience, salary level, training hours, or date-based fields",
    "region": "site or location codes, region-specific KPI targets, or differences in data quality between sites",
    "department": "KPI scales that differ between functions, or targets that are harder to hit in some functions",
}


# --------------------------------------------------------------------------- helpers
def label(s: str) -> str:
    return str(s).replace("_", " ").title()


def pct(x) -> str:
    return "—" if x is None or pd.isna(x) else f"{x * 100:.1f}%"


def r2(x) -> str:
    return "—" if x is None or pd.isna(x) else f"{x:.2f}"


def pts(x) -> str:
    return "—" if x is None or pd.isna(x) else f"{x:+.1f} pts"


def band_of(score: pd.Series) -> pd.Series:
    s = pd.to_numeric(score, errors="coerce")
    out = pd.Series("Unsatisfactory", index=s.index, dtype=object)
    for cut, name in reversed(BAND_CUTS):
        out[s >= cut] = name
    return out.where(s.notna())


def _find(cols: dict, names: set):
    return next((cols[c] for c in cols if c in names), None)


def _truthy(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(TRUTHY)


# --------------------------------------------------------------------------- data
@dataclass
class Prepared:
    df: pd.DataFrame      # dims..., decision (AI top rating), qualified (mgmt top rating), scores/bands if present
    dims: list[str]
    has_truth: bool       # management benchmark present -> equalised odds
    has_scores: bool      # both scores present -> average score gap
    has_bands: bool       # both rating bands available -> rating distribution
    top_label: str


def candidate_dims(raw: pd.DataFrame) -> list[str]:
    """Columns that look like demographic or organisational groupings, preferred ones first."""
    out = []
    for c in raw.columns:
        lc = c.strip().lower()
        if lc in OUTCOME_COLS or "score" in lc or "rating" in lc:
            continue
        if lc == "id" or lc.endswith("_id") or lc.startswith("id_") or "name" in lc or "email" in lc:
            continue
        if 2 <= raw[c].nunique(dropna=True) <= 12:
            out.append(c)
    pref = [c for p in PREFERRED_DIMS for c in out if c.strip().lower() == p]
    return pref + [c for c in out if c not in pref]


def prepare(raw: pd.DataFrame, dims: list[str] | None = None, top: str = "Exceeds or Outstanding") -> Prepared:
    """Detect columns and normalise a ratings table. Raises ValueError with a readable message."""
    if raw.empty:
        raise ValueError("The file has no data rows.")
    cols = {c.strip().lower(): c for c in raw.columns}
    top_bands = TOP_BANDS[top]
    min_score = 90 if top == "Outstanding only" else HIGH_CUTOFF

    def outcome(score_names, rating_names, flag_names):
        sc, rt, fl = _find(cols, score_names), _find(cols, rating_names), _find(cols, flag_names)
        score = pd.to_numeric(raw[sc], errors="coerce") if sc else None
        rating = raw[rt].astype(str).str.strip() if rt else (band_of(score) if score is not None else None)
        if rating is not None:
            return rating.isin(top_bands), score, rating
        if fl:
            return _truthy(raw[fl]), None, None
        return None, None, None

    ai_flag, ai_score, ai_rating = outcome(AI_SCORE, AI_RATING, AI_FLAG)
    if ai_flag is None:
        raise ValueError("No AI rating found. Add an `ai_score` (0-100) or `ai_rating` (band) column.")
    m_flag, m_score, m_rating = outcome(MGMT_SCORE, MGMT_RATING, MGMT_FLAG)

    if dims is None:
        dims = candidate_dims(raw)[:3]
    if not dims:
        raise ValueError("No demographic columns found. Each needs 2-12 distinct values (e.g. gender, age_group, region).")

    df = pd.DataFrame({d: raw[d].astype(str).str.strip().replace({"": "Unknown", "nan": "Unknown"}) for d in dims})
    df["decision"] = ai_flag.to_numpy()
    df["qualified"] = m_flag.to_numpy() if m_flag is not None else False
    df["ai_score"] = ai_score.to_numpy() if ai_score is not None else np.nan
    df["mgmt_score"] = m_score.to_numpy() if m_score is not None else np.nan
    df["ai_rating"] = ai_rating.to_numpy() if ai_rating is not None else None
    df["mgmt_rating"] = m_rating.to_numpy() if m_rating is not None else None
    return Prepared(df=df, dims=list(dims), has_truth=m_flag is not None,
                    has_scores=ai_score is not None and m_score is not None,
                    has_bands=ai_rating is not None and m_rating is not None, top_label=top)


# --------------------------------------------------------------------------- metrics
@dataclass
class DimResult:
    dim: str
    groups: pd.DataFrame      # one row per group
    reference: str
    dpd: float                # demographic parity difference (AI top-rating rates)
    tpr_gap: float | None
    fpr_gap: float | None
    eod: float | None         # equalised odds difference
    worst_group: str
    worst_di: float
    status: str               # "bad" | "warn" | "ok"


STATUS_TEXT = {"bad": "Adverse impact", "warn": "Monitor", "ok": "Within threshold", "small": "Low sample"}


def group_status(row, thr: float) -> str:
    if row["n"] < MIN_N:
        return "small"
    if row["impact_ratio"] < thr:
        return "bad"
    if row["impact_ratio"] < 0.9:
        return "warn"
    return "ok"


def analyse(p: Prepared, dim: str, thr: float = 0.8) -> DimResult:
    q = p.df["qualified"].astype(bool)
    d = p.df["decision"].astype(bool)
    agg = p.df.assign(tp=q & d, fn=q & ~d, fp=~q & d, tn=~q & ~d, gap=p.df["ai_score"] - p.df["mgmt_score"]) \
        .groupby(dim).agg(n=("decision", "size"), selected=("decision", "sum"), mgmt_top=("qualified", "sum"),
                          tp=("tp", "sum"), fn=("fn", "sum"), fp=("fp", "sum"), tn=("tn", "sum"),
                          ai_avg=("ai_score", "mean"), mgmt_avg=("mgmt_score", "mean"), score_gap=("gap", "mean"))
    agg = agg.sort_values("n", ascending=False)
    agg["selection_rate"] = agg["selected"] / agg["n"]
    agg["mgmt_rate"] = agg["mgmt_top"] / agg["n"] if p.has_truth else np.nan
    if p.has_truth:
        agg["tpr"] = agg["tp"] / (agg["tp"] + agg["fn"]).replace(0, np.nan)
        agg["fpr"] = agg["fp"] / (agg["fp"] + agg["tn"]).replace(0, np.nan)
    else:
        agg["tpr"] = np.nan
        agg["fpr"] = np.nan

    elig = agg[agg["n"] >= MIN_N]
    if elig.empty:
        elig = agg
    ref = elig["selection_rate"].idxmax()
    ref_rate = agg.loc[ref, "selection_rate"]
    agg["impact_ratio"] = agg["selection_rate"] / ref_rate if ref_rate else np.nan
    agg["status"] = agg.apply(group_status, axis=1, thr=thr)
    elig = agg.loc[elig.index]

    def gap(col):
        v = elig[col].dropna()
        return float(v.max() - v.min()) if len(v) > 1 else None

    tpr_gap, fpr_gap = (gap("tpr"), gap("fpr")) if p.has_truth else (None, None)
    eod = None if tpr_gap is None and fpr_gap is None else max(tpr_gap or 0, fpr_gap or 0)
    worst = elig["impact_ratio"].idxmin()
    worst_di = float(elig.loc[worst, "impact_ratio"])
    status = "bad" if worst_di < thr else "warn" if worst_di < 0.9 else "ok"

    return DimResult(dim=dim, groups=agg.reset_index().rename(columns={dim: "group"}), reference=str(ref),
                     dpd=float(elig["selection_rate"].max() - elig["selection_rate"].min()),
                     tpr_gap=tpr_gap, fpr_gap=fpr_gap, eod=eod,
                     worst_group=str(worst), worst_di=worst_di, status=status)


def rating_distribution(p: Prepared, dim: str) -> pd.DataFrame:
    """Share of each rating band by group, for the AI and for management (long format)."""
    parts = []
    for src, col in (("Manager", "mgmt_rating"), ("AI", "ai_rating")):
        t = p.df.groupby(dim)[col].value_counts(normalize=True).rename("share").reset_index()
        parts.append(t.rename(columns={dim: "group", col: "band"}).assign(source=src))
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------- report
@dataclass
class Rec:
    priority: str  # High | Medium | Low
    title: str
    finding: str
    actions: list[str] = field(default_factory=list)


def recommendations(results: list[DimResult], thr: float, p: Prepared) -> list[Rec]:
    recs: list[Rec] = []
    top = p.top_label
    for r in results:
        L = label(r.dim)
        g = r.groups
        big = g[g["n"] >= MIN_N]
        ref = g.loc[g["group"] == r.reference].iloc[0]
        bad = big[big["impact_ratio"] < thr]
        warn = big[(big["impact_ratio"] >= thr) & (big["impact_ratio"] < 0.9)]
        proxies = DIM_PROXIES.get(r.dim.strip().lower(), f"any input that is closely linked to {L.lower()}")

        if len(bad):
            names = ", ".join(bad["group"])
            finding = "; ".join(f"{x.group} employees get a top AI rating {pct(x.selection_rate)} of the time (ratio {r2(x.impact_ratio)})"
                                for x in bad.itertuples())
            finding += f", compared with {pct(ref.selection_rate)} for {r.reference}."
            if p.has_truth:
                finding += " Managers rated the same groups top at " + "; ".join(
                    f"{pct(x.mgmt_rate)} ({x.group})" for x in bad.itertuples()) + f" vs {pct(ref.mgmt_rate)} ({r.reference})"
                finding += ", so the gap comes from the AI, not from performance." if \
                    (bad["mgmt_rate"] / ref.mgmt_rate).min() >= thr else "."
            if p.has_scores:
                finding += " On average the AI scores " + "; ".join(
                    f"{x.group} {pts(x.score_gap)}" for x in bad.itertuples()) + \
                    f" against the manager's score, vs {pts(ref.score_gap)} for {r.reference}."
            recs.append(Rec("High", f"AI under-rates {names} ({L})", finding, [
                f"Hold AI-suggested ratings for {names} employees for a manager calibration review before they are final.",
                f"Check the AI's inputs for stand-ins for {L.lower()}, such as {proxies}, and remove or correct them.",
                "Compare the AI's KPI, competency and behaviour scores for these groups with the managers' to find where the gap enters.",
                "Do not add group-based score adjustments without HR and legal review; fix the inputs and the process instead."]))
        elif len(warn):
            recs.append(Rec("Low", f"Watch {L}",
                "; ".join(f"{x.group} (ratio {r2(x.impact_ratio)})" for x in warn.itertuples())
                + " are within the threshold but below 0.90.",
                ["Track these groups in each appraisal cycle and investigate if the ratio falls further."]))

        if r.tpr_gap is not None and r.tpr_gap > 0.10:
            hi, lo = big.loc[big["tpr"].idxmax()], big.loc[big["tpr"].idxmin()]
            recs.append(Rec("High" if r.tpr_gap > 0.15 else "Medium",
                f"{lo.group} top performers are missed more often ({L})",
                f"Of the employees managers rated {top}, the AI also rated {pct(lo.tpr)} of {lo.group} employees that high, "
                f"against {pct(hi.tpr)} of {hi.group} (gap {pct(r.tpr_gap)}).",
                [f"Re-review the {lo.group} top performers the AI under-rated before ratings feed pay or promotion decisions.",
                 "Flag every case where the AI and the manager are a full band apart for a second look."]))

        if r.fpr_gap is not None and r.fpr_gap > 0.10:
            hi, lo = big.loc[big["fpr"].idxmax()], big.loc[big["fpr"].idxmin()]
            recs.append(Rec("Medium", f"{hi.group} employees get unearned top ratings more often ({L})",
                f"Of the employees managers did not rate {top}, the AI still rated {pct(hi.fpr)} of {hi.group} employees that high, "
                f"against {pct(lo.fpr)} of {lo.group} (gap {pct(r.fpr_gap)}).",
                [f"Review top AI ratings for {hi.group} where the manager disagreed.",
                 "Require the same evidence for a top rating in every group."]))

        small = g[g["n"] < MIN_N]
        if len(small):
            recs.append(Rec("Low", f"Too few employees in some {L} groups",
                ", ".join(f"{x.group} (n={x.n})" for x in small.itertuples())
                + f" {'have' if len(small) > 1 else 'has'} fewer than {MIN_N} employees, so their ratios are unreliable.",
                ["Pool several appraisal cycles before drawing conclusions, and review these ratings manually meanwhile."]))

    recs.append(Rec("Medium", "Governance and monitoring",
        "Bias can return as the workforce, targets and the AI tool change.",
        ["Re-run this audit every appraisal cycle before ratings are finalised, and keep each report on file.",
         "Treat AI ratings as suggestions: a named manager stays accountable for every final rating, and overrides are recorded.",
         "Give employees a clear way to ask for a review of their rating.",
         "Share results with HR leadership and legal/compliance, and record what changed in response."]))
    order = {"High": 0, "Medium": 1, "Low": 2}
    return sorted(recs, key=lambda x: order[x.priority])


def summary_text(results: list[DimResult], thr: float, p: Prepared) -> str:
    flagged = [r for r in results if r.status == "bad"]
    if not flagged:
        return f"No dimension falls below the {thr} impact-ratio threshold for AI {p.top_label.lower()} ratings."
    worst = min(results, key=lambda r: r.worst_di)
    return (f"The AI rating tool shows adverse impact in {len(flagged)} of {len(results)} dimensions "
            f"({', '.join(label(r.dim) for r in flagged)}). The lowest impact ratio is {r2(worst.worst_di)} "
            f"({worst.worst_group}, {label(worst.dim)}).")


def method_text(thr: float, p: Prepared) -> str:
    eo = ("Equalised odds compares the AI rating with the management rating, which serves as the benchmark of actual "
          "performance: the true positive rate is the share of employees managers rated top whom the AI also rated top, "
          "and the false positive rate is the share of the rest whom the AI rated top anyway. The equalised odds "
          "difference is the larger of the two gaps between groups." if p.has_truth
          else "Equalised odds was not computed because no management rating was provided.")
    return (f"A top rating means {p.top_label} (score {90 if p.top_label == 'Outstanding only' else HIGH_CUTOFF}+). "
            f"For each dimension, the share of employees the AI gave a top rating was compared across groups. The impact "
            f"ratio divides each group's rate by the highest group's rate (four-fifths rule, threshold {thr}). "
            f"Demographic parity difference is the gap between the highest and lowest rates. {eo} Groups with fewer than "
            f"{MIN_N} employees are excluded from summary figures. Retraining the AI model is out of scope; recommendations "
            f"target the rating process, inputs and governance.")


def report_markdown(results: list[DimResult], p: Prepared, thr: float, source: str) -> str:
    lines = [
        "# AI Bias Audit Report — Performance Ratings", "",
        f"{date.today().isoformat()} · {source} · {len(p.df):,} employees · "
        f"AI {p.top_label.lower()} rate {pct(p.df['decision'].mean())}"
        + (f" · Management {pct(p.df['qualified'].mean())}" if p.has_truth else ""), "",
        "## Summary", "", summary_text(results, thr, p), "",
        "## Method", "", method_text(thr, p), "",
        "## Results by dimension",
    ]
    for r in results:
        lines += ["", f"### {label(r.dim)} — {STATUS_TEXT[r.status]}", "",
                  f"Demographic parity difference {pct(r.dpd)} · Equalised odds difference {pct(r.eod)}", "",
                  "| Group | Employees | AI top rate | Mgmt top rate | Impact ratio | AI − mgmt score | TPR | FPR |",
                  "|---|---|---|---|---|---|---|---|"]
        lines += [f"| {x.group} | {x.n} | {pct(x.selection_rate)} | {pct(x.mgmt_rate)} | {r2(x.impact_ratio)} | "
                  f"{pts(x.score_gap)} | {pct(x.tpr)} | {pct(x.fpr)} |" for x in r.groups.itertuples()]
    lines += ["", "## Remediation recommendations"]
    for rec in recommendations(results, thr, p):
        lines += ["", f"### [{rec.priority}] {rec.title}", "", rec.finding, ""] + [f"- {a}" for a in rec.actions]
    return "\n".join(lines) + "\n"
