"""Fairness metrics and remediation report for AI-assisted hiring decisions.

Pure pandas/numpy, no Streamlit, so it can be tested and reused on its own.

Expected input columns
----------------------
decision   : 1 if the AI recommended / selected the candidate, else 0   (required)
qualified  : 1 if the candidate was actually suitable, else 0             (optional, enables equalised odds)
1-3 demographic columns with 2-12 distinct values each (e.g. gender, ethnicity, age_band)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

MIN_N = 30  # groups smaller than this are reported but excluded from summary figures

DECISION_NAMES = {"decision", "ai_decision", "selected", "shortlisted", "ai_recommended"}
TRUTH_NAMES = {"qualified", "actual", "ground_truth", "label", "true_outcome"}
TRUTHY = {"1", "yes", "y", "true", "selected", "hired", "shortlisted", "pass", "advance", "recommended"}


# --------------------------------------------------------------------------- data
def generate_sample(n: int = 1200, seed: int = 42) -> pd.DataFrame:
    """Synthetic hiring-screen data with bias deliberately injected for demonstration."""
    rng = np.random.default_rng(seed)
    gender = rng.choice(["Male", "Female", "Non-binary"], n, p=[0.52, 0.45, 0.03])
    ethnicity = rng.choice(["Group A", "Group B", "Group C", "Group D"], n, p=[0.45, 0.25, 0.20, 0.10])
    age_band = rng.choice(["Under 30", "30-44", "45+"], n, p=[0.38, 0.42, 0.20])
    qualified = rng.random(n) < 0.42

    p = np.where(qualified, 0.72, 0.10)
    m = np.ones(n)
    m *= np.where(gender == "Female", 0.80, 1)
    m *= np.where(gender == "Non-binary", 0.78, 1)
    m *= np.where(ethnicity == "Group C", 0.78, 1)
    m *= np.where(ethnicity == "Group D", 0.90, 1)
    m *= np.where(age_band == "45+", 0.68, 1)
    m *= np.where(~qualified & (gender == "Male"), 1.4, 1)  # leniency towards unqualified men
    decision = rng.random(n) < np.minimum(0.97, p * m)

    return pd.DataFrame({
        "candidate_id": [f"C{i + 1:04d}" for i in range(n)],
        "gender": gender,
        "ethnicity": ethnicity,
        "age_band": age_band,
        "qualified": qualified.astype(int),
        "decision": decision.astype(int),
    })


def _truthy(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(TRUTHY)


@dataclass
class Prepared:
    df: pd.DataFrame          # columns: dims..., decision (bool), qualified (bool or NA)
    dims: list[str]
    has_truth: bool


def prepare(raw: pd.DataFrame, dims: list[str] | None = None) -> Prepared:
    """Detect columns and normalise a raw decisions table. Raises ValueError with a readable message."""
    if raw.empty:
        raise ValueError("The file has no data rows.")
    cols = {c.strip().lower(): c for c in raw.columns}
    dec = next((cols[c] for c in cols if c in DECISION_NAMES), None)
    if dec is None:
        raise ValueError('No "decision" column found. Add one with 1 = selected/recommended by the AI and 0 = not.')
    truth = next((cols[c] for c in cols if c in TRUTH_NAMES), None)

    if dims is None:
        dims = [c for c in candidate_dims(raw)][:3]
    if not dims:
        raise ValueError("No demographic columns found. Each needs 2-12 distinct values (e.g. gender, ethnicity, age_band).")

    df = pd.DataFrame({d: raw[d].astype(str).str.strip().replace({"": "Unknown", "nan": "Unknown"}) for d in dims})
    df["decision"] = _truthy(raw[dec])
    # without a ground-truth column, qualified is a placeholder and TPR/FPR are reported as missing
    df["qualified"] = _truthy(raw[truth]) if truth is not None else False
    return Prepared(df=df, dims=list(dims), has_truth=truth is not None)


def candidate_dims(raw: pd.DataFrame) -> list[str]:
    """Columns that look like categorical demographic attributes."""
    out = []
    for c in raw.columns:
        lc = c.strip().lower()
        if lc in DECISION_NAMES or lc in TRUTH_NAMES:
            continue
        if lc == "id" or lc.endswith("_id") or lc.startswith("id_") or any(k in lc for k in ("name", "email", "score")):
            continue
        if 2 <= raw[c].nunique(dropna=True) <= 12:
            out.append(c)
    return out


# --------------------------------------------------------------------------- metrics
@dataclass
class DimResult:
    dim: str
    groups: pd.DataFrame      # one row per group
    reference: str
    dpd: float                # demographic parity difference
    tpr_gap: float | None
    fpr_gap: float | None
    eod: float | None         # equalised odds difference
    worst_group: str
    worst_di: float
    status: str               # "bad" | "warn" | "ok"


def group_status(row, thr: float) -> str:
    if row["n"] < MIN_N:
        return "small"
    if row["impact_ratio"] < thr:
        return "bad"
    if row["impact_ratio"] < 0.9:
        return "warn"
    return "ok"


STATUS_TEXT = {"bad": "Adverse impact", "warn": "Monitor", "ok": "Within threshold", "small": "Low sample"}


def analyse(p: Prepared, dim: str, thr: float = 0.8) -> DimResult:
    df = p.df
    q = df["qualified"].astype(bool)
    d = df["decision"].astype(bool)
    agg = df.assign(tp=q & d, fn=q & ~d, fp=~q & d, tn=~q & ~d).groupby(dim).agg(n=("decision", "size"), selected=("decision", "sum"),
                       tp=("tp", "sum"), fn=("fn", "sum"), fp=("fp", "sum"), tn=("tn", "sum"))
    agg = agg.sort_values("n", ascending=False)
    agg["selection_rate"] = agg["selected"] / agg["n"]
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


# --------------------------------------------------------------------------- report
def label(s: str) -> str:
    return s.replace("_", " ").title()


def pct(x) -> str:
    return "—" if x is None or pd.isna(x) else f"{x * 100:.1f}%"


def r2(x) -> str:
    return "—" if x is None or pd.isna(x) else f"{x:.2f}"


@dataclass
class Rec:
    priority: str  # High | Medium | Low
    title: str
    finding: str
    actions: list[str] = field(default_factory=list)


def recommendations(results: list[DimResult], thr: float) -> list[Rec]:
    recs: list[Rec] = []
    for r in results:
        L = label(r.dim)
        g = r.groups
        big = g[g["n"] >= MIN_N]
        ref_rate = g.loc[g["group"] == r.reference, "selection_rate"].iloc[0]
        bad = big[big["impact_ratio"] < thr]
        warn = big[(big["impact_ratio"] >= thr) & (big["impact_ratio"] < 0.9)]

        if len(bad):
            names = ", ".join(bad["group"])
            recs.append(Rec("High", f"Adverse impact in {L}",
                "; ".join(f"{x.group} ({pct(x.selection_rate)}, ratio {r2(x.impact_ratio)})" for x in bad.itertuples())
                + f" fall below {thr} of the reference group {r.reference} ({pct(ref_rate)}).",
                [f"Route AI rejections of {names} applicants to a blind human review before any final decision.",
                 f"Audit model inputs for proxies of {L.lower()} (e.g. career gaps, graduation year, names, location, school) "
                 "and remove or reweight them in the screening rules.",
                 "Re-validate that each screening criterion is job-related and document the justification for any criterion that drives the gap.",
                 "Do not apply group-specific cut-offs without legal review; prefer criteria and process fixes."]))
        elif len(warn):
            recs.append(Rec("Low", f"Watch {L}",
                "; ".join(f"{x.group} (ratio {r2(x.impact_ratio)})" for x in warn.itertuples())
                + " are within the threshold but below 0.90.",
                ["Track these groups in each audit cycle and investigate if the ratio declines further."]))

        if r.tpr_gap is not None and r.tpr_gap > 0.10:
            hi, lo = big.loc[big["tpr"].idxmax()], big.loc[big["tpr"].idxmin()]
            recs.append(Rec("High" if r.tpr_gap > 0.15 else "Medium",
                f"Qualified {lo.group} applicants are missed more often ({L})",
                f"True positive rate is {pct(lo.tpr)} for {lo.group} versus {pct(hi.tpr)} for {hi.group} (gap {pct(r.tpr_gap)}).",
                [f"Sample false negatives for {lo.group} and identify which criteria rejected qualified applicants.",
                 "Add a second-look queue for borderline AI scores in affected groups until the gap closes."]))

        if r.fpr_gap is not None and r.fpr_gap > 0.10:
            hi, lo = big.loc[big["fpr"].idxmax()], big.loc[big["fpr"].idxmin()]
            recs.append(Rec("Medium", f"Unqualified {hi.group} applicants advance more often ({L})",
                f"False positive rate is {pct(hi.fpr)} for {hi.group} versus {pct(lo.fpr)} for {lo.group} "
                f"(gap {pct(r.fpr_gap)}), suggesting standards are applied unevenly.",
                ["Review a sample of advanced-but-unqualified applicants to find the leniency pattern.",
                 "Standardise interviewer and screener rubrics for the next stage."]))

        small = g[g["n"] < MIN_N]
        if len(small):
            recs.append(Rec("Low", f"Insufficient data for some {L} groups",
                ", ".join(f"{x.group} (n={x.n})" for x in small.itertuples())
                + f" {'have' if len(small) > 1 else 'has'} fewer than {MIN_N} records, so ratios are unreliable.",
                ["Pool several hiring cycles before drawing conclusions, and review these decisions manually in the meantime."]))

    recs.append(Rec("Medium", "Governance and monitoring",
        "Bias can re-emerge as applicant pools and job requirements change.",
        ["Re-run this audit every hiring cycle (at least quarterly) and keep each report on file.",
         "Keep a human decision-maker accountable for every final hiring outcome.",
         "Share results with HR leadership and legal/compliance, and record what changed in response."]))
    order = {"High": 0, "Medium": 1, "Low": 2}
    return sorted(recs, key=lambda x: order[x.priority])


def summary_text(results: list[DimResult], thr: float) -> str:
    flagged = [r for r in results if r.status == "bad"]
    if not flagged:
        return f"No dimension falls below the {thr} impact-ratio threshold."
    worst = min(results, key=lambda r: r.worst_di)
    return (f"The AI screening shows adverse impact in {len(flagged)} of {len(results)} dimensions "
            f"({', '.join(label(r.dim) for r in flagged)}). The lowest impact ratio is {r2(worst.worst_di)} "
            f"({worst.worst_group}, {label(worst.dim)}).")


def method_text(thr: float, has_truth: bool) -> str:
    eo = ("Equalised odds difference is the larger of the gaps in true positive and false positive rates, "
          "using the qualified column as ground truth." if has_truth
          else "Equalised odds was not computed because no ground-truth column was provided.")
    return (f"Selection rates were compared across groups in each dimension. The impact ratio divides each group's "
            f"rate by the highest-rate group (four-fifths rule, threshold {thr}). Demographic parity difference is the "
            f"gap between the highest and lowest selection rates. {eo} Groups with fewer than {MIN_N} records are "
            f"excluded from summary figures. Model retraining is out of scope; recommendations target process and criteria.")


def report_markdown(results: list[DimResult], p: Prepared, thr: float, source: str) -> str:
    overall = p.df["decision"].mean()
    lines = [
        "# AI Bias Audit Report — Hiring", "",
        f"{date.today().isoformat()} · {source} · {len(p.df):,} decisions · Overall selection rate {pct(overall)}", "",
        "## Summary", "", summary_text(results, thr), "",
        "## Method", "", method_text(thr, p.has_truth), "",
        "## Results by dimension",
    ]
    for r in results:
        lines += ["", f"### {label(r.dim)} — {STATUS_TEXT[r.status]}", "",
                  f"Demographic parity difference {pct(r.dpd)} · Equalised odds difference {pct(r.eod)}", "",
                  "| Group | n | Selection rate | Impact ratio | TPR | FPR |", "|---|---|---|---|---|---|"]
        lines += [f"| {x.group} | {x.n} | {pct(x.selection_rate)} | {r2(x.impact_ratio)} | {pct(x.tpr)} | {pct(x.fpr)} |"
                  for x in r.groups.itertuples()]
    lines += ["", "## Remediation recommendations"]
    for rec in recommendations(results, thr):
        lines += ["", f"### [{rec.priority}] {rec.title}", "", rec.finding, ""] + [f"- {a}" for a in rec.actions]
    return "\n".join(lines) + "\n"
