"""Fairness metrics and remediation report for AI-assisted performance ratings.

Pure pandas/numpy, no Streamlit, so it can be tested and reused on its own.

Expected input columns
----------------------
rating          : the AI-suggested performance rating, either on a numeric scale (e.g. 1-5)
                  or already binary (1 = high rating, 0 = not)                          (required)
met_objectives  : 1 if the employee met or exceeded their objective targets, else 0     (optional,
                  used as ground truth for equalised odds)
1-3 demographic columns with 2-12 distinct values each (e.g. gender, ethnicity, age_band)

On a numeric scale, a rating at or above the cut-off (default 4, "Exceeds expectations") counts as the
favourable outcome, since high ratings drive pay rises, bonuses and promotion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

MIN_N = 30            # groups smaller than this are reported but excluded from summary figures
DEFAULT_CUTOFF = 4    # ratings >= this count as "high"
MEAN_GAP_FLAG = 0.30  # gap in average rating (scale points) worth reporting

RATING_NAMES = {"rating", "ai_rating", "performance_rating", "perf_rating", "ai_performance_rating",
                "high_rating", "decision", "ai_decision"}
TRUTH_NAMES = {"met_objectives", "objectives_met", "met_targets", "kpi_met", "goals_met", "high_performer",
               "qualified", "ground_truth", "actual", "true_outcome", "label"}
TRUTHY = {"1", "1.0", "yes", "y", "true", "high", "met", "exceeds", "pass"}

SCALE_LABELS_5 = {1: "1 · Below expectations", 2: "2 · Partially meets", 3: "3 · Meets",
                  4: "4 · Exceeds", 5: "5 · Outstanding"}


# --------------------------------------------------------------------------- data
def generate_sample(n: int = 1500, seed: int = 7) -> pd.DataFrame:
    """Synthetic annual-review data (1-5 AI-suggested ratings) with bias deliberately injected."""
    rng = np.random.default_rng(seed)
    gender = rng.choice(["Male", "Female", "Non-binary"], n, p=[0.50, 0.46, 0.04])
    ethnicity = rng.choice(["Group A", "Group B", "Group C", "Group D"], n, p=[0.45, 0.25, 0.20, 0.10])
    age_band = rng.choice(["Under 30", "30-44", "45+"], n, p=[0.30, 0.45, 0.25])
    met = rng.random(n) < 0.40  # objective results: goals met or exceeded

    # unbiased rating given objective results
    rating = np.where(met, rng.choice([2, 3, 4, 5], n, p=[0.03, 0.25, 0.50, 0.22]),
                      rng.choice([1, 2, 3, 4], n, p=[0.12, 0.33, 0.47, 0.08]))

    # injected bias: chance of being marked down one point
    down = np.zeros(n)
    down += np.where(gender == "Female", 0.20, 0)
    down += np.where(gender == "Non-binary", 0.28, 0)
    down += np.where(ethnicity == "Group C", 0.24, 0)
    down += np.where(ethnicity == "Group D", 0.08, 0)
    down += np.where(age_band == "45+", 0.30, 0)
    rating = rating - (rng.random(n) < np.minimum(down, 0.9))
    # leniency: men who missed objectives are bumped up more often
    rating = rating + ((~met) & (gender == "Male") & (rng.random(n) < 0.18))
    rating = np.clip(rating, 1, 5)

    return pd.DataFrame({
        "employee_id": [f"E{i + 1:04d}" for i in range(n)],
        "gender": gender,
        "ethnicity": ethnicity,
        "age_band": age_band,
        "met_objectives": met.astype(int),
        "rating": rating.astype(int),
    })


def _truthy(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(TRUTHY)


def find_column(raw: pd.DataFrame, names: set[str]) -> str | None:
    return next((c for c in raw.columns if c.strip().lower() in names), None)


def rating_scale(raw: pd.DataFrame) -> list[float] | None:
    """Sorted distinct values of the rating column if it is a numeric scale (3+ levels), else None."""
    col = find_column(raw, RATING_NAMES)
    if col is None:
        return None
    num = pd.to_numeric(raw[col], errors="coerce")
    if num.notna().mean() < 0.95:
        return None
    vals = sorted(num.dropna().unique().tolist())
    return vals if 3 <= len(vals) <= 11 else None


@dataclass
class Prepared:
    df: pd.DataFrame          # columns: dims..., rating (float or NA), decision (bool), qualified (bool)
    dims: list[str]
    has_truth: bool
    scale: list[float] | None  # rating levels if numeric, None if the input was already binary
    cutoff: float | None
    dropped: int = 0           # rows without a usable rating


def prepare(raw: pd.DataFrame, dims: list[str] | None = None, cutoff: float = DEFAULT_CUTOFF) -> Prepared:
    """Detect columns and normalise a raw ratings table. Raises ValueError with a readable message."""
    if raw.empty:
        raise ValueError("The file has no data rows.")
    col = find_column(raw, RATING_NAMES)
    if col is None:
        raise ValueError('No "rating" column found. Add one with the AI-suggested performance rating '
                         "(e.g. 1-5), or 1 = high rating and 0 = not.")
    truth = find_column(raw, TRUTH_NAMES)

    if dims is None:
        dims = candidate_dims(raw)[:3]
    if not dims:
        raise ValueError("No demographic columns found. Each needs 2-12 distinct values (e.g. gender, ethnicity, age_band).")

    scale = rating_scale(raw)
    dropped = 0
    if scale is not None:
        num = pd.to_numeric(raw[col], errors="coerce")
        keep = num.notna()
        dropped = int((~keep).sum())
        raw, num = raw[keep], num[keep]
        if not (scale[0] < cutoff <= scale[-1]):
            raise ValueError(f"The high-rating cut-off {cutoff:g} must be above the lowest rating "
                             f"({scale[0]:g}) and no higher than the top rating ({scale[-1]:g}).")
        decision, rating = num >= cutoff, num.astype(float)
    else:
        decision, rating, cutoff = _truthy(raw[col]), pd.Series(np.nan, index=raw.index), None

    df = pd.DataFrame({d: raw[d].astype(str).str.strip().replace({"": "Unknown", "nan": "Unknown"}) for d in dims})
    df["rating"] = rating
    df["decision"] = decision
    # without a ground-truth column, qualified is a placeholder and TPR/FPR are reported as missing
    df["qualified"] = _truthy(raw[truth]) if truth is not None else False
    return Prepared(df=df, dims=list(dims), has_truth=truth is not None, scale=scale, cutoff=cutoff, dropped=dropped)


def candidate_dims(raw: pd.DataFrame) -> list[str]:
    """Columns that look like categorical demographic attributes."""
    out = []
    for c in raw.columns:
        lc = c.strip().lower()
        if lc in RATING_NAMES or lc in TRUTH_NAMES:
            continue
        if lc == "id" or lc.endswith("_id") or lc.startswith("id_") or any(
                k in lc for k in ("name", "email", "score", "rating", "salary", "bonus")):
            continue
        if 2 <= raw[c].nunique(dropna=True) <= 12:
            out.append(c)
    return out


def scale_label(v: float, scale: list[float]) -> str:
    if [int(x) for x in scale] == [1, 2, 3, 4, 5] and float(v).is_integer():
        return SCALE_LABELS_5[int(v)]
    return f"{v:g}"


def distribution(p: Prepared, dim: str) -> pd.DataFrame | None:
    """Share of each rating level within each group (rows = groups, columns = rating levels)."""
    if p.scale is None:
        return None
    return pd.crosstab(p.df[dim], p.df["rating"], normalize="index").reindex(columns=p.scale, fill_value=0)


# --------------------------------------------------------------------------- metrics
@dataclass
class DimResult:
    dim: str
    groups: pd.DataFrame      # one row per group
    reference: str
    dpd: float                # demographic parity difference (high-rating rate)
    tpr_gap: float | None
    fpr_gap: float | None
    eod: float | None         # equalised odds difference
    mean_gap: float | None    # highest minus lowest average rating (scale points)
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
    agg = df.assign(tp=q & d, fn=q & ~d, fp=~q & d, tn=~q & ~d).groupby(dim).agg(
        n=("decision", "size"), selected=("decision", "sum"), mean_rating=("rating", "mean"),
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
                     mean_gap=gap("mean_rating") if p.scale is not None else None,
                     worst_group=str(worst), worst_di=worst_di, status=status)


# --------------------------------------------------------------------------- report
def label(s: str) -> str:
    return s.replace("_", " ").title()


def pct(x) -> str:
    return "—" if x is None or pd.isna(x) else f"{x * 100:.1f}%"


def r2(x) -> str:
    return "—" if x is None or pd.isna(x) else f"{x:.2f}"


def high_text(p: Prepared) -> str:
    """What counts as the favourable outcome, in words."""
    if p.cutoff is None:
        return "a high rating"
    top = p.scale[-1] if p.scale else None
    return f"a rating of {p.cutoff:g}" + ("" if p.cutoff == top else " or above")


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
        mean_note = ""
        if r.mean_gap is not None and r.mean_gap >= MEAN_GAP_FLAG:
            lo, hi = big.loc[big["mean_rating"].idxmin()], big.loc[big["mean_rating"].idxmax()]
            mean_note = (f" Average rating is {lo.mean_rating:.2f} for {lo.group} versus "
                         f"{hi.mean_rating:.2f} for {hi.group}.")

        if len(bad):
            names = ", ".join(bad["group"])
            recs.append(Rec("High", f"Adverse impact in {L}",
                "; ".join(f"{x.group} ({pct(x.selection_rate)} rated high, ratio {r2(x.impact_ratio)})"
                          for x in bad.itertuples())
                + f" fall below {thr} of the reference group {r.reference} ({pct(ref_rate)})." + mean_note,
                [f"Hold a calibration review of AI-suggested ratings for {names} employees before ratings are "
                 "finalised, checking each against objective results (goals met, output, quality).",
                 f"Audit the AI's inputs for proxies of {L.lower()} (e.g. part-time or flexible working, leave of "
                 "absence, tenure, location, wording in manager comments or peer feedback) and remove or down-weight them.",
                 "Tie every rating criterion to documented job objectives; redefine or drop subjective criteria "
                 "(e.g. 'leadership presence', 'culture fit') that drive the gap.",
                 "Do not adjust ratings by group or set group quotas without legal review; fix criteria and process instead."]))
        elif len(warn):
            recs.append(Rec("Low", f"Watch {L}",
                "; ".join(f"{x.group} (ratio {r2(x.impact_ratio)})" for x in warn.itertuples())
                + " are within the threshold but below 0.90." + mean_note,
                ["Track these groups in each review cycle and investigate if the ratio declines further."]))
        elif mean_note:
            recs.append(Rec("Low", f"Average ratings differ in {L}", mean_note.strip(),
                ["Check whether the gap is explained by objective results before the next review cycle."]))

        if r.tpr_gap is not None and r.tpr_gap > 0.10:
            hi, lo = big.loc[big["tpr"].idxmax()], big.loc[big["tpr"].idxmin()]
            recs.append(Rec("High" if r.tpr_gap > 0.15 else "Medium",
                f"{lo.group} employees who met objectives are under-rated ({L})",
                f"{pct(lo.tpr)} of {lo.group} employees who met their objectives were rated high, versus "
                f"{pct(hi.tpr)} for {hi.group} (gap {pct(r.tpr_gap)}).",
                [f"Review a sample of {lo.group} employees who met objectives but were not rated high, and "
                 "identify which inputs pulled their rating down.",
                 "Require second-level manager sign-off when an AI rating is lower than objective results "
                 "suggest, until the gap closes."]))

        if r.fpr_gap is not None and r.fpr_gap > 0.10:
            hi, lo = big.loc[big["fpr"].idxmax()], big.loc[big["fpr"].idxmin()]
            recs.append(Rec("Medium", f"{hi.group} employees who missed objectives are rated high more often ({L})",
                f"{pct(hi.fpr)} of {hi.group} employees who missed their objectives were still rated high, versus "
                f"{pct(lo.fpr)} for {lo.group} (gap {pct(r.fpr_gap)}), suggesting standards are applied unevenly.",
                ["Review high ratings that lack matching objective results to find the leniency pattern.",
                 "Standardise the rating rubric and the evidence managers must attach to top ratings."]))

        small = g[g["n"] < MIN_N]
        if len(small):
            recs.append(Rec("Low", f"Insufficient data for some {L} groups",
                ", ".join(f"{x.group} (n={x.n})" for x in small.itertuples())
                + f" {'have' if len(small) > 1 else 'has'} fewer than {MIN_N} records, so ratios are unreliable.",
                ["Pool several review cycles before drawing conclusions, and review these ratings manually in the meantime."]))

    recs.append(Rec("Medium", "Governance and monitoring",
        "Ratings feed pay, bonus and promotion decisions, and bias can re-emerge as teams and objectives change.",
        ["Re-run this audit every review cycle, before ratings are released, and keep each report on file.",
         "Keep the AI rating advisory: a named manager stays accountable for every final rating.",
         "Check downstream outcomes (pay rises, bonuses, promotions) for the same groups.",
         "Share results with HR leadership and legal/compliance, and record what changed in response."]))
    order = {"High": 0, "Medium": 1, "Low": 2}
    return sorted(recs, key=lambda x: order[x.priority])


def summary_text(results: list[DimResult], thr: float) -> str:
    flagged = [r for r in results if r.status == "bad"]
    if not flagged:
        return f"No dimension falls below the {thr} impact-ratio threshold for high ratings."
    worst = min(results, key=lambda r: r.worst_di)
    return (f"The AI-suggested performance ratings show adverse impact in {len(flagged)} of {len(results)} dimensions "
            f"({', '.join(label(r.dim) for r in flagged)}). The lowest impact ratio is {r2(worst.worst_di)} "
            f"({worst.worst_group}, {label(worst.dim)}).")


def method_text(p: Prepared, thr: float) -> str:
    eo = ("Equalised odds difference is the larger of the gaps in true positive rate (employees who met objectives "
          "and were rated high) and false positive rate (employees who missed objectives but were rated high), "
          "using the met-objectives column as ground truth." if p.has_truth
          else "Equalised odds was not computed because no ground-truth column (e.g. met_objectives) was provided.")
    return (f"A rating counts as high when it is {high_text(p)}. The high-rating rate was compared across groups in "
            f"each dimension. The impact ratio divides each group's rate by the highest-rate group (four-fifths rule, "
            f"threshold {thr}). Demographic parity difference is the gap between the highest and lowest high-rating "
            f"rates. {eo} Groups with fewer than {MIN_N} records are excluded from summary figures. Model retraining "
            f"is out of scope; recommendations target process and criteria.")


def report_markdown(results: list[DimResult], p: Prepared, thr: float, source: str) -> str:
    overall = p.df["decision"].mean()
    lines = [
        "# AI Bias Audit Report — Performance Ratings", "",
        f"{date.today().isoformat()} · {source} · {len(p.df):,} employees · "
        f"Overall high-rating rate {pct(overall)} ({high_text(p)})", "",
        "## Summary", "", summary_text(results, thr), "",
        "## Method", "", method_text(p, thr), "",
        "## Results by dimension",
    ]
    for r in results:
        mean = f" · Average rating gap {r2(r.mean_gap)} points" if r.mean_gap is not None else ""
        lines += ["", f"### {label(r.dim)} — {STATUS_TEXT[r.status]}", "",
                  f"Demographic parity difference {pct(r.dpd)} · Equalised odds difference {pct(r.eod)}{mean}", "",
                  "| Group | n | Avg rating | High-rating rate | Impact ratio | TPR | FPR |",
                  "|---|---|---|---|---|---|---|"]
        lines += [f"| {x.group} | {x.n} | {r2(x.mean_rating)} | {pct(x.selection_rate)} | {r2(x.impact_ratio)} | "
                  f"{pct(x.tpr)} | {pct(x.fpr)} |" for x in r.groups.itertuples()]
    lines += ["", "## Remediation recommendations"]
    for rec in recommendations(results, thr):
        lines += ["", f"### [{rec.priority}] {rec.title}", "", rec.finding, ""] + [f"- {a}" for a in rec.actions]
    return "\n".join(lines) + "\n"
