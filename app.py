"""AI Bias Audit Dashboard for AI-suggested performance ratings (Streamlit).

Run locally:  streamlit run app.py
"""
from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import audit
from audit import BANDS, MIN_N, STATUS_TEXT, TOP_BANDS, label, pct, pts, r2

st.set_page_config(page_title="Performance Rating Bias Audit", page_icon="⚖️", layout="wide")

SAMPLE_PATH = Path(__file__).parent / "data" / "sample_performance_ratings.csv"
COLORS = {"bad": "#d6336c", "warn": "#f08c00", "ok": "#3b5bdb", "small": "#adb5bd"}
CELL_BG = {"bad": "rgba(214,51,108,.14)", "warn": "rgba(240,140,0,.16)", "ok": "rgba(47,158,68,.14)", "small": "rgba(134,142,150,.14)"}
CELL_FG = {"bad": "#c2255c", "warn": "#b35c00", "ok": "#2b8a3e", "small": "#868e96"}
BAND_COLORS = {"Outstanding": "#1c7ed6", "Exceeds Expectations": "#74c0fc", "Meets Expectations": "#ced4da",
               "Needs Improvement": "#ffa94d", "Unsatisfactory": "#e8590c"}
MGMT_MARK = "#2b2f36"


@st.cache_data
def load_sample() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_PATH)


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload ratings CSV", type=["csv"])
    st.caption(
        "One row per employee: an **ai_score** or **ai_rating**, a **management_score** or **management_rating** "
        "(the fair benchmark) and demographic columns such as **gender**, **age_group** and **region**."
    )
    st.download_button("Download sample / template CSV", load_sample().to_csv(index=False),
                       "performance_ratings_template.csv", "text/csv")

    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded)
        except Exception as e:  # malformed file
            st.error(f"Could not read the CSV: {e}")
            st.stop()
        source = uploaded.name
    else:
        raw = load_sample()
        source = "Synthetic performance ratings (1,000 employees)"

    st.header("Settings")
    candidates = audit.candidate_dims(raw)
    dims = st.multiselect("Compare groups by", candidates, default=candidates[:3], max_selections=3,
                          format_func=label)

    dept_col = next((c for c in raw.columns if c.strip().lower() in ("department", "function")), None)
    if dept_col:
        depts = ["All"] + sorted(raw[dept_col].dropna().astype(str).unique())
        dept = st.selectbox("Department", depts, help="Audit one function at a time, e.g. Finance across regions.")
        if dept != "All":
            raw = raw[raw[dept_col].astype(str) == dept]
            source += f" · {dept} only"
            dims = [d for d in dims if d != dept_col]

    top = st.radio("Count as a top rating", list(TOP_BANDS), index=0,
                   help="The outcome being compared between groups. Top ratings usually drive bonuses and promotions.")
    thr = st.slider("Impact ratio threshold", 0.50, 0.95, 0.80, 0.05,
                    help="Groups whose top-rating rate is below this fraction of the best group's rate are flagged "
                         "(0.80 = the four-fifths rule).")
    st.caption("Uploaded files are processed in memory for this session only and are not stored.")

# ----------------------------------------------------------------------------- analysis
st.title("AI Bias Audit · Performance Ratings")
try:
    prep = audit.prepare(raw, dims or None, top)
except ValueError as e:
    st.error(str(e))
    st.stop()

results = [audit.analyse(prep, d, thr) for d in prep.dims]
flagged = [r for r in results if r.status == "bad"]
worst = min(results, key=lambda r: r.worst_di)
st.caption(f"{source} · {len(prep.df):,} employees · top rating = {top}")

with st.expander("How to read this dashboard"):
    st.markdown(f"""
- **The question:** does the AI rating tool give top ratings ({top}) to every group as often as it should?
- **Management rating** is the fair benchmark: what each employee's manager concluded about their actual performance.
- **Top-rating rate:** the share of a group the AI rated {top}.
- **Impact ratio:** a group's top-rating rate divided by the best group's rate. Below **{thr}** (the four-fifths rule) is a red flag.
- **Parity difference:** the gap, in percentage points, between the highest and lowest top-rating rates.
- **Equalised odds:** among employees managers rated top, does the AI also rate them top equally often in every group
  (*true positive rate*)? And among the rest, does it hand out unearned top ratings equally often (*false positive rate*)?
- **AI − manager score:** how many points, on average, the AI scores a group above (+) or below (−) its managers.
- Groups with fewer than {MIN_N} employees are shown in grey and left out of the summary figures, because a few
  ratings can swing their numbers.
""")

# ----------------------------------------------------------------------------- KPIs
k = st.columns(4)
k[0].metric("Employees audited", f"{len(prep.df):,}", f"{len(prep.dims)} dimensions", delta_color="off")
if prep.has_truth:
    diff = (prep.df["decision"].mean() - prep.df["qualified"].mean()) * 100
    k[1].metric("Top ratings · AI vs managers", f"{pct(prep.df['decision'].mean())} vs {pct(prep.df['qualified'].mean())}",
                f"{diff:+.1f} pts", delta_color="off")
else:
    k[1].metric("Top ratings by AI", pct(prep.df["decision"].mean()))
k[2].metric("Dimensions with adverse impact", f"{len(flagged)} of {len(results)}")
k[3].metric("Lowest impact ratio", r2(worst.worst_di), f"{worst.worst_group} · {label(worst.dim)}", delta_color="off")

# ----------------------------------------------------------------------------- heatmap
st.subheader("Impact ratio overview")
st.caption("Each group's AI top-rating rate relative to the best group in that dimension.")
rows_html = []
for r in results:
    cells = "".join(
        f'<td><div style="background:{CELL_BG[g.status]};color:{CELL_FG[g.status]};border-radius:8px;'
        f'padding:6px 10px;min-width:110px;text-align:center">{escape(str(g.group))}<br>'
        f'<b style="font-size:1.1em">{r2(g.impact_ratio)}</b><br>'
        f'<span style="font-size:.8em;opacity:.8">{pct(g.selection_rate)} top-rated'
        f'{" · n&lt;" + str(MIN_N) if g.status == "small" else ""}</span></div></td>'
        for g in r.groups.itertuples()
    )
    rows_html.append(f'<tr><td style="padding-right:16px"><b>{escape(label(r.dim))}</b></td>{cells}</tr>')
st.markdown(
    f'<div style="overflow-x:auto"><table style="border-collapse:separate;border-spacing:8px;border:none">'
    f'{"".join(rows_html)}</table></div>', unsafe_allow_html=True)
st.caption(f"🔴 below {thr} (adverse impact) · 🟠 {thr}–0.90 (monitor) · 🟢 0.90+ · ⚪ fewer than {MIN_N} employees")

# ----------------------------------------------------------------------------- per-dimension tabs
st.subheader("Detail by dimension")
tabs = st.tabs([label(r.dim) + (" 🔴" if r.status == "bad" else "") for r in results])

for tab, r in zip(tabs, results):
    g = r.groups
    with tab:
        badge = {"bad": "🔴", "warn": "🟠", "ok": "🟢"}[r.status]
        st.markdown(f"**Status:** {badge} {STATUS_TEXT[r.status]}")
        m = st.columns(4)
        m[0].metric("Parity difference", pct(r.dpd),
                    help="Highest minus lowest AI top-rating rate. 0 = equal.")
        m[1].metric("Equalised odds difference", pct(r.eod),
                    help="Largest gap between groups in true or false positive rate. Needs a management rating.")
        m[2].metric("Lowest impact ratio", r2(r.worst_di), r.worst_group, delta_color="off")
        if prep.has_scores:
            big = g[g["n"] >= MIN_N] if (g["n"] >= MIN_N).any() else g
            lo = big.loc[big["score_gap"].idxmin()]
            m[3].metric("Largest AI under-scoring", pts(lo.score_gap), str(lo.group), delta_color="off",
                        help="Average AI score minus management score for the group the AI marks down most.")

        c1, c2 = st.columns(2)
        with c1:
            ref_rate = g.loc[g["group"] == r.reference, "selection_rate"].iloc[0]
            fig = go.Figure(go.Bar(
                name="AI", x=g["selection_rate"], y=g["group"], orientation="h",
                marker_color=[COLORS[s] for s in g["status"]],
                text=[pct(v) for v in g["selection_rate"]], textposition="outside",
                hovertemplate="%{y}: AI %{x:.1%}<extra></extra>"))
            if prep.has_truth:
                fig.add_trace(go.Scatter(
                    name="Managers", x=g["mgmt_rate"], y=g["group"], mode="markers",
                    marker=dict(symbol="diamond", size=12, color=MGMT_MARK, line=dict(color="white", width=1)),
                    hovertemplate="%{y}: managers %{x:.1%}<extra></extra>"))
            fig.add_vline(x=ref_rate * thr, line_dash="dash", line_color="gray",
                          annotation_text=f"{thr:.0%} of {r.reference}", annotation_position="top")
            xmax = max(g["selection_rate"].max(), g["mgmt_rate"].max() if prep.has_truth else 0) * 1.25
            fig.update_layout(title="Top-rating rate by group (bars = AI, ◆ = managers)", height=330,
                              margin=dict(l=10, r=30, t=60, b=10), showlegend=False,
                              xaxis=dict(tickformat=".0%", range=[0, max(xmax, ref_rate * thr * 1.1)]),
                              yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, key=f"rate_{r.dim}")
        with c2:
            if prep.has_truth:
                fig = go.Figure([
                    go.Bar(name="True positive rate", x=g["group"], y=g["tpr"], marker_color="#3b5bdb",
                           text=[pct(v) for v in g["tpr"]], textposition="outside"),
                    go.Bar(name="False positive rate", x=g["group"], y=g["fpr"], marker_color="#91a7ff",
                           text=[pct(v) for v in g["fpr"]], textposition="outside")])
                fig.update_layout(title="Equalised odds · AI agreement with managers", barmode="group", height=330,
                                  margin=dict(l=10, r=10, t=60, b=10), yaxis=dict(tickformat=".0%", range=[0, 1.1]),
                                  legend=dict(orientation="h", y=-0.15))
                st.plotly_chart(fig, key=f"eo_{r.dim}")
                st.caption("True positive rate: of employees managers rated top, the share the AI also rated top. "
                           "False positive rate: of the rest, the share the AI rated top anyway.")
            else:
                st.info("Add a **management_score** or **management_rating** column to compare the AI with managers "
                        "(equalised odds).")

        if prep.has_scores or prep.has_bands:
            c3, c4 = st.columns(2)
            if prep.has_scores:
                with c3:
                    fig = go.Figure(go.Bar(
                        x=g["group"], y=g["score_gap"], text=[pts(v) for v in g["score_gap"]], textposition="outside",
                        marker_color=["#d6336c" if v <= -2 else "#f08c00" if v <= -1 else "#3b5bdb" for v in g["score_gap"]],
                        hovertemplate="%{x}: %{y:+.1f} points<extra></extra>"))
                    fig.add_hline(y=0, line_color="gray")
                    lim = max(3, g["score_gap"].abs().max() * 1.4)
                    fig.update_layout(title="Average AI score minus manager score", height=320,
                                      margin=dict(l=10, r=10, t=60, b=10), yaxis=dict(range=[-lim, lim], title="points"))
                    st.plotly_chart(fig, key=f"gap_{r.dim}")
            if prep.has_bands:
                with c4:
                    dist = audit.rating_distribution(prep, r.dim)
                    order = list(g["group"])
                    fig = go.Figure()
                    for b in BANDS:
                        sub = dist[dist["band"] == b].set_index(["group", "source"])["share"]
                        xs = [[grp for grp in order for _ in ("Manager", "AI")], ["Manager", "AI"] * len(order)]
                        ys = [sub.get((grp, s), 0) for grp in order for s in ("Manager", "AI")]
                        fig.add_trace(go.Bar(name=b, x=xs, y=ys, marker_color=BAND_COLORS[b],
                                             hovertemplate=b + ": %{y:.1%}<extra></extra>"))
                    fig.update_layout(title="Rating mix · managers vs AI", barmode="stack", height=320,
                                      margin=dict(l=10, r=10, t=60, b=10), yaxis=dict(tickformat=".0%"),
                                      legend=dict(orientation="h", y=-0.25, font=dict(size=10)))
                    st.plotly_chart(fig, key=f"dist_{r.dim}")

        table = pd.DataFrame({
            "Group": [f"{x} (reference)" if x == r.reference else x for x in g["group"]],
            "Employees": g["n"], "AI top rate": g["selection_rate"].map(pct),
            "Manager top rate": g["mgmt_rate"].map(pct), "Impact ratio": g["impact_ratio"].map(r2),
            "AI − manager score": g["score_gap"].map(pts),
            "TPR": g["tpr"].map(pct), "FPR": g["fpr"].map(pct), "Status": g["status"].map(STATUS_TEXT)})
        if not prep.has_truth:
            table = table.drop(columns=["Manager top rate", "TPR", "FPR"])
        if not prep.has_scores:
            table = table.drop(columns=["AI − manager score"])
        st.dataframe(table, hide_index=True)

# ----------------------------------------------------------------------------- report
st.subheader("Remediation report")
report_md = audit.report_markdown(results, prep, thr, source)
metrics_csv = pd.concat([r.groups.assign(dimension=r.dim) for r in results])[
    ["dimension", "group", "n", "selected", "selection_rate", "mgmt_rate", "impact_ratio", "ai_avg", "mgmt_avg",
     "score_gap", "tpr", "fpr", "status"]].to_csv(index=False)

d1, d2, _ = st.columns([1, 1, 3])
d1.download_button("Download report (.md)", report_md, "bias_audit_report.md", "text/markdown")
d2.download_button("Download metrics (.csv)", metrics_csv, "bias_audit_metrics.csv", "text/csv")

with st.container(border=True):
    st.markdown(f"**Summary.** {audit.summary_text(results, thr, prep)}")
    with st.expander("Method"):
        st.write(audit.method_text(thr, prep))
    icons = {"High": "🔴", "Medium": "🟠", "Low": "⚪"}
    for rec in audit.recommendations(results, thr, prep):
        st.markdown(f"#### {icons[rec.priority]} [{rec.priority}] {rec.title}")
        st.write(rec.finding)
        st.markdown("\n".join(f"- {a}" for a in rec.actions))
