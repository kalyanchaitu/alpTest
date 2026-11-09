"""AI Bias Audit Dashboard for HR hiring decisions (Streamlit).

Run locally:  streamlit run app.py
"""
from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import audit
from audit import MIN_N, STATUS_TEXT, label, pct, r2

st.set_page_config(page_title="HR Bias Audit", page_icon="⚖️", layout="wide")

SAMPLE_PATH = Path(__file__).parent / "data" / "sample_hiring_decisions.csv"
COLORS = {"bad": "#d6336c", "warn": "#f08c00", "ok": "#3b5bdb", "small": "#adb5bd"}
CELL_BG = {"bad": "rgba(214,51,108,.14)", "warn": "rgba(240,140,0,.16)", "ok": "rgba(47,158,68,.14)", "small": "rgba(134,142,150,.14)"}
CELL_FG = {"bad": "#c2255c", "warn": "#b35c00", "ok": "#2b8a3e", "small": "#868e96"}


@st.cache_data
def load_sample() -> pd.DataFrame:
    if SAMPLE_PATH.exists():
        return pd.read_csv(SAMPLE_PATH)
    return audit.generate_sample()


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload decisions CSV", type=["csv"])
    st.caption(
        "Needs a **decision** column (1 = AI selected), an optional **qualified** column "
        "(1 = actually suitable, enables equalised odds) and 1–3 demographic columns."
    )
    st.download_button("Download CSV template", load_sample().to_csv(index=False),
                       "hr_decisions_template.csv", "text/csv")

    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded)
        except Exception as e:  # malformed file
            st.error(f"Could not read the CSV: {e}")
            st.stop()
        source = uploaded.name
    else:
        raw = load_sample()
        source = "Synthetic hiring sample"

    candidates = audit.candidate_dims(raw)
    st.header("Settings")
    dims = st.multiselect("Demographic dimensions", candidates, default=candidates[:3], max_selections=3,
                          format_func=label)
    thr = st.slider("Impact ratio threshold", 0.50, 0.95, 0.80, 0.05,
                    help="Groups whose selection rate is below this fraction of the highest group's rate are flagged "
                         "(0.80 = the four-fifths rule).")
    st.caption("Uploaded files are processed in memory for this session only and are not stored.")

# ----------------------------------------------------------------------------- analysis
try:
    prep = audit.prepare(raw, dims or None)
except ValueError as e:
    st.title("AI Bias Audit · Hiring Decisions")
    st.error(str(e))
    st.stop()

results = [audit.analyse(prep, d, thr) for d in prep.dims]
overall = prep.df["decision"].mean()
flagged = [r for r in results if r.status == "bad"]
worst = min(results, key=lambda r: r.worst_di)

# ----------------------------------------------------------------------------- header + KPIs
st.title("AI Bias Audit · Hiring Decisions")
st.caption(f"{source} · {len(prep.df):,} decisions")

k = st.columns(4)
k[0].metric("Decisions audited", f"{len(prep.df):,}", f"{len(prep.dims)} dimensions", delta_color="off")
k[1].metric("Overall selection rate", pct(overall))
k[2].metric("Dimensions with adverse impact", f"{len(flagged)} of {len(results)}")
k[3].metric("Lowest impact ratio", r2(worst.worst_di), f"{worst.worst_group} · {label(worst.dim)}",
            delta_color="off")

# ----------------------------------------------------------------------------- heatmap
st.subheader("Impact ratio overview")
st.caption("Each group's selection rate relative to the highest-rate group in that dimension.")
rows_html = []
for r in results:
    cells = "".join(
        f'<td><div style="background:{CELL_BG[g.status]};color:{CELL_FG[g.status]};border-radius:8px;'
        f'padding:6px 10px;min-width:110px;text-align:center">{escape(str(g.group))}<br>'
        f'<b style="font-size:1.1em">{r2(g.impact_ratio)}</b><br>'
        f'<span style="font-size:.8em;opacity:.8">{pct(g.selection_rate)} selected'
        f'{" · n&lt;" + str(MIN_N) if g.status == "small" else ""}</span></div></td>'
        for g in r.groups.itertuples()
    )
    rows_html.append(f'<tr><td style="padding-right:16px"><b>{escape(label(r.dim))}</b></td>{cells}</tr>')
st.markdown(
    f'<div style="overflow-x:auto"><table style="border-collapse:separate;border-spacing:8px;border:none">'
    f'{"".join(rows_html)}</table></div>', unsafe_allow_html=True)
st.caption(f"🔴 below {thr} (adverse impact) · 🟠 {thr}–0.90 (monitor) · 🟢 0.90+ · ⚪ fewer than {MIN_N} records")

# ----------------------------------------------------------------------------- per-dimension tabs
st.subheader("Detail by dimension")
tabs = st.tabs([label(r.dim) + (" 🔴" if r.status == "bad" else "") for r in results])

for tab, r in zip(tabs, results):
    g = r.groups
    with tab:
        badge = {"bad": "🔴", "warn": "🟠", "ok": "🟢"}[r.status]
        st.markdown(f"**Status:** {badge} {STATUS_TEXT[r.status]}")
        m = st.columns(3)
        m[0].metric("Demographic parity difference", pct(r.dpd),
                    help="Highest minus lowest group selection rate. 0 = perfect parity.")
        m[1].metric("Equalised odds difference", pct(r.eod),
                    help="Largest gap in true or false positive rate between groups. Needs a qualified column.")
        m[2].metric("Lowest impact ratio", r2(r.worst_di), r.worst_group, delta_color="off")

        c1, c2 = st.columns(2)
        with c1:
            ref_rate = g.loc[g["group"] == r.reference, "selection_rate"].iloc[0]
            fig = go.Figure(go.Bar(
                x=g["selection_rate"], y=g["group"], orientation="h",
                marker_color=[COLORS[s] for s in g["status"]],
                text=[pct(v) for v in g["selection_rate"]], textposition="outside",
                hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
            fig.add_vline(x=ref_rate * thr, line_dash="dash", line_color="gray",
                          annotation_text=f"{thr:.0%} of {r.reference}", annotation_position="top")
            fig.update_layout(title="Selection rate by group", height=320, margin=dict(l=10, r=30, t=60, b=10),
                              xaxis=dict(tickformat=".0%", range=[0, max(g["selection_rate"].max() * 1.25, ref_rate * thr * 1.1)]),
                              yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, key=f"rate_{r.dim}")
        with c2:
            if prep.has_truth:
                fig = go.Figure([
                    go.Bar(name="True positive rate", x=g["group"], y=g["tpr"], marker_color="#3b5bdb",
                           text=[pct(v) for v in g["tpr"]], textposition="outside"),
                    go.Bar(name="False positive rate", x=g["group"], y=g["fpr"], marker_color="#91a7ff",
                           text=[pct(v) for v in g["fpr"]], textposition="outside")])
                fig.update_layout(title="Equalised odds · error rates by group", barmode="group", height=320,
                                  margin=dict(l=10, r=10, t=60, b=10), yaxis=dict(tickformat=".0%", range=[0, 1.05]),
                                  legend=dict(orientation="h", y=-0.15))
                st.plotly_chart(fig, key=f"eo_{r.dim}")
            else:
                st.info("Add a **qualified** column to the CSV to compute true and false positive rates (equalised odds).")

        table = pd.DataFrame({
            "Group": [f"{x} (reference)" if x == r.reference else x for x in g["group"]],
            "Applicants": g["n"], "Selected": g["selected"],
            "Selection rate": g["selection_rate"].map(pct), "Impact ratio": g["impact_ratio"].map(r2),
            "TPR": g["tpr"].map(pct), "FPR": g["fpr"].map(pct), "Status": g["status"].map(STATUS_TEXT)})
        st.dataframe(table, hide_index=True)

# ----------------------------------------------------------------------------- report
st.subheader("Remediation report")
report_md = audit.report_markdown(results, prep, thr, source)
metrics_csv = pd.concat([r.groups.assign(dimension=r.dim) for r in results])[
    ["dimension", "group", "n", "selected", "selection_rate", "impact_ratio", "tpr", "fpr", "status"]].to_csv(index=False)

d1, d2, _ = st.columns([1, 1, 3])
d1.download_button("Download report (.md)", report_md, "bias_audit_report.md", "text/markdown")
d2.download_button("Download metrics (.csv)", metrics_csv, "bias_audit_metrics.csv", "text/csv")

with st.container(border=True):
    st.markdown(f"**Summary.** {audit.summary_text(results, thr)}")
    with st.expander("Method"):
        st.write(audit.method_text(thr, prep.has_truth))
    icons = {"High": "🔴", "Medium": "🟠", "Low": "⚪"}
    for rec in audit.recommendations(results, thr):
        st.markdown(f"#### {icons[rec.priority]} [{rec.priority}] {rec.title}")
        st.write(rec.finding)
        st.markdown("\n".join(f"- {a}" for a in rec.actions))
