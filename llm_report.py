"""LLM-drafted remediation report via OpenRouter.

Design
------
- All numbers are computed in `audit.py`. The LLM only writes the narrative.
- Only group-level aggregates are sent (group sizes, rates, ratios, score gaps, rule-based findings).
  No employee rows, IDs or names ever leave the app.
- Every number in the draft is checked against the numbers that were sent; anything that does not
  match is listed so a reviewer can check it.
- If no API key is set or the call fails, the app keeps showing the rule-based report.

Configuration (Streamlit secrets or environment variables)
----------------------------------------------------------
OPENROUTER_API_KEY   required to enable the AI draft
OPENROUTER_MODEL     optional, default "openrouter/free" (OpenRouter picks a currently free model);
                     set a specific model such as "<provider>/<model>:free" for consistent output
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass

import pandas as pd

import audit
from audit import MIN_N, STATUS_TEXT, label

API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openrouter/free"
TIMEOUT_S = 90

SYSTEM_PROMPT = """You are an HR analytics and AI-fairness specialist writing an audit report for HR leadership \
of a manufacturing company. The company uses an AI tool that suggests annual performance ratings. \
Management ratings are the benchmark of actual performance.

You receive group-level fairness metrics that have already been calculated, plus rule-based findings. \
Write a clear, practical remediation report in Markdown for non-technical HR readers.

Rules:
- Use ONLY the numbers in the data provided. Do not calculate new statistics, estimate, or invent figures.
- When you cite a number, copy it exactly as written in the data (e.g. "36.6%", "0.75", "-3.3 pts").
- Name the groups exactly as they appear in the data.
- Describe possible causes as hypotheses to investigate, not as facts.
- Do not recommend adjusting scores by demographic group; recommend fixing inputs, process and oversight.
- Groups marked as low sample must not be treated as conclusive.
- This is not legal advice; say so once in the closing section.
- Keep it under about 700 words.

Use exactly these sections:
## Executive summary
(3-4 sentences: is the AI rating tool fair, which groups are affected, how serious)
## Key findings
(one short paragraph per affected dimension, citing the impact ratio, top-rating rates, equalised odds and score gap)
## Likely causes to investigate
(bullets)
## Recommended actions
(a Markdown table with columns: Priority | Action | Owner | When. Owners are roles such as HR Business Partner, \
People Analytics, Line managers, AI vendor, Legal/Compliance. When is e.g. "Before ratings are final", "Within 30 days")
## Monitoring and governance
(bullets, ending with the not-legal-advice note)"""


def build_payload(results: list[audit.DimResult], p: audit.Prepared, thr: float, source: str) -> dict:
    """Aggregate-only summary sent to the LLM. Every value is pre-formatted as it may be quoted."""
    dims = []
    for r in results:
        groups = []
        for g in r.groups.itertuples():
            row = {"group": str(g.group), "employees": int(g.n), "low_sample": bool(g.n < MIN_N),
                   "ai_top_rating_rate": audit.pct(g.selection_rate), "impact_ratio": audit.r2(g.impact_ratio),
                   "status": STATUS_TEXT[g.status]}
            if p.has_truth:
                row.update(manager_top_rating_rate=audit.pct(g.mgmt_rate),
                           true_positive_rate=audit.pct(g.tpr), false_positive_rate=audit.pct(g.fpr))
            if p.has_scores:
                row.update(avg_ai_score=f"{g.ai_avg:.1f}", avg_manager_score=f"{g.mgmt_avg:.1f}",
                           ai_minus_manager_score=audit.pts(g.score_gap))
            groups.append(row)
        dims.append({"dimension": label(r.dim), "status": STATUS_TEXT[r.status], "reference_group": r.reference,
                     "parity_difference": audit.pct(r.dpd), "equalised_odds_difference": audit.pct(r.eod),
                     "groups": groups})
    return {
        "dataset": source,
        "employees": len(p.df),
        "top_rating_definition": p.top_label,
        "impact_ratio_threshold": f"{thr:.2f}",
        "minimum_group_size": MIN_N,
        "ai_top_rating_rate_overall": audit.pct(p.df["decision"].mean()),
        "manager_top_rating_rate_overall": audit.pct(p.df["qualified"].mean()) if p.has_truth else None,
        "summary": audit.summary_text(results, thr, p),
        "dimensions": dims,
        "rule_based_findings": [{"priority": x.priority, "title": x.title, "finding": x.finding}
                                for x in audit.recommendations(results, thr, p)],
    }


def build_messages(payload: dict) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Audit data (JSON):\n```json\n" + json.dumps(payload, indent=1) + "\n```\n\n"
                                        "Write the report."}]


class LLMError(Exception):
    pass


@dataclass
class Draft:
    text: str
    model: str
    unverified: list[str]


def call_openrouter(messages: list[dict], api_key: str, model: str = DEFAULT_MODEL,
                    timeout: int = TIMEOUT_S) -> tuple[str, str]:
    """Return (text, model_used). Raises LLMError with a readable message."""
    body = json.dumps({"model": model, "messages": messages, "temperature": 0.3}).encode()
    req = urllib.request.Request(API_URL, data=body, method="POST", headers={
        "Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/kalyanchaitu/alpTest", "X-Title": "AI Bias Audit Dashboard"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = json.loads(e.read().decode()).get("error", {}).get("message", "")
        except Exception:
            pass
        hints = {401: "The API key was rejected. Check OPENROUTER_API_KEY.",
                 402: "The account has no credit for this model. Use a free model (openrouter/free or a ':free' model).",
                 404: "No model matched. If you use free models, allow them in OpenRouter → Settings → Privacy.",
                 429: "Free-model limit reached (50 requests a day without credits). Try again later."}
        raise LLMError(f"OpenRouter error {e.code}: {detail or e.reason}. {hints.get(e.code, '')}".strip()) from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise LLMError(f"Could not reach OpenRouter: {getattr(e, 'reason', e)}") from e
    if "error" in data:
        raise LLMError(f"OpenRouter error: {data['error'].get('message', data['error'])}")
    try:
        text = data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as e:
        raise LLMError("OpenRouter returned an unexpected response.") from e
    if not text.strip():
        raise LLMError("The model returned an empty report. Try again; the free router may pick another model.")
    return text.strip(), data.get("model", model)


# --------------------------------------------------------------------------- number check
_NUM = re.compile(r"(?<![\w.])[-+−]?\d+(?:\.\d+)?%?")


def _allowed_numbers(payload: dict) -> set[float]:
    found = set()

    def walk(v):
        if isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            found.add(round(float(v), 2))
        elif isinstance(v, str):
            for m in _NUM.findall(v):
                n = float(m.replace("%", "").replace("−", "-").replace("+", ""))
                found.update({round(n, 2), round(abs(n), 2)})
    walk(payload)
    return found


def unverified_numbers(text: str, payload: dict) -> list[str]:
    """Numbers in the draft that do not appear in the data sent (ignores list numbers, small counts and years)."""
    allowed = _allowed_numbers(payload) | {80.0, 90.0, 70.0, 60.0, 100.0, 0.8, 0.9, 4.0, 5.0, 30.0}
    out = []
    for m in _NUM.findall(text):
        raw = m
        n = float(m.replace("%", "").replace("−", "-").replace("+", ""))
        if not raw.endswith("%") and abs(n) <= 12 and float(n).is_integer():
            continue  # list numbering, "3 dimensions", "4/5" etc.
        if not raw.endswith("%") and 1900 <= n <= 2100:
            continue  # years
        if round(abs(n), 2) in allowed or round(n, 2) in allowed:
            continue
        if any(abs(abs(n) - a) <= 0.051 for a in allowed):
            continue  # rounding of a provided figure
        out.append(raw)
    return sorted(set(out), key=out.index)


def draft_report(results, p, thr, source, api_key, model=DEFAULT_MODEL, caller=call_openrouter) -> Draft:
    payload = build_payload(results, p, thr, source)
    text, used = caller(build_messages(payload), api_key, model)
    return Draft(text=text, model=used, unverified=unverified_numbers(text, payload))
