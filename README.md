# AI Bias Audit Dashboard — Performance Ratings

A Streamlit dashboard that checks AI-suggested annual performance ratings for demographic bias. It compares the AI's ratings with the managers' ratings across gender, age group and region, shows where the AI treats groups differently, and writes a remediation report.

ISB "AI in Business" ALP project, Team 1.

## What it does

- **Compares groups** (up to three dimensions at a time, by default gender, age group and region), optionally within one department.
- **Fairness metrics** for the share of each group the AI gives a top rating (Exceeds or Outstanding, or Outstanding only):
  - **Impact ratio**: a group's top-rating rate divided by the best group's rate; below 0.80 fails the four-fifths rule (threshold adjustable).
  - **Parity difference**: the gap between the highest and lowest top-rating rates.
  - **Equalised odds**: compares the AI rating with the management rating, the benchmark of actual performance. The true positive rate is the share of employees managers rated top whom the AI also rated top; the false positive rate is the share of the rest whom the AI rated top anyway.
  - **AI − manager score gap**: how many points, on average, the AI scores each group above or below its managers.
- **Charts**: impact-ratio heatmap, AI vs manager top-rating rates, equalised odds, score gap, and the rating mix (managers vs AI) per group.
- **Remediation report**, in two forms:
  - **Rule-based**: findings ranked High, Medium or Low with concrete actions on the rating process, inputs and governance.
  - **AI-drafted** (optional): an LLM, called through [OpenRouter](https://openrouter.ai), writes a narrative report for HR leadership from the computed metrics.
  Both download as Markdown, and the metrics download as CSV.

## AI-drafted report (OpenRouter)

How it works (`llm_report.py`):

1. The dashboard calculates every metric itself (`audit.py`). The LLM never does the maths.
2. Only group-level figures are sent: group sizes, top-rating rates, impact ratios, true/false positive rates, score gaps and the rule-based findings. No employee rows, IDs or names leave the app. The dashboard shows exactly what is sent.
3. A fixed prompt asks for an executive summary, key findings, likely causes, an actions table (priority, owner, timing) and a monitoring plan, using only the numbers provided.
4. Every number in the draft is checked against the figures sent; any that don't match are flagged for review.
5. If no key is set, or the call fails (for example the free daily limit is reached), the rule-based report is still shown.

Setup:

1. Create a free account at [openrouter.ai](https://openrouter.ai) and create an API key.
2. If requests fail with "No endpoints found matching your data policy", allow free models in OpenRouter **Settings → Privacy**.
3. On Streamlit Community Cloud, open the app's **Settings → Secrets** and add:

   ```toml
   OPENROUTER_API_KEY = "sk-or-..."
   # optional: pin one free model for consistent wording; the default lets OpenRouter pick a free model
   # OPENROUTER_MODEL = "openrouter/free"
   ```

   For local runs, copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` (it is git-ignored). Never commit a real key: this repository is public.

Free models are limited to 50 requests a day across the account (1,000 a day after buying at least 10 credits), and the default `openrouter/free` setting may pick a different model each time, so wording varies between drafts.

Out of scope: retraining the AI model.

## Data format

One row per employee.

| column | required | meaning |
|---|---|---|
| `ai_score` or `ai_rating` | yes | Score (0–100) or band suggested by the AI tool |
| `management_score` or `management_rating` | recommended | Score or band given by management; enables equalised odds and the score gap |
| demographic columns | 1–3 | e.g. `gender`, `age_group`, `region`, each with 2–12 distinct values |
| `department` | optional | Enables the single-department filter |

Bands follow the performance template: 90+ Outstanding, 80–89 Exceeds Expectations, 70–79 Meets Expectations, 60–69 Needs Improvement, below 60 Unsatisfactory. If only scores are given, bands are derived from them. Files with a 0/1 `decision` (AI outcome) and `qualified` (benchmark) column also work.

`data/sample_performance_ratings.csv` is a synthetic dataset of 1,000 employees of a manufacturing company (the team's 100 original employees plus 900 generated). Management ratings are fair by design; the AI score has bias deliberately built in against women, employees aged 45+ and the East region, so the dashboard should flag those groups. The full workbook, with score detail and the answer key, is in the project's shared drive (`ISB synthetic data – audit ready.xlsx`).

**Do not commit real employee data to this repository.**

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Or open the repository in GitHub Codespaces: the dev container installs the requirements and starts the app.

## Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub and click **Create app**.
2. Pick this repository and the `main` branch, set the main file to `app.py`, and deploy.

## Project structure

```
app.py                              Streamlit UI
audit.py                            Metrics, recommendations and rule-based report (pandas only, reusable)
llm_report.py                       AI-drafted report via OpenRouter (aggregate-only payload, number check)
data/sample_performance_ratings.csv Synthetic sample data
requirements.txt
.streamlit/config.toml
.streamlit/secrets.toml.example     Template for the OpenRouter key (local runs)
.devcontainer/                      GitHub Codespaces setup
```

## Method notes

- The reference group in each dimension is the group with the highest AI top-rating rate.
- Groups with fewer than 30 employees are shown but excluded from summary figures, since their ratios are unreliable.
- The four-fifths rule is a screening heuristic, not a legal test. Treat flagged results as prompts for investigation and review them with HR and legal or compliance colleagues.
