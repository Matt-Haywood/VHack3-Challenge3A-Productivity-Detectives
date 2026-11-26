# Productivity Detective: Streamlit site

An interactive Streamlit version of `productivity_analysis.ipynb` for **Virtual Hack 3, Challenge 3: Productivity Detective**.

## Run it

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python -m streamlit run app.py
```

Then open http://localhost:8501. The first load takes about 30 seconds while scikit-learn imports and the driver model trains. After that, results are cached.

## Pages

| Page | What it contains |
|---|---|
| Overview | Filtered metrics, crew comparisons and productivity over time |
| Compare patterns | Crew, work package, period, region and project comparisons |
| Explore factors | Model feature importance, factor comparisons, correlations and model caveats |
| Business impact | Labour spend, cost per completed unit, overtime and spend by delivery outcome |
| Scenario explorer | User-adjustable input scenarios with model estimates against the filtered baseline |
| Variation & study size | Process-variation chart and an interactive two-group sample-size estimator |
| Data & methods | Data quality, caveats, dictionary and a cleaned-data explorer |

Use the sidebar filters (dates, work package, crew, region, project type) to explore every page. Charts describe
patterns and associations; spend on below-target observations is not an estimate of avoidable loss. Scenario estimates
are not causal forecasts or recommendations.

## Files

- `app.py`: Streamlit UI (multi-page via `st.navigation`, Plotly charts)
- `analysis.py`: loading, cleaning, gradient-boosting driver models, scenario engine, SPC and sample-size helpers
- `Data/construction_workplace_productivity_pack.xlsx`: source data
