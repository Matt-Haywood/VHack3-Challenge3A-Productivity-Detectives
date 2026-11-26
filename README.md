# Productivity Detective: Streamlit site

An interactive Streamlit version of `productivity_analysis.ipynb` for **Virtual Hack 3, Challenge 3: Productivity Detective**.

## Run it

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python -m streamlit run app.py
```

Then open http://localhost:8501. The first load takes about 30 seconds while scikit-learn imports and the driver model trains. After that, results are cached.

## Pages (mapped to the brief)

| Page | Brief question |
|---|---|
| Executive summary | The headline story, the two focus opportunities and pain points answered |
| Q1 · Where are the gaps? | Crews (teams), work packages (departments), periods/trend, regions and projects |
| Q2 · What drives them? | Rework, idle time, overtime (Simpson's paradox), incentives, design changes and site conditions, plus model driver importance and a genuine vs misleading scorecard |
| Q3 · Recommendations | What-if simulator (rework cut, idle removal, overtime cap), impact waterfall and a pilot roadmap |
| Q4 · Measuring success | KPI tree, SPC p-chart and a pilot sample-size calculator |
| Persona lens | Tailored views for all seven personas |
| Data & limitations | Data quality, caveats, dictionary and an explorer |

Use the sidebar filters (dates, work package, crew, region, project type) to slice every page.

## Files

- `app.py`: Streamlit UI (multi-page via `st.navigation`, Plotly charts)
- `analysis.py`: loading, cleaning, gradient-boosting driver models, scenario engine, SPC and sample-size helpers
- `Data/construction_workplace_productivity_pack.xlsx`: source data
