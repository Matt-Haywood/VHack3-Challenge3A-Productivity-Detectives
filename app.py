"""Productivity Detective — Streamlit dashboard for Virtual Hack 3, Challenge 3.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="Productivity Detective", page_icon="🕵️", layout="wide")

# scikit-learn's first import is slow; show feedback instead of a blank page on cold start.
with st.spinner("🕵️ Warming up the detective (first load only)…"):
    import analysis as A

# --------------------------------------------------------------------------------------------
# Theme
# --------------------------------------------------------------------------------------------
YELLOW, PAPER, LIGHT_GREY, GREY, GRID = "#f2c744", "#f3f1e8", "#b9b8b2", "#858585", "#373737"
# Keep existing chart roles mapped to the restrained brand palette.
NAVY, CORAL, TEAL, GOLD, SKY = PAPER, GREY, YELLOW, "#c7a331", LIGHT_GREY
GOOD, BAD = YELLOW, GREY
SEQ = [YELLOW, PAPER, LIGHT_GREY, GOLD, GREY, "#626262"]
pio.templates["detective"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Inter, Segoe UI, sans-serif", color=PAPER, size=13),
        colorway=SEQ,
        title=dict(font=dict(size=17, color=PAPER), x=0.01),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor=GRID, zerolinecolor=GREY),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GREY),
        hoverlabel=dict(bgcolor="#252525", font_color=PAPER, font_size=12),
        margin=dict(l=10, r=10, t=55, b=10),
        legend=dict(orientation="h", y=-0.18),
    )
)
pio.templates.default = "plotly_dark+detective"

st.markdown(
    f"""
<style>
:root {{
    color-scheme: dark;
    --primary-color: {YELLOW};
    --background-color: #111111;
    --secondary-background-color: #1b1b1b;
    --text-color: {PAPER};
}}
html, body, .stApp, [data-testid="stAppViewContainer"] {{background: #111111; color: {PAPER};}}
[data-testid="stHeader"] {{background: #111111;}}
.block-container {{padding-top: 2rem; padding-bottom: 3rem;}}
h1, h2, h3, h4 {{color: {PAPER}; letter-spacing: -.02em;}}
a {{color: {YELLOW};}}
.hero {{background: #1b1b1b; color: {PAPER}; padding: 1.45rem 1.75rem; border: 1px solid #303030;
        border-left: 4px solid {YELLOW}; border-radius: 10px; margin-bottom: 1.2rem;}}
.hero h1 {{color: {PAPER}; margin: 0; font-size: 2rem; letter-spacing: -.035em;}}
.hero p {{color: {LIGHT_GREY}; margin: .4rem 0 0 0; font-size: 1rem;}}
.hero .tag {{display:inline-block; background:{YELLOW}; color:#111111; padding:3px 8px; border-radius:4px;
            font-size:.68rem; font-weight:750; letter-spacing:.08em; margin-bottom:.55rem;}}
.kpi {{background: #1b1b1b; color: {PAPER}; border-radius: 10px; padding: .9rem 1rem;
       border: 1px solid #303030; box-shadow: none; height: 100%;}}
.kpi .label {{font-size: .72rem; text-transform: uppercase; letter-spacing: .08em; color: {LIGHT_GREY}; font-weight: 650;}}
.kpi .value {{font-size: 1.8rem; font-weight: 750; color: {PAPER}; line-height: 1.15; margin-top: .25rem;}}
.kpi .delta {{font-size: .82rem; font-weight: 500; color: {GREY}; margin-top: .25rem;}}
.kpi.accent {{border-top: 3px solid {YELLOW};}}
.kpi.good, .kpi.navy, .kpi.gold {{border-top: 3px solid #303030;}}
section[data-testid="stSidebar"] {{background: #171717; color: {PAPER}; border-right: 1px solid #303030;}}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{color: {LIGHT_GREY};}}
[data-testid="stSidebarNavLink"] {{border-left: 3px solid transparent; border-radius: 5px;}}
[data-testid="stSidebarNavLink"][aria-current="page"] {{background: #292719; color: {YELLOW}; border-left-color: {YELLOW};}}
[data-testid="stSidebarNavLink"]:hover {{background: #242424; color: {PAPER};}}
.stTabs [data-baseweb="tab-list"] {{gap: .35rem; border-bottom: 1px solid #303030;}}
.stTabs [data-baseweb="tab"] {{color: {LIGHT_GREY}; height: 2.8rem;}}
.stTabs [data-baseweb="tab"][aria-selected="true"] {{color: {YELLOW}; border-bottom-color: {YELLOW};}}
button[kind="primary"] {{background: {YELLOW}; color: #111111; border-color: {YELLOW};}}
div[data-baseweb="select"] > div {{background: #202020; border-color: #3a3a3a;}}
</style>
""",
    unsafe_allow_html=True,
)


def hero(title: str, subtitle: str, tag: str = "PRODUCTIVITY DETECTIVE") -> None:
    st.markdown(f'<div class="hero"><span class="tag">{tag}</span><h1>{title}</h1><p>{subtitle}</p></div>',
                unsafe_allow_html=True)


def kpi(col, label: str, value: str, delta: str = "", tone: str = "navy", delta_color: str | None = None) -> None:
    color = delta_color or GREY
    col.markdown(
        f'<div class="kpi {tone}"><div class="label">{label}</div><div class="value">{value}</div>'
        f'<div class="delta" style="color:{color}">{delta}</div></div>',
        unsafe_allow_html=True,
    )


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


# --------------------------------------------------------------------------------------------
# Data & model (cached)
# --------------------------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading and cleaning the workbook…")
def get_data():
    wb = A.load_workbook()
    df, audit = A.clean(wb["Productivity_Data"])
    return df, audit, wb.get("Data_Dictionary", pd.DataFrame()), wb.get("Read_Me", pd.DataFrame()), wb.get(
        "Interpretation_Notes", pd.DataFrame())


@st.cache_resource(show_spinner="Training driver models (one-off)…")
def get_models(_df: pd.DataFrame) -> A.DriverModels:
    return A.DriverModels(_df)


@st.cache_data(show_spinner=False)
def rework_curve(_key: str, _df: pd.DataFrame) -> pd.DataFrame:
    return get_models(DF).response_curve(_df, "Rework_Hours", np.round(np.linspace(0.2, 5, 25), 2))


DF, AUDIT, DICTIONARY, README, NOTES = get_data()
MODELS = get_models(DF)

# --------------------------------------------------------------------------------------------
# Sidebar filters
# --------------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Productivity Detective")
    st.caption("Virtual Hack 3 · Challenge 3")
    st.markdown("### Filters")
    min_d, max_d = DF["Observation_Date"].min().date(), DF["Observation_Date"].max().date()
    date_range = st.date_input("Observation dates", (min_d, max_d), min_value=min_d, max_value=max_d)
    f_wp = st.multiselect("Work package", sorted(DF["Work_Package"].unique()), placeholder="All packages")
    f_crew = st.multiselect("Crew", sorted(DF["Crew_ID"].unique()), placeholder="All crews")
    f_region = st.multiselect("Region", sorted(DF["Site_Region"].unique()), placeholder="All regions")
    f_type = st.multiselect("Project type", sorted(DF["Project_Type"].unique()), placeholder="All types")
    st.markdown("---")
    st.caption("Filters apply to every page. The driver model is trained once on all 1,197 records.")

mask = pd.Series(True, index=DF.index)
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    mask &= DF["Observation_Date"].between(pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]))
for col, sel in [("Work_Package", f_wp), ("Crew_ID", f_crew), ("Site_Region", f_region), ("Project_Type", f_type)]:
    if sel:
        mask &= DF[col].isin(sel)
D = DF[mask].copy()
FILTERED = len(D) < len(DF)

if D.empty:
    st.warning("No observations match the current filters — widen the selection in the sidebar.")
    st.stop()

def filter_note() -> None:
    if FILTERED:
        st.caption(f"Filtered view: {len(D):,} of {len(DF):,} observations.")


# --------------------------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------------------------
def page_summary() -> None:
    hero("Explore productivity patterns",
         f"{len(DF):,} observations · {DF['Crew_ID'].nunique()} crews · "
         f"{DF['Project_ID'].nunique()} projects · {min_d:%d %b %Y} → {max_d:%d %b %Y}. "
         "Use the filters and comparisons to investigate the data.")
    filter_note()

    met = D["Met_Target"].mean()
    c = st.columns(4)
    kpi(c[0], "Observations", f"{len(D):,}", "in the current view", "accent")
    kpi(c[1], "Target hit rate", pct(met), "actual at or above plan", "navy")
    kpi(c[2], "Actual productivity", f"{D['Actual_pct'].mean():.1f}%",
        f"{D['Variance_pp'].mean():+.2f}pp vs plan ({D['Planned_pct'].mean():.1f}%)", "navy")
    kpi(c[3], "Labour cost", f"£{D['Estimated_Labour_Cost_GBP'].sum() / 1e6:.2f}m",
        f"{D['Overtime_Hours'].sum():,.0f} overtime hrs", "gold", GREY)

    left, right = st.columns(2)
    with left:
        cs = A.group_summary(D, "Crew_ID").sort_values("Met_Rate_pct")
        fig = px.bar(cs, x="Met_Rate_pct", y="Crew_ID", orientation="h", color="Met_Rate_pct",
                     color_continuous_scale=[CORAL, GOLD, TEAL], range_color=[50, 90],
                     title="Target hit rate by crew",
                     labels={"Met_Rate_pct": "Hit rate (%)", "Crew_ID": ""})
        fig.update_layout(height=420, coloraxis_showscale=False, xaxis_range=[0, 100])
        st.plotly_chart(fig, width="stretch")
    with right:
        daily = D.groupby("Observation_Date").agg(
            actual=("Actual_pct", "mean"), planned=("Planned_pct", "mean")).reset_index()
        fig = go.Figure()
        fig.add_scatter(x=daily["Observation_Date"], y=daily["planned"], name="Planned",
                        line=dict(color=GREY, dash="dash"))
        fig.add_scatter(x=daily["Observation_Date"], y=daily["actual"], name="Actual",
                        line=dict(color=NAVY, width=2))
        fig.update_layout(title="Daily mean productivity", yaxis_title="Productivity (%)", height=420)
        fig.update_xaxes(tickformat="%d %b")
        st.plotly_chart(fig, width="stretch")


def page_gaps() -> None:
    hero("Compare productivity patterns",
         "Compare actual and planned productivity across crews, work packages, periods and project context.")
    filter_note()

    tabs = st.tabs(["Crews", "Work packages", "Periods & trend", "Project context"])

    with tabs[0]:
        cs = A.group_summary(D, "Crew_ID").sort_values("Total_Shortfall_pp", ascending=False)
        cs["Cum_Share"] = cs["Share_of_Shortfall_pct"].cumsum()
        c1, c2 = st.columns([1.2, 1])
        with c1:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_bar(x=cs["Crew_ID"], y=cs["Total_Shortfall_pp"], name="Lost productivity (pp-days)",
                        marker_color=GREY)
            fig.add_scatter(x=cs["Crew_ID"], y=cs["Cum_Share"], name="Cumulative share", mode="lines+markers",
                            line=dict(color=NAVY, width=3), secondary_y=True)
            fig.add_hline(y=80, line_dash="dot", line_color=GREY, secondary_y=True)
            fig.update_yaxes(title_text="Sum of shortfall below plan (pp)", secondary_y=False)
            fig.update_yaxes(title_text="Cumulative %", range=[0, 105], secondary_y=True)
            fig.update_layout(title="Pareto of lost productivity by crew", height=420)
            st.plotly_chart(fig, width="stretch")
        with c2:
            cs2 = cs.sort_values("Mean_Actual_pct")
            fig = go.Figure()
            for _, r in cs2.iterrows():
                fig.add_shape(type="line", x0=r["Mean_Planned_pct"], x1=r["Mean_Actual_pct"], y0=r["Crew_ID"],
                              y1=r["Crew_ID"], line=dict(color=GOOD if r["Mean_Actual_pct"] >= r["Mean_Planned_pct"] else BAD, width=4))
            fig.add_scatter(x=cs2["Mean_Planned_pct"], y=cs2["Crew_ID"], mode="markers", name="Planned",
                            marker=dict(size=12, color="white", line=dict(color=NAVY, width=2)))
            fig.add_scatter(x=cs2["Mean_Actual_pct"], y=cs2["Crew_ID"], mode="markers", name="Actual",
                            marker=dict(size=13, color=NAVY))
            fig.update_layout(title="Planned vs actual productivity (mean %)", height=420, xaxis_title="Productivity (%)")
            st.plotly_chart(fig, width="stretch")

        hm = D.pivot_table(index="Crew_ID", columns="Work_Package", values="Met_Target", aggfunc="mean") * 100
        n = D.pivot_table(index="Crew_ID", columns="Work_Package", values="Met_Target", aggfunc="size")
        fig = px.imshow(hm, text_auto=".0f", color_continuous_scale=[[0, GREY], [0.5, GOLD], [1, YELLOW]],
                        zmin=20, zmax=100, aspect="auto", labels=dict(color="Hit rate %"),
                        title="Hit rate (%) by crew and work package")
        fig.update_traces(customdata=n.values, hovertemplate="%{y} · %{x}<br>Hit rate %{z:.0f}%<br>n=%{customdata}<extra></extra>")
        fig.update_layout(height=460, xaxis_title="", yaxis_title="")
        st.plotly_chart(fig, width="stretch")

        st.dataframe(
            cs[["Crew_ID", "Observations", "Met_Rate_pct", "Mean_Actual_pct", "Mean_Variance_pp", "Missed_Days",
                "Share_of_Shortfall_pct", "Mean_Rework_Hours", "Idle_Hours", "Mean_Overtime"]],
            hide_index=True, width="stretch",
            column_config={
                "Met_Rate_pct": st.column_config.ProgressColumn("Hit rate", format="%.0f%%", min_value=0, max_value=100),
                "Share_of_Shortfall_pct": st.column_config.ProgressColumn("Share of shortfall", format="%.1f%%", min_value=0, max_value=max(20, cs["Share_of_Shortfall_pct"].max())),
                "Mean_Actual_pct": st.column_config.NumberColumn("Actual %", format="%.1f"),
                "Mean_Variance_pp": st.column_config.NumberColumn("Variance pp", format="%+.2f"),
                "Mean_Rework_Hours": st.column_config.NumberColumn("Rework hrs/day", format="%.2f"),
                "Idle_Hours": st.column_config.NumberColumn("Idle hrs", format="%.1f"),
                "Mean_Overtime": st.column_config.NumberColumn("Overtime hrs/day", format="%.0f"),
            })

    with tabs[1]:
        ws = A.group_summary(D, "Work_Package").sort_values("Met_Rate")
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(ws, x="Work_Package", y="Met_Rate_pct", color="Met_Rate_pct",
                         color_continuous_scale=[CORAL, GOLD, TEAL], range_color=[50, 90],
                         text=ws["Met_Rate_pct"].map("{:.0f}%".format), title="Target hit rate by work package",
                         labels={"Met_Rate_pct": "Hit rate (%)", "Work_Package": ""})
            fig.update_layout(height=400, coloraxis_showscale=False)
            st.plotly_chart(fig, width="stretch")
        with c2:
            fig = px.sunburst(D[D["Shortfall_pp"] > 0], path=["Work_Package", "Crew_ID"], values="Shortfall_pp",
                              color="Work_Package", color_discrete_sequence=[NAVY, CORAL, TEAL],
                              title="Where the lost productivity sits (package → crew)")
            fig.update_traces(hovertemplate="%{label}<br>%{value:.0f} pp lost<br>%{percentRoot:.1%} of total<extra></extra>")
            fig.update_layout(height=400)
            st.plotly_chart(fig, width="stretch")
        fig = px.violin(D, x="Work_Package", y="Variance_pp", color="Work_Package", box=True, points="outliers",
                        color_discrete_sequence=[NAVY, CORAL, TEAL],
                        title="Distribution of actual − planned productivity (pp)",
                        labels={"Variance_pp": "Variance (pp)", "Work_Package": ""})
        fig.add_hline(y=0, line_dash="dash", line_color=CORAL)
        fig.update_layout(height=420, showlegend=False)
        st.plotly_chart(fig, width="stretch")
        st.caption("`Work_In_Progress_Units` is only recorded for Structural installation.")

    with tabs[2]:
        daily = D.groupby("Observation_Date").agg(met=("Met_Target", "mean"), actual=("Actual_pct", "mean"),
                                                  planned=("Planned_pct", "mean"), n=("Record_ID", "count")).reset_index()
        daily["roll"] = daily["actual"].rolling(7, min_periods=1).mean()
        fig = go.Figure()
        fig.add_scatter(x=daily["Observation_Date"], y=daily["planned"], name="Planned", line=dict(color=GREY, dash="dash"))
        fig.add_scatter(x=daily["Observation_Date"], y=daily["actual"], name="Actual (daily)", mode="lines",
                        line=dict(color=GREY, width=1.5), fill="tonexty", fillcolor="rgba(242,199,68,0.08)")
        fig.add_scatter(x=daily["Observation_Date"], y=daily["roll"], name="Actual (7-day rolling)",
                        line=dict(color=NAVY, width=3.5))
        fig.update_layout(title="Daily mean productivity vs plan", yaxis_title="Productivity (%)", height=420)
        st.plotly_chart(fig, width="stretch")

        c1, c2 = st.columns(2)
        ps = A.group_summary(D, "Quarter").sort_values("Quarter")
        fig = px.bar(ps, x="Quarter", y="Met_Rate_pct", text=ps["Met_Rate_pct"].map("{:.0f}%".format),
                     color="Mean_Variance_pp", color_continuous_scale=[GREY, GOLD, YELLOW], color_continuous_midpoint=0,
                     hover_data={"Observations": True}, title="Hit rate by period (`Quarter` field)",
                     labels={"Met_Rate_pct": "Hit rate (%)", "Quarter": "", "Mean_Variance_pp": "Var pp"})
        fig.update_layout(height=380)
        c1.plotly_chart(fig, width="stretch")
        c1.caption("`Quarter` behaves like a within-month period (data span ~10 weeks). Quarter5 has only "
                   f"{int(ps.loc[ps['Quarter'] == 'Quarter5', 'Observations'].sum())} observations — interpret with care.")
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"]
        ds = A.group_summary(D, "Day_of_Week").set_index("Day_of_Week").reindex([d for d in order if d in D["Day_of_Week"].unique()]).reset_index()
        fig = px.line_polar(ds, r="Met_Rate_pct", theta="Day_of_Week", line_close=True, markers=True,
                            title="Hit rate by day of week", range_r=[50, 90])
        fig.update_traces(fill="toself", line_color=YELLOW, fillcolor="rgba(242,199,68,0.16)")
        fig.update_layout(height=380)
        c2.plotly_chart(fig, width="stretch")

        cal = D.groupby(["Week", "Crew_ID"]).Met_Target.mean().unstack() * 100
        fig = px.imshow(cal.T, color_continuous_scale=[[0, GREY], [0.5, GOLD], [1, YELLOW]], zmin=0, zmax=100,
                        aspect="auto", labels=dict(color="Hit %", x="Week starting", y=""),
                        title="Weekly hit rate by crew — persistent problems vs one-off bad weeks")
        fig.update_xaxes(tickformat="%d %b")
        fig.update_layout(height=430)
        st.plotly_chart(fig, width="stretch")

    with tabs[3]:
        dim = st.segmented_control("Break down by", ["Site_Region", "Project_Type", "Project_Phase",
                                                     "Work_Package_Complexity", "Site_Access_Condition", "Crew_Size_Band"],
                                   default="Site_Region", format_func=A.pretty)
        dim = dim or "Site_Region"
        gs = A.group_summary(D, dim, min_n=10).sort_values("Mean_Variance_pp")
        c1, c2 = st.columns(2)
        fig = px.bar(gs, y=dim, x="Mean_Variance_pp", orientation="h",
                     color=np.where(gs["Mean_Variance_pp"] >= 0, "Above plan", "Below plan"),
                     color_discrete_map={"Above plan": TEAL, "Below plan": CORAL},
                     text=gs["Mean_Variance_pp"].map("{:+.1f}".format),
                     title=f"Mean variance vs plan by {A.pretty(dim)} (pp)", labels={dim: "", "Mean_Variance_pp": "pp"})
        fig.update_layout(height=420, legend_title=None)
        c1.plotly_chart(fig, width="stretch")
        fig = px.scatter(gs, x="Met_Rate_pct", y="Mean_Actual_pct", size="Observations", color="Total_Shortfall_pp",
                         text=dim, color_continuous_scale=[TEAL, GOLD, CORAL], size_max=45,
                         title="Hit rate vs mean actual (bubble = observations, colour = lost pp)",
                         labels={"Met_Rate_pct": "Hit rate (%)", "Mean_Actual_pct": "Mean actual (%)", "Total_Shortfall_pp": "Lost pp"})
        fig.update_traces(textposition="top center", textfont_size=10)
        fig.update_layout(height=420)
        c2.plotly_chart(fig, width="stretch")
        spread = gs["Mean_Variance_pp"].max() - gs["Mean_Variance_pp"].min()
        st.caption(f"Observed range in mean variance across {A.pretty(dim)} groups: {spread:.1f}pp.")


def page_drivers() -> None:
    hero("Explore operational factors",
         "Compare distributions and associations. Use the model as one lens, not as evidence of causation.")
    filter_note()

    c1, c2 = st.columns([1.3, 1])
    with c1:
        imp = MODELS.importance.head(12).iloc[::-1]
        fig = px.bar(imp, x="Importance", y="Label", orientation="h", error_x="Std",
                     color="Importance", color_continuous_scale=[GREY, GOLD, YELLOW],
                     title="Model feature importance for target misses",
                     labels={"Label": "", "Importance": "Drop in AUC when shuffled"})
        fig.update_layout(height=460, coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.markdown("#### About the model")
        m1, m2 = st.columns(2)
        kpi(m1, "Hold-out AUC", f"{MODELS.auc:.2f}", "predicting target miss", "good", GREY)
        kpi(m2, "Productivity MAE", f"{MODELS.mae_pp:.1f}pp", f"vs {MODELS.baseline_mae_pp:.1f}pp naive", "good", GREY)
        st.markdown(
            f"""
Gradient-boosted trees trained on **{len(DF) - MODELS.test_size:,}** records and tested on **{MODELS.test_size}**
unseen records, using only *operational* inputs (rework, idle, overtime, crew, materials, weather, etc.).

**Deliberately excluded** to avoid misleading results:
- Outcome-derived fields (actual, variance, completed qty, progress %) — *leakage*.
- **Incentive £** — likely paid *because* the target was hit (reverse causality).
- **Delay risk score** — illustrative only, per the data pack.

Importance shows association with the outcome, **not causation**.
""")

    st.markdown("### Factor views")
    tabs = st.tabs(["Rework", "Idle time", "Overtime", "Incentives", "Design changes",
                    "Explore a factor", "Correlation matrix"])

    with tabs[0]:
        c1, c2 = st.columns(2)
        order = ["Very low", "Low", "Medium", "High", "Very high"]
        rb = D.groupby(["Work_Package", "Rework_Band"], observed=True).Met_Target.mean().mul(100).reset_index()
        fig = px.bar(rb, x="Rework_Band", y="Met_Target", color="Work_Package", barmode="group",
                     category_orders={"Rework_Band": order}, color_discrete_sequence=[NAVY, CORAL, TEAL],
                     title="Target hit rate by rework band and work package",
                     labels={"Met_Target": "Hit rate (%)", "Rework_Band": "Rework hours (quintile)", "Work_Package": ""})
        fig.update_layout(height=420)
        c1.plotly_chart(fig, width="stretch")
        curve = rework_curve("all", DF)
        fig = go.Figure()
        fig.add_scatter(x=curve["value"], y=curve["met_rate"], mode="lines", line=dict(color=YELLOW, width=3, shape="spline"),
                        fill="tozeroy", fillcolor="rgba(242,199,68,0.12)", name="Model")
        fig.add_vline(x=D["Rework_Hours"].median(), line_dash="dot", line_color=NAVY, annotation_text="view median")
        fig.update_layout(title="Model-estimated hit rate by rework level",
                          xaxis_title="Rework hours per crew-day", yaxis_title="Predicted hit rate (%)",
                          yaxis_range=[50, 100], height=420, showlegend=False)
        c2.plotly_chart(fig, width="stretch")
        st.caption("The grouped rates are descriptive. The model curve estimates associations within the fitted model; it does not estimate intervention effects.")

    with tabs[1]:
        idle = D.groupby("Had_Idle").agg(n=("Record_ID", "count"), met=("Met_Target", "mean"),
                                         actual=("Actual_pct", "mean"), var=("Variance_pp", "mean"))
        c = st.columns(4)
        if True in idle.index:
            kpi(c[0], "Days with idle time", f"{int(idle.loc[True, 'n'])}", f"{pct(idle.loc[True, 'n'] / len(D))} of days", "gold", GREY)
            kpi(c[1], "Hit rate on idle days", pct(idle.loc[True, "met"], 0),
                f"vs {pct(idle.loc[False, 'met'], 0)} otherwise", "accent", CORAL)
            kpi(c[2], "Productivity on idle days", f"{idle.loc[True, 'actual']:.0f}%",
                f"{idle.loc[True, 'actual'] - idle.loc[False, 'actual']:+.0f}pp vs normal", "accent")
            kpi(c[3], "Idle worker-hours", f"{(D['Idle_Time_Hours'] * D['Idle_Workers']).sum():,.0f}",
                "lost capacity", "navy", GREY)
            st.write("")
            c1, c2 = st.columns(2)
            ev = D[D["Had_Idle"]]
            fig = px.scatter(ev, x="Observation_Date", y="Crew_ID", size="Idle_Workers", color="Variance_pp",
                             color_continuous_scale=[CORAL, GOLD, TEAL], size_max=40,
                             hover_data=["Idle_Time_Hours", "Idle_Workers", "Work_Package", "Material_Availability_Percent"],
                             title="Idle-time events (bubble = idle workers)",
                             labels={"Variance_pp": "Var pp", "Observation_Date": "", "Crew_ID": ""})
            fig.update_layout(height=400)
            c1.plotly_chart(fig, width="stretch")
            byc = ev.groupby("Crew_ID").Idle_Time_Hours.sum().sort_values().reset_index()
            fig = px.bar(byc, x="Idle_Time_Hours", y="Crew_ID", orientation="h", color_discrete_sequence=[GOLD],
                         title="Idle hours concentrated in a few crews", labels={"Crew_ID": "", "Idle_Time_Hours": "Idle hours"})
            fig.update_layout(height=400)
            c2.plotly_chart(fig, width="stretch")
            st.caption("Counts and rates describe the selected records; idle-time events may be too sparse for stable comparisons.")
        else:
            st.caption("No idle-time events in the current filter.")

    with tabs[2]:
        ob = D.groupby("Overtime_Band", observed=True).agg(
            met=("Met_Target", "mean"), actual=("Actual_pct", "mean"), cost=("Estimated_Labour_Cost_GBP", "mean"),
            cpu=("Cost_per_Unit_GBP", "median"), n=("Record_ID", "count")).reset_index()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_bar(x=ob["Overtime_Band"], y=ob["cost"], name="Mean labour cost (£)", marker_color=GREY, secondary_y=True)
        fig.add_scatter(x=ob["Overtime_Band"], y=ob["actual"], name="Mean actual productivity (%)", mode="lines+markers",
                        line=dict(color=GREY, width=3), marker=dict(size=9))
        fig.add_scatter(x=ob["Overtime_Band"], y=100 * ob["met"], name="Hit rate (%)", mode="lines+markers",
                        line=dict(color=YELLOW, width=3, dash="dot"), marker=dict(size=8))
        fig.update_yaxes(title_text="%", range=[40, 100], secondary_y=False)
        fig.update_yaxes(title_text="£ per crew-day", secondary_y=True, showgrid=False)
        fig.update_layout(title="Overtime bands: productivity and mean labour cost", height=440)
        st.plotly_chart(fig, width="stretch")
        c1, c2 = st.columns(2)
        fig = px.scatter(D, x="Overtime_Hours", y="Actual_pct", color="Work_Package", trendline="lowess",
                         opacity=0.45, color_discrete_sequence=[NAVY, CORAL, TEAL],
                         title="Overtime hours and actual productivity, by work package",
                         labels={"Overtime_Hours": "Overtime hours", "Actual_pct": "Actual productivity (%)", "Work_Package": ""})
        fig.update_layout(height=420)
        c1.plotly_chart(fig, width="stretch")
        rho = D.groupby("Work_Package")[["Overtime_Hours", "Actual_Productivity"]].corr(method="spearman").xs(
            "Overtime_Hours", level=1)["Actual_Productivity"].reset_index().rename(columns={"Actual_Productivity": "rho"})
        fig = px.bar(rho, x="Work_Package", y="rho", text=rho["rho"].map("{:+.2f}".format), color_discrete_sequence=[GOLD],
                     title="Spearman ρ: overtime vs actual productivity", labels={"rho": "ρ", "Work_Package": ""})
        fig.update_layout(yaxis_range=[-1, 1], height=420)
        c2.plotly_chart(fig, width="stretch")
        st.caption("Aggregate patterns can differ from within-package patterns when work-package mix varies (Simpson's paradox). Compare the grouped chart, scatter plot and correlations together.")

    with tabs[3]:
        st_df = D[D["Work_Package"] == "Structural installation"]
        c1, c2 = st.columns(2)
        if not st_df.empty:
            inc = st_df.groupby("Has_Incentive").agg(met=("Met_Target", "mean"), n=("Record_ID", "count")).reset_index()
            inc["Label"] = inc["Has_Incentive"].map({True: "Incentive paid", False: "No incentive"})
            fig = px.bar(inc, x="Label", y=inc["met"] * 100, text=(inc["met"] * 100).map("{:.0f}%".format),
                         color="Label", color_discrete_map={"Incentive paid": TEAL, "No incentive": CORAL},
                         title="Structural installation: hit rate with/without incentive",
                         labels={"y": "Hit rate (%)", "Label": ""}, hover_data=["n"])
            fig.update_layout(height=400, showlegend=False)
            c1.plotly_chart(fig, width="stretch")
            paid = st_df[st_df["Has_Incentive"]]
            fig = px.scatter(paid, x="Incentive_GBP", y="Actual_pct", color="Met_Target", opacity=.6,
                             color_discrete_map={True: TEAL, False: CORAL}, log_x=True,
                             title="Incentive £ vs actual productivity (paid days)",
                             labels={"Incentive_GBP": "Incentive £ (log)", "Actual_pct": "Actual (%)", "Met_Target": "Met target"})
            fig.update_layout(height=400)
            c2.plotly_chart(fig, width="stretch")
        share = D.groupby("Work_Package").Has_Incentive.mean().mul(100).round(0)
        st.caption(f"Incentives appear at different rates across packages (share of days with an incentive: {share.to_dict()}). Timing and selection can affect this association; it should not be read as a causal effect.")

    with tabs[4]:
        dc = A.group_summary(D, "Design_Change_Count")
        c1, c2 = st.columns([1.3, 1])
        fig = px.bar(dc, x="Design_Change_Count", y="Met_Rate_pct", text=dc["Met_Rate_pct"].map("{:.0f}%".format),
                     color_discrete_sequence=[SKY], hover_data=["Observations"],
                     title="Hit rate by number of design changes on the day",
                     labels={"Design_Change_Count": "Design changes", "Met_Rate_pct": "Hit rate (%)"})
        fig.update_layout(height=400, yaxis_range=[0, 100])
        c1.plotly_chart(fig, width="stretch")
        c2.dataframe(dc[["Design_Change_Count", "Observations", "Met_Rate_pct"]].round(1), hide_index=True)
        st.caption("These are unadjusted group comparisons. Schedule and cost effects are not shown here.")

    with tabs[5]:
        numeric = [c for c in A.MODEL_NUMERIC + ["Incentive_GBP", "Estimated_Labour_Cost_GBP", "Delay_Risk_Score"]]
        cats = A.MODEL_CATEGORICAL + ["Crew_Size_Band"]
        c1, c2 = st.columns([1, 3])
        kind = c1.radio("Factor type", ["Numeric", "Category"])
        if kind == "Numeric":
            f = c1.selectbox("Factor", numeric, format_func=A.pretty, index=numeric.index("Rework_Hours"))
            split = c1.checkbox("Split by work package", True)
            fig = px.scatter(D, x=f, y="Variance_pp", color="Work_Package" if split else None, trendline="lowess",
                             opacity=.4, color_discrete_sequence=[NAVY, CORAL, TEAL], marginal_x="histogram",
                             title=f"{A.pretty(f)} vs productivity variance",
                             labels={f: A.pretty(f), "Variance_pp": "Actual − planned (pp)", "Work_Package": ""})
            fig.add_hline(y=0, line_dash="dash", line_color=GREY)
            fig.update_layout(height=520)
            c2.plotly_chart(fig, width="stretch")
            rho = D[[f, "Productivity_Variance"]].corr(method="spearman").iloc[0, 1]
            c1.metric("Spearman ρ with variance", f"{rho:+.3f}")
        else:
            f = c1.selectbox("Factor", cats, format_func=A.pretty)
            gs = A.group_summary(D, f).sort_values("Met_Rate_pct")
            fig = px.box(D, x="Variance_pp", y=f, color_discrete_sequence=[NAVY], points=False,
                         category_orders={f: gs[f].tolist()}, title=f"Productivity variance by {A.pretty(f)}",
                         labels={"Variance_pp": "Actual − planned (pp)", f: ""})
            fig.add_vline(x=0, line_dash="dash", line_color=CORAL)
            fig.update_layout(height=520)
            c2.plotly_chart(fig, width="stretch")
            c1.dataframe(gs[[f, "Observations", "Met_Rate_pct"]].round(1), hide_index=True)

    with tabs[6]:
        cols = ["Actual_Productivity", "Productivity_Variance", "Rework_Hours", "Rework_Rate_Percent", "Idle_Time_Hours",
                "Overtime_Hours", "Incentive_GBP", "Crew_Size", "Design_Change_Count", "Material_Availability_Percent",
                "Equipment_Downtime_Hours", "Workforce_Absence_Rate_Percent", "Average_Workforce_Experience_Years",
                "Training_Hours", "Quality_Pass_Rate_Percent", "Safety_Standdown_Hours", "Rainfall_mm", "Delay_Risk_Score"]
        corr = D[cols].corr(method="spearman")
        corr.index = corr.columns = [A.pretty(c) if c in A.PRETTY else c.replace("_", " ") for c in cols]
        fig = px.imshow(corr, color_continuous_scale=[[0, GREY], [0.5, "#202020"], [1, YELLOW]],
                        zmin=-1, zmax=1, text_auto=".2f", aspect="auto",
                        title="Spearman rank correlations (descriptive — not causal)")
        fig.update_traces(textfont_size=9)
        fig.update_layout(height=680)
        st.plotly_chart(fig, width="stretch")



def page_scenarios() -> None:
    hero("Explore hypothetical scenarios",
         "Change selected inputs and compare model estimates with the filtered baseline.")
    filter_note()
    st.caption("These are model-based scenarios, not causal forecasts. The source data is synthetic, and results depend on the selected assumptions.")
    c1, c2 = st.columns([1, 2.2])
    with c1:
        with st.container(border=True):
            crews = st.multiselect("Crews", sorted(D["Crew_ID"].unique()), placeholder="All crews")
            pkgs = st.multiselect("Work packages", sorted(D["Work_Package"].unique()), placeholder="All packages")
            st.caption("Leave a selection empty to include all values in the current view.")
            cut = st.slider("Rework-hours change (%)", 0, 80, 0, 5)
            idle_fix = st.toggle("Set idle-time inputs to zero", False)
            cap_on = st.toggle("Apply an overtime-hours cap", False)
            cap = st.slider("Overtime cap (hrs/crew-day)", 20, 200, 110, 5, disabled=not cap_on)
            premium = st.slider("Overtime pay multiplier", 1.0, 2.0, 1.5, 0.1, disabled=not cap_on)

    base = D.copy()
    scen = A.scenario(base, cut, idle_fix, cap if cap_on else None, crews or None, pkgs or None)
    tgt = scen["_targeted"]
    p0, p1 = MODELS.predict_met(base), MODELS.predict_met(scen)
    s_rw = A.scenario(base, cut, False, None, crews or None, pkgs or None)
    s_idle = A.scenario(base, cut, idle_fix, None, crews or None, pkgs or None)
    p_rw, p_idle = MODELS.predict_met(s_rw), MODELS.predict_met(s_idle)
    hourly = base["Estimated_Labour_Cost_GBP"] / (base["Planned_Labour_Hours"] + base["Overtime_Hours"])
    ot_saved_hrs = (base["Overtime_Hours"] - scen["Overtime_Hours"]).clip(lower=0)
    saving = float((ot_saved_hrs * hourly * premium).sum()) if cap_on else 0.0

    with c2:
        k = st.columns(4)
        if tgt.any():
            kpi(k[0], "Scenario hit rate", pct(p1[tgt.values].mean()),
                f"{100 * (p1 - p0)[tgt.values].mean():+.1f}pp vs baseline", "good")
        else:
            kpi(k[0], "Scenario hit rate", "—", "no matching rows", "good", GREY)
        kpi(k[1], "Baseline hit rate", pct(p0.mean()), f"{len(base):,} observations", "navy", GREY)
        kpi(k[2], "Modelled change", f"{100 * (p1.mean() - p0.mean()):+.1f}pp",
            f"{(p1 - p0).sum():+.1f} predicted target-days", "gold", GREY)
        cost_value = f"−£{saving / 1e3:,.1f}k" if saving else "£0.0k" if cap_on else "—"
        cost_note = f"{ot_saved_hrs.sum():,.0f} overtime hours capped" if cap_on else "Apply a cap to estimate"
        kpi(k[3], "Overtime cost delta", cost_value, cost_note, "navy", GREY)

        if cut or idle_fix or cap_on:
            wf = go.Figure(go.Waterfall(
                orientation="v", measure=["absolute", "relative", "relative", "relative", "total"],
                x=["Baseline", "Rework change", "Idle-time change", "Overtime cap", "Scenario"],
                y=[100 * p0.mean(), 100 * (p_rw.mean() - p0.mean()), 100 * (p_idle.mean() - p_rw.mean()),
                   100 * (p1.mean() - p_idle.mean()), 0],
                text=[f"{100 * p0.mean():.1f}%", f"{100 * (p_rw.mean() - p0.mean()):+.1f}",
                      f"{100 * (p_idle.mean() - p_rw.mean()):+.1f}", f"{100 * (p1.mean() - p_idle.mean()):+.1f}",
                      f"{100 * p1.mean():.1f}%"],
                textposition="outside",
                increasing=dict(marker_color=YELLOW), decreasing=dict(marker_color=GREY),
                totals=dict(marker_color=YELLOW), connector=dict(line=dict(color=GREY, dash="dot")),
            ))
            lower = max(0, min(100 * p0.mean(), 100 * p1.mean()) - 8)
            upper = min(100, max(100 * p0.mean(), 100 * p1.mean()) + 8)
            wf.update_layout(title="Sequential modelled hit-rate changes across the current view (%)",
                             yaxis_range=[lower, upper], height=360, showlegend=False)
            st.plotly_chart(wf, width="stretch")
            st.caption("Waterfall contributions are sequential and depend on the order in which the scenario changes are applied.")

            by = pd.DataFrame({"Crew_ID": base["Crew_ID"], "Baseline": p0, "Scenario": p1}).groupby("Crew_ID").mean() * 100
            by = by.sort_values("Baseline").reset_index()
            fig = go.Figure()
            fig.add_bar(x=by["Crew_ID"], y=by["Baseline"], name="Baseline", marker_color=GREY)
            fig.add_bar(x=by["Crew_ID"], y=by["Scenario"], name="Scenario", marker_color=YELLOW)
            fig.update_layout(barmode="group", title="Modelled hit rate by crew: baseline and scenario",
                              yaxis_title="Hit rate (%)", height=420, yaxis_range=[40, 100])
            st.plotly_chart(fig, width="stretch")
        else:
            st.caption("Adjust one or more inputs to compare a scenario with the baseline.")


def page_measure() -> None:
    hero("Monitor variation & explore study size",
         "Inspect hit-rate variation over time and estimate sample sizes using your own assumptions.")
    filter_note()

    st.markdown("### Process variation")
    freq = st.radio("Time interval", ["Daily", "Weekly"], horizontal=True)
    pch = A.p_chart(D, "D" if freq == "Daily" else "W")
    fig = go.Figure()
    fig.add_scatter(x=pch["Observation_Date"], y=100 * pch["UCL"],
                    line=dict(color=GREY, dash="dash", width=1), name="Upper limit")
    fig.add_scatter(x=pch["Observation_Date"], y=100 * pch["LCL"],
                    line=dict(color=GREY, dash="dash", width=1), name="Lower limit",
                    fill="tonexty", fillcolor="rgba(242,199,68,0.08)")
    fig.add_scatter(x=pch["Observation_Date"], y=100 * pch["CL"],
                    line=dict(color=YELLOW, width=2), name="Centre line")
    fig.add_scatter(x=pch["Observation_Date"], y=100 * pch["p"], mode="lines+markers", name="Hit rate",
                    line=dict(color=LIGHT_GREY, width=2),
                    marker=dict(size=7, color=np.where(pch["Signal"], YELLOW, LIGHT_GREY)))
    fig.update_layout(title="Target hit rate with 3σ control limits",
                      yaxis_title="Hit rate (%)", height=420, yaxis_range=[-2, 102])
    st.plotly_chart(fig, width="stretch")
    st.metric("Points outside control limits", int(pch["Signal"].sum()))
    st.caption("Limits are calculated from the selected records. A signal flags unusual variation; it does not explain its cause.")

    st.markdown("### Two-group sample-size estimator")
    p_base = float(D["Met_Target"].mean())
    st.caption(f"Baseline hit rate in the current view: {pct(p_base)}.")
    c1, c2 = st.columns([1, 1.5])
    with c1:
        change_pp = st.slider("Difference to detect (percentage points)", 1, 30, 10)
        p_goal = min(0.99, p_base + change_pp / 100)
        alpha = st.select_slider("Significance level (α)", [0.01, 0.05, 0.1], 0.05)
        power = st.select_slider("Statistical power", [0.7, 0.8, 0.9], 0.8)
        n = A.two_proportion_sample_size(p_base, p_goal, alpha, power)
        st.metric("Observations needed per group", f"{n:,}")
        st.caption(f"Requested difference: {change_pp}pp; used difference: {100 * (p_goal - p_base):.1f}pp (maximum hit rate: 99%).")
    with c2:
        changes = np.arange(1, 31)
        sizes = [
            A.two_proportion_sample_size(p_base, min(0.99, p_base + change / 100), alpha, power)
            for change in changes
        ]
        fig = px.line(x=changes, y=sizes, markers=True,
                      labels={"x": "Difference to detect (pp)", "y": "Observations per group"},
                      title="Sample-size sensitivity to the selected assumptions")
        fig.add_vline(x=change_pp, line_dash="dash", line_color=CORAL)
        fig.update_layout(height=340)
        st.plotly_chart(fig, width="stretch")
    st.caption("The estimate assumes independent observations. Repeated observations from the same crews may require a larger sample.")


def page_data() -> None:
    hero("Data, methods & limitations", "Review the source data, analytical methods and assumptions.")
    c1, c2 = st.columns([1, 1.3])
    with c1:
        st.markdown("#### Data quality audit")
        st.dataframe(AUDIT, hide_index=True, width="stretch")
        st.markdown("#### About the data pack")
        st.dataframe(README, hide_index=True, width="stretch")
        st.dataframe(NOTES, hide_index=True, width="stretch")
    with c2:
        st.markdown("#### Limitations & guardrails")
        st.markdown(f"""
- **Synthetic data.** The pack is fictional (a manufacturing dataset recast for construction). Findings illustrate a
  method; validate against real project data before acting.
- **Associations, not causes.** Group differences and correlations don't control for every confounder; repeated crews
  and dates mean observations are not fully independent.
- **Short window.** {days_span()} of data — no seasonality can be inferred.
- **Small samples.** Only {int(DF['Had_Idle'].sum())} idle-time days and {int((DF['Quarter'] == 'Quarter5').sum())} Quarter5 records.
- **Incentive** likely reflects reverse causality and is excluded from the driver model.
- **`Delay_Risk_Score`** is illustrative only — not used for decisions.
- **Work-in-progress** is only recorded for Structural installation ({int(DF['Work_In_Progress_Units'].isna().sum())} blanks kept as blank).
- **Target hit** is defined as actual ≥ planned − 0.1pp (rounding tolerance).
""")
        st.markdown("#### Method")
        st.markdown("""
1. **Clean** — trim text, type-cast, check duplicates/ranges/variance arithmetic. No rows dropped or imputed.
2. **Describe** — hit rate, variance and lost productivity (sum of pp below plan) by crew, package, period, context.
3. **Explain** — rank correlations, within-package checks (to catch Simpson's paradox), gradient-boosted model with
   permutation importance on a hold-out set.
4. **Explore scenarios** — adjust assumptions and inspect model estimates.
5. **Monitor** — review process variation and estimate sample sizes.
""")
    st.markdown("#### Explore the cleaned data")
    cols = st.multiselect("Columns", list(A.NUMERIC_COLUMNS) + A.MODEL_CATEGORICAL,
                          default=["Crew_ID", "Work_Package", "Planned_Productivity", "Actual_Productivity",
                                   "Productivity_Variance", "Rework_Hours", "Idle_Time_Hours", "Overtime_Hours", "Incentive_GBP"])
    view = D[["Record_ID", "Observation_Date"] + cols]
    st.dataframe(view, hide_index=True, width="stretch", height=360)
    st.download_button("⬇️ Download filtered data (CSV)", view.to_csv(index=False).encode(), "productivity_filtered.csv", "text/csv")
    with st.expander("Data dictionary"):
        st.dataframe(DICTIONARY, hide_index=True, width="stretch")


def days_span() -> str:
    return f"{(DF['Observation_Date'].max() - DF['Observation_Date'].min()).days + 1} days"


pg = st.navigation([
    st.Page(page_summary, title="Overview", icon=":material/dashboard:", default=True),
    st.Page(page_gaps, title="Compare patterns", icon=":material/compare_arrows:", url_path="compare"),
    st.Page(page_drivers, title="Explore factors", icon=":material/query_stats:", url_path="factors"),
    st.Page(page_scenarios, title="Scenario explorer", icon=":material/tune:", url_path="scenarios"),
    st.Page(page_measure, title="Variation & study size", icon=":material/monitoring:", url_path="measure"),
    st.Page(page_data, title="Data & methods", icon=":material/database:", url_path="data"),
])
pg.run()
