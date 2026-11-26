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
NAVY, CORAL, TEAL, GOLD, SKY, GREY = "#16365c", "#e85a3c", "#1b998b", "#f4a259", "#3a86ff", "#8d99ae"
GOOD, BAD = "#1b998b", "#e85a3c"
SEQ = [NAVY, CORAL, TEAL, GOLD, SKY, "#8338ec", "#6c757d", "#ff006e"]
pio.templates["detective"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Inter, Segoe UI, sans-serif", color=NAVY, size=13),
        colorway=SEQ,
        title=dict(font=dict(size=17, color=NAVY), x=0.01),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#e3eaf3", zerolinecolor="#c5d3e3"),
        yaxis=dict(gridcolor="#e3eaf3", zerolinecolor="#c5d3e3"),
        hoverlabel=dict(bgcolor="white", font_size=12),
        margin=dict(l=10, r=10, t=55, b=10),
        legend=dict(orientation="h", y=-0.18),
    )
)
pio.templates.default = "plotly_white+detective"

st.markdown(
    f"""
<style>
.block-container {{padding-top: 1.6rem; padding-bottom: 3rem;}}
.hero {{background: linear-gradient(120deg, {NAVY} 0%, #1f4e86 55%, {SKY} 130%); color: white;
        padding: 1.6rem 2rem; border-radius: 18px; margin-bottom: 1.2rem; position: relative; overflow:hidden;}}
.hero h1 {{color: white; margin: 0; font-size: 2.1rem; letter-spacing: -0.5px;}}
.hero p {{color: #dbe7f7; margin: .35rem 0 0 0; font-size: 1.02rem;}}
.hero .tag {{display:inline-block; background:{CORAL}; color:white; padding:2px 10px; border-radius:999px;
            font-size:.75rem; font-weight:700; letter-spacing:.5px; margin-bottom:.5rem;}}
.kpi {{background: white; border-radius: 14px; padding: 1rem 1.1rem; border: 1px solid #e3eaf3;
       box-shadow: 0 2px 10px rgba(22,54,92,.06); height: 100%;}}
.kpi .label {{font-size: .78rem; text-transform: uppercase; letter-spacing: .6px; color: {GREY}; font-weight: 600;}}
.kpi .value {{font-size: 1.85rem; font-weight: 800; color: {NAVY}; line-height: 1.15; margin-top: .2rem;}}
.kpi .delta {{font-size: .85rem; font-weight: 600; margin-top: .15rem;}}
.kpi.accent {{border-top: 4px solid {CORAL};}}
.kpi.good {{border-top: 4px solid {TEAL};}}
.kpi.navy {{border-top: 4px solid {NAVY};}}
.kpi.gold {{border-top: 4px solid {GOLD};}}
.qcard {{background: #f5f8fc; border-left: 5px solid {CORAL}; border-radius: 10px; padding: .9rem 1.1rem; margin-bottom: .7rem;}}
.qcard h4 {{margin: 0 0 .3rem 0; color: {NAVY}; font-size: 1.02rem;}}
.qcard p {{margin: 0; color: #33475b; font-size: .94rem;}}
.opp {{background: white; border: 1px solid #e3eaf3; border-radius: 14px; padding: 1.1rem 1.3rem;
       box-shadow: 0 2px 10px rgba(22,54,92,.06); height: 100%;}}
.opp h3 {{margin-top: 0; color: {NAVY};}}
.pill {{display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: .75rem; font-weight: 700; margin-right: 4px;}}
.pill.real {{background: #d8f3ef; color: #0f6e63;}}
.pill.myth {{background: #fde2dc; color: #a3341b;}}
.pill.caution {{background: #fff1dc; color: #9a5b00;}}
.small {{font-size: .85rem; color: {GREY};}}
section[data-testid="stSidebar"] {{background: #f3f7fc;}}
</style>
""",
    unsafe_allow_html=True,
)


def hero(title: str, subtitle: str, tag: str = "PRODUCTIVITY DETECTIVE") -> None:
    st.markdown(f'<div class="hero"><span class="tag">{tag}</span><h1>{title}</h1><p>{subtitle}</p></div>',
                unsafe_allow_html=True)


def kpi(col, label: str, value: str, delta: str = "", tone: str = "navy", delta_color: str | None = None) -> None:
    color = delta_color or (GOOD if delta.startswith(("+", "▲")) else BAD if delta.startswith(("-", "−", "▼")) else GREY)
    col.markdown(
        f'<div class="kpi {tone}"><div class="label">{label}</div><div class="value">{value}</div>'
        f'<div class="delta" style="color:{color}">{delta}</div></div>',
        unsafe_allow_html=True,
    )


def qcard(title: str, body: str) -> None:
    st.markdown(f'<div class="qcard"><h4>{title}</h4><p>{body}</p></div>', unsafe_allow_html=True)


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
    st.markdown(f"## 🕵️ Productivity Detective")
    st.caption("Virtual Hack 3 · Challenge 3")
    st.markdown("### 🎛️ Filters")
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

# Shared headline numbers (always from the full dataset so recommendations stay stable).
CREWS = A.group_summary(DF, "Crew_ID").sort_values("Met_Rate")
WP = A.group_summary(DF, "Work_Package").sort_values("Met_Rate")
WORST_CREWS = CREWS.head(3)["Crew_ID"].tolist()
BEST_RATE = CREWS.tail(3)["Met_Rate"].mean()
IDLE = DF.groupby("Had_Idle").agg(n=("Met_Target", "size"), met=("Met_Target", "mean"), var=("Variance_pp", "mean"))
RW = DF.groupby("Rework_Band", observed=True).agg(met=("Met_Target", "mean"))
RW_LOW, RW_HIGH = RW.loc["Very low", "met"], RW.loc["Very high", "met"]


def filter_note() -> None:
    if FILTERED:
        st.info(f"🔎 Showing **{len(D):,}** of {len(DF):,} observations based on sidebar filters.", icon="ℹ️")


# --------------------------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------------------------
def page_summary() -> None:
    hero("Turning data into better performance",
         f"{len(DF):,} daily crew observations · {DF['Crew_ID'].nunique()} crews · "
         f"{DF['Project_ID'].nunique()} projects · {min_d:%d %b %Y} → {max_d:%d %b %Y}. "
         "Where are we losing productivity, why, what should change and how will we know it worked?")
    filter_note()

    met = D["Met_Target"].mean()
    c = st.columns(5)
    kpi(c[0], "Target hit rate", pct(met), f"{pct(met - BEST_RATE)} vs best-3 crews", "accent")
    kpi(c[1], "Actual productivity", f"{D['Actual_pct'].mean():.1f}%",
        f"{D['Variance_pp'].mean():+.2f}pp vs plan ({D['Planned_pct'].mean():.1f}%)", "navy")
    kpi(c[2], "Days missing target", f"{int(D['Missed_Target'].sum()):,}",
        f"avg shortfall {D.loc[D['Missed_Target'], 'Shortfall_pp'].mean():.1f}pp when missed", "gold", CORAL)
    kpi(c[3], "Labour cost", f"£{D['Estimated_Labour_Cost_GBP'].sum() / 1e6:.2f}m",
        f"{D['Overtime_Hours'].sum():,.0f} overtime hrs", "navy", GREY)
    kpi(c[4], "Rework hours", f"{D['Rework_Hours'].sum():,.0f}",
        f"strongest driver of missed targets", "accent", CORAL)

    st.write("")
    left, right = st.columns([1.05, 1])
    with left:
        st.markdown("### 🧭 The four management questions — answered")
        w1, w2 = WP.iloc[0], CREWS.head(3)
        qcard("1 · Where are the biggest productivity gaps?",
              f"<b>{', '.join(WORST_CREWS)}</b> hit target only {pct(w2['Met_Rate'].mean(), 0)} of the time "
              f"(vs {pct(BEST_RATE, 0)} for the best three) and carry "
              f"<b>{w2['Share_of_Shortfall_pct'].sum():.0f}%</b> of all lost productivity. "
              f"By work package, <b>{w1['Work_Package']}</b> has the lowest hit rate ({pct(w1['Met_Rate'], 0)}).")
        qcard("2 · Which operational factors are associated with those gaps?",
              f"<b>Rework</b> is the stand-out driver: crews with very low rework hit target {pct(RW_LOW, 0)} of the time "
              f"vs {pct(RW_HIGH, 0)} with very high rework, and the pattern holds in every work package. "
              f"<b>Idle-time</b> events are rare ({int(IDLE.loc[True, 'n'])} days) but severe "
              f"({pct(IDLE.loc[True, 'met'], 0)} hit rate, {IDLE.loc[True, 'var']:+.0f}pp). "
              "Overtime and design changes look important but don't hold up as drivers.")
        qcard("3 · What practical changes could improve performance?",
              "<b>①</b> A <b>first-time-right rework programme</b> targeted at the lowest crews and Fit-out & finishing. "
              "<b>②</b> An <b>idle-time rapid response</b> (materials/equipment triage) plus a "
              "<b>smart overtime cap</b> to stop paying for hours that don't lift output.")
        qcard("4 · How should management measure the results?",
              "A 12-week controlled pilot (pilot vs comparison crews) tracking <b>target hit rate</b> on a "
              "statistical process-control chart, with rework hours/day and idle events as leading indicators "
              "and overtime £ per unit as the cost guardrail.")
    with right:
        st.markdown("### 🎯 Hit rate vs best-in-class")
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=100 * met,
            number=dict(suffix="%", font=dict(size=46, color=NAVY)),
            delta=dict(reference=100 * BEST_RATE, suffix="pp", valueformat=".1f"),
            gauge=dict(
                axis=dict(range=[0, 100], ticksuffix="%"),
                bar=dict(color=NAVY, thickness=0.28),
                steps=[dict(range=[0, 60], color="#fde2dc"), dict(range=[60, 80], color="#fff1dc"),
                       dict(range=[80, 100], color="#d8f3ef")],
                threshold=dict(line=dict(color=CORAL, width=5), thickness=0.85, value=100 * BEST_RATE),
            ),
            title=dict(text=f"Share of crew-days at or above plan<br><span style='font-size:12px;color:{GREY}'>"
                            f"Red marker = average of top-3 crews ({pct(BEST_RATE)})</span>"),
        ))
        fig.update_layout(height=320, margin=dict(t=90, b=20, l=40, r=40))
        st.plotly_chart(fig, width="stretch")

        cs = A.group_summary(D, "Crew_ID").sort_values("Met_Rate_pct")
        cs["Status"] = np.where(cs["Crew_ID"].isin(WORST_CREWS), "Priority crew", "Other crews")
        fig = px.bar(cs, x="Met_Rate_pct", y="Crew_ID", orientation="h", color="Status",
                     color_discrete_map={"Priority crew": CORAL, "Other crews": "#9fb6d1"},
                     text=cs["Met_Rate_pct"].map("{:.0f}%".format),
                     title="Target hit rate by crew", labels={"Met_Rate_pct": "Hit rate (%)", "Crew_ID": ""})
        fig.add_vline(x=100 * met, line_dash="dash", line_color=NAVY, annotation_text=f"overall {pct(met)}",
                      annotation_position="bottom right")
        fig.update_layout(height=380, legend_title=None, xaxis_range=[0, 100])
        st.plotly_chart(fig, width="stretch")

    st.markdown("### 💡 Our focused recommendation")
    o1, o2 = st.columns(2)
    o1.markdown(
        f"""<div class="opp"><h3>① First-time-right: cut rework</h3>
        <span class="pill real">Genuine signal</span><span class="pill real">Holds within every work package</span>
        <p>Rework hours are the #1 driver in our model (permutation importance) and the only factor whose effect is
        large, consistent and plausible. Target the three lowest crews and Fit-out &amp; finishing with
        pre-start quality briefings, hold-point inspections and root-cause logging of every rework event.</p>
        <p class="small">See <b>Recommendations</b> to simulate the impact.</p></div>""",
        unsafe_allow_html=True)
    o2.markdown(
        f"""<div class="opp"><h3>② Kill idle time, cap unproductive overtime</h3>
        <span class="pill real">Severe but rare</span><span class="pill caution">Cost lever</span>
        <p>Idle-time events cost ~{abs(IDLE.loc[True, 'var']):.0f}pp of productivity on the day, and are concentrated in a
        few crews. Overtime rises with labour cost but actual productivity is flat across overtime bands — so a cap
        releases cost without hurting output.</p>
        <p class="small">See <b>Drivers → Overtime</b> for the evidence.</p></div>""",
        unsafe_allow_html=True)


def page_gaps() -> None:
    hero("Q1 · Where are the biggest productivity gaps?",
         "Actual vs target productivity by crew (team), work package (department), period and project context.")
    filter_note()

    tabs = st.tabs(["👷 Crews (teams)", "🏗️ Work packages", "📅 Periods & trend", "🗺️ Projects & regions"])

    with tabs[0]:
        cs = A.group_summary(D, "Crew_ID").sort_values("Total_Shortfall_pp", ascending=False)
        cs["Cum_Share"] = cs["Share_of_Shortfall_pct"].cumsum()
        c1, c2 = st.columns([1.2, 1])
        with c1:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_bar(x=cs["Crew_ID"], y=cs["Total_Shortfall_pp"], name="Lost productivity (pp-days)",
                        marker_color=[CORAL if c in WORST_CREWS else "#9fb6d1" for c in cs["Crew_ID"]])
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
        fig = px.imshow(hm, text_auto=".0f", color_continuous_scale=[[0, CORAL], [0.5, "#fff1dc"], [1, TEAL]],
                        zmin=20, zmax=100, aspect="auto", labels=dict(color="Hit rate %"),
                        title="Hit rate (%) by crew × work package — where exactly is each crew struggling?")
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
        st.caption("Structural installation has the most consistent achievement; Fit-out & finishing has the widest "
                   "downside tail. Note: `Work_In_Progress_Units` is only recorded for Structural installation.")

    with tabs[2]:
        daily = D.groupby("Observation_Date").agg(met=("Met_Target", "mean"), actual=("Actual_pct", "mean"),
                                                  planned=("Planned_pct", "mean"), n=("Record_ID", "count")).reset_index()
        daily["roll"] = daily["actual"].rolling(7, min_periods=1).mean()
        fig = go.Figure()
        fig.add_scatter(x=daily["Observation_Date"], y=daily["planned"], name="Planned", line=dict(color=GREY, dash="dash"))
        fig.add_scatter(x=daily["Observation_Date"], y=daily["actual"], name="Actual (daily)", mode="lines",
                        line=dict(color="#9fb6d1", width=1.5), fill="tonexty", fillcolor="rgba(232,90,60,0.08)")
        fig.add_scatter(x=daily["Observation_Date"], y=daily["roll"], name="Actual (7-day rolling)",
                        line=dict(color=NAVY, width=3.5))
        fig.update_layout(title="Daily mean productivity vs plan", yaxis_title="Productivity (%)", height=420)
        st.plotly_chart(fig, width="stretch")

        c1, c2 = st.columns(2)
        ps = A.group_summary(D, "Quarter").sort_values("Quarter")
        fig = px.bar(ps, x="Quarter", y="Met_Rate_pct", text=ps["Met_Rate_pct"].map("{:.0f}%".format),
                     color="Mean_Variance_pp", color_continuous_scale=[CORAL, "#fff1dc", TEAL], color_continuous_midpoint=0,
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
        fig.update_traces(fill="toself", line_color=CORAL, fillcolor="rgba(232,90,60,0.25)")
        fig.update_layout(height=380)
        c2.plotly_chart(fig, width="stretch")

        cal = D.groupby(["Week", "Crew_ID"]).Met_Target.mean().unstack() * 100
        fig = px.imshow(cal.T, color_continuous_scale=[[0, CORAL], [0.5, "#fff1dc"], [1, TEAL]], zmin=0, zmax=100,
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
        st.caption(f"Spread across {A.pretty(dim)} groups: {spread:.1f}pp. Compare with the crew-level spread "
                   f"({CREWS['Mean_Variance_pp'].max() - CREWS['Mean_Variance_pp'].min():.1f}pp) — crew and "
                   "process factors matter far more than geography or project type.")


def page_drivers() -> None:
    hero("Q2 · Which operational factors are associated with the gaps?",
         "Separating genuine improvement opportunities from misleading correlations.")
    filter_note()

    c1, c2 = st.columns([1.3, 1])
    with c1:
        imp = MODELS.importance.head(12).iloc[::-1]
        fig = px.bar(imp, x="Importance", y="Label", orientation="h", error_x="Std",
                     color="Importance", color_continuous_scale=["#c9d7ea", NAVY, CORAL],
                     title="What best predicts missing target? (ML permutation importance)",
                     labels={"Label": "", "Importance": "Drop in AUC when shuffled"})
        fig.update_layout(height=460, coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.markdown("#### 🤖 About the model")
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

    st.markdown("### 🔬 Factor deep-dives")
    tabs = st.tabs(["♻️ Rework", "⏸️ Idle time", "⏱️ Overtime", "💷 Incentives", "📝 Design changes",
                    "🧩 Explore any factor", "🌡️ Correlation matrix", "✅ Genuine vs misleading"])

    with tabs[0]:
        c1, c2 = st.columns(2)
        order = ["Very low", "Low", "Medium", "High", "Very high"]
        rb = D.groupby(["Work_Package", "Rework_Band"], observed=True).Met_Target.mean().mul(100).reset_index()
        fig = px.bar(rb, x="Rework_Band", y="Met_Target", color="Work_Package", barmode="group",
                     category_orders={"Rework_Band": order}, color_discrete_sequence=[NAVY, CORAL, TEAL],
                     title="Hit rate falls as rework rises — in every work package",
                     labels={"Met_Target": "Hit rate (%)", "Rework_Band": "Rework hours (quintile)", "Work_Package": ""})
        fig.update_layout(height=420)
        c1.plotly_chart(fig, width="stretch")
        curve = rework_curve("all", DF)
        fig = go.Figure()
        fig.add_scatter(x=curve["value"], y=curve["met_rate"], mode="lines", line=dict(color=CORAL, width=4, shape="spline"),
                        fill="tozeroy", fillcolor="rgba(232,90,60,0.12)", name="Model")
        fig.add_vline(x=DF["Rework_Hours"].median(), line_dash="dot", line_color=NAVY, annotation_text="median today")
        fig.update_layout(title="Model: expected hit rate if every crew-day had X rework hours",
                          xaxis_title="Rework hours per crew-day", yaxis_title="Predicted hit rate (%)",
                          yaxis_range=[50, 100], height=420, showlegend=False)
        c2.plotly_chart(fig, width="stretch")
        st.success(f"**Genuine opportunity.** Very-low-rework days hit target {pct(RW_LOW, 0)} of the time vs "
                   f"{pct(RW_HIGH, 0)} for very-high-rework days. The model shows steepest gains between ~2 and ~0.5 "
                   "rework hours — i.e. eliminating the *first* hours of rework matters most.")

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
            st.success("**Genuine but rare.** Idle events are infrequent, so the total impact is smaller than rework — "
                       "but each event is severe and concentrated, making it a cheap, quick win (escalation + root-cause).")
        else:
            st.info("No idle-time events in the current filter.")

    with tabs[2]:
        ob = D.groupby("Overtime_Band", observed=True).agg(
            met=("Met_Target", "mean"), actual=("Actual_pct", "mean"), cost=("Estimated_Labour_Cost_GBP", "mean"),
            cpu=("Cost_per_Unit_GBP", "median"), n=("Record_ID", "count")).reset_index()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_bar(x=ob["Overtime_Band"], y=ob["cost"], name="Mean labour cost (£)", marker_color="#c9d7ea", secondary_y=True)
        fig.add_scatter(x=ob["Overtime_Band"], y=ob["actual"], name="Mean actual productivity (%)", mode="lines+markers",
                        line=dict(color=CORAL, width=4), marker=dict(size=11))
        fig.add_scatter(x=ob["Overtime_Band"], y=100 * ob["met"], name="Hit rate (%)", mode="lines+markers",
                        line=dict(color=NAVY, width=3, dash="dot"), marker=dict(size=9))
        fig.update_yaxes(title_text="%", range=[40, 100], secondary_y=False)
        fig.update_yaxes(title_text="£ per crew-day", secondary_y=True, showgrid=False)
        fig.update_layout(title="More overtime = more cost, but actual productivity is flat", height=440)
        st.plotly_chart(fig, width="stretch")
        c1, c2 = st.columns(2)
        fig = px.scatter(D, x="Overtime_Hours", y="Actual_pct", color="Work_Package", trendline="lowess",
                         opacity=0.45, color_discrete_sequence=[NAVY, CORAL, TEAL],
                         title="Within each package, overtime barely moves productivity",
                         labels={"Overtime_Hours": "Overtime hours", "Actual_pct": "Actual productivity (%)", "Work_Package": ""})
        fig.update_layout(height=420)
        c1.plotly_chart(fig, width="stretch")
        rho = D.groupby("Work_Package")[["Overtime_Hours", "Actual_Productivity"]].corr(method="spearman").xs(
            "Overtime_Hours", level=1)["Actual_Productivity"].reset_index().rename(columns={"Actual_Productivity": "rho"})
        fig = px.bar(rho, x="Work_Package", y="rho", text=rho["rho"].map("{:+.2f}".format), color_discrete_sequence=[GOLD],
                     title="Spearman ρ: overtime vs actual productivity", labels={"rho": "ρ", "Work_Package": ""})
        fig.update_layout(yaxis_range=[-1, 1], height=420)
        c2.plotly_chart(fig, width="stretch")
        st.warning("**Misleading correlation.** Hit rate *rises* with overtime, but that's because big Structural "
                   "installation crews (which hit easier plans) work the most overtime — a **Simpson's paradox**. "
                   "Actual productivity doesn't improve, while cost per crew-day roughly quadruples.")

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
        st.warning("**Treat with caution.** The incentive association is huge, but incentives appear almost exclusively "
                   f"in Structural installation (share of days with an incentive: {share.to_dict()}). Bonuses are typically "
                   "*paid because* output was achieved, so the arrow likely points the other way. A randomised pilot "
                   "would be needed before claiming incentives *cause* higher productivity.")

    with tabs[4]:
        dc = A.group_summary(D, "Design_Change_Count")
        c1, c2 = st.columns([1.3, 1])
        fig = px.bar(dc, x="Design_Change_Count", y="Met_Rate_pct", text=dc["Met_Rate_pct"].map("{:.0f}%".format),
                     color_discrete_sequence=[SKY], hover_data=["Observations"],
                     title="Hit rate by number of design changes on the day",
                     labels={"Design_Change_Count": "Design changes", "Met_Rate_pct": "Hit rate (%)"})
        fig.update_layout(height=400, yaxis_range=[0, 100])
        c1.plotly_chart(fig, width="stretch")
        c2.markdown("#### Verdict")
        c2.markdown('<span class="pill myth">Not supported by the data</span>', unsafe_allow_html=True)
        c2.markdown(f"""
Management suspected that *changing requirements* disrupt delivery. In this dataset, days with 1–3 design changes
perform **no worse** than days with none (n = {', '.join(f"{int(r.Design_Change_Count)}: {r.Observations}" for r in dc.itertuples())}).

The model ranks design changes near the bottom of importance. Changes may still matter for *schedule* or *cost*
(not captured here), but they aren't where daily productivity is being lost.
""")

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
        fig = px.imshow(corr, color_continuous_scale="RdBu", zmin=-1, zmax=1, text_auto=".2f", aspect="auto",
                        title="Spearman rank correlations (descriptive — not causal)")
        fig.update_traces(textfont_size=9)
        fig.update_layout(height=680)
        st.plotly_chart(fig, width="stretch")

    with tabs[7]:
        st.markdown("#### Evidence scorecard — what management can and can't act on")
        rows = [
            ("Rework hours", "Strong negative", "Yes — all 3 packages", "#1 in model", "real", "Act: primary opportunity"),
            ("Idle time", "Severe on the day", "Yes", "Mid", "real", "Act: quick win (rare events)"),
            ("Crew identity", "Large spread", "Partly (mix of packages)", "Top 5", "caution", "Act: target pilot at lowest crews"),
            ("Overtime", "Hit rate ↑ with overtime", "No — flat within packages", "Low", "myth", "Don't buy more overtime; cap it"),
            ("Incentive £", "Very strong", "Only exists in one package", "Excluded (reverse causality)", "caution", "Test with a randomised pilot"),
            ("Design changes", "None", "—", "Very low", "myth", "Not a productivity lever here"),
            ("Weather / region / site access", "Small (±2pp)", "—", "Low", "myth", "Don't prioritise"),
            ("Delay risk score", "≈ 0", "—", "Excluded (illustrative)", "caution", "Not a validated measure"),
        ]
        label = {"real": "Genuine", "myth": "Misleading / weak", "caution": "Caution"}
        html = "<table style='width:100%;border-collapse:collapse;font-size:.92rem'><tr style='background:#16365c;color:white'>" + "".join(
            f"<th style='padding:8px;text-align:left'>{h}</th>" for h in
            ["Factor", "Raw association", "Holds within work package?", "Model importance", "Verdict", "Management action"]) + "</tr>"
        for i, (f, a, w, m, v, act) in enumerate(rows):
            bg = "#f5f8fc" if i % 2 else "white"
            html += (f"<tr style='background:{bg}'><td style='padding:8px'><b>{f}</b></td><td style='padding:8px'>{a}</td>"
                     f"<td style='padding:8px'>{w}</td><td style='padding:8px'>{m}</td>"
                     f"<td style='padding:8px'><span class='pill {v}'>{label[v]}</span></td><td style='padding:8px'>{act}</td></tr>")
        st.markdown(html + "</table>", unsafe_allow_html=True)


def page_recommend() -> None:
    hero("Q3 · What practical changes could improve performance?",
         "Two focused, evidence-backed opportunities — with an interactive what-if simulator.")

    o1, o2 = st.columns(2)
    with o1:
        st.markdown(f"""<div class="opp"><h3>① First-time-right rework programme</h3>
<span class="pill real">Primary</span><span class="pill real">Continuous Improvement Manager</span>
<p><b>What:</b> Pre-start quality briefings, hold-point inspections at critical steps, a 5-whys log for every rework
event, and buddying lowest-performing crews with top crews ({', '.join(CREWS.tail(3)['Crew_ID'])}).</p>
<p><b>Where first:</b> {', '.join(WORST_CREWS)} and Fit-out &amp; finishing.</p>
<p><b>Why:</b> Rework is the strongest and most consistent driver; the model shows the first hour of rework removed
gives the largest gain.</p></div>""", unsafe_allow_html=True)
    with o2:
        st.markdown(f"""<div class="opp"><h3>② Idle-time rapid response + smart overtime</h3>
<span class="pill real">Quick win</span><span class="pill caution">Cost release</span>
<p><b>What:</b> 30-minute escalation rule for any idle crew (materials/equipment/permit triage by the Operations
Manager), plus a planned-overtime cap with sign-off above the threshold.</p>
<p><b>Why:</b> Idle days hit target only {pct(IDLE.loc[True, 'met'], 0)} of the time. Overtime adds cost but not
productivity, so capping it funds the rework programme.</p></div>""", unsafe_allow_html=True)

    st.markdown("### 🧪 What-if simulator")
    st.caption("Applies the intervention to the selected crews/packages and re-scores every affected crew-day with the "
               "trained driver model. Results are modelled associations on synthetic data — use them to size a pilot, "
               "not as a guaranteed outcome.")
    c1, c2 = st.columns([1, 2.2])
    with c1:
        with st.container(border=True):
            crews = st.multiselect("Target crews", sorted(DF["Crew_ID"].unique()), default=WORST_CREWS)
            pkgs = st.multiselect("Target work packages", sorted(DF["Work_Package"].unique()), placeholder="All packages")
            cut = st.slider("Rework reduction (%)", 0, 80, 30, 5)
            idle_fix = st.toggle("Eliminate idle-time events", True)
            cap_on = st.toggle("Cap overtime per crew-day", True)
            cap = st.slider("Overtime cap (hrs/crew-day)", 20, 200, 110, 5, disabled=not cap_on)
            premium = st.slider("Overtime pay multiplier (assumption)", 1.0, 2.0, 1.5, 0.1, disabled=not cap_on)

    base = D.copy()
    scen = A.scenario(base, cut, idle_fix, cap if cap_on else None, crews, pkgs)
    tgt = scen["_targeted"]
    p0, p1 = MODELS.predict_met(base), MODELS.predict_met(scen)
    a0, a1 = MODELS.predict_actual(base), MODELS.predict_actual(scen)
    # Sequential contributions for the waterfall.
    s_rw = A.scenario(base, cut, False, None, crews, pkgs)
    s_idle = A.scenario(base, cut, idle_fix, None, crews, pkgs)
    p_rw, p_idle = MODELS.predict_met(s_rw), MODELS.predict_met(s_idle)
    hourly = base["Estimated_Labour_Cost_GBP"] / (base["Planned_Labour_Hours"] + base["Overtime_Hours"])
    ot_saved_hrs = (base["Overtime_Hours"] - scen["Overtime_Hours"]).clip(lower=0)
    saving = float((ot_saved_hrs * hourly * premium).sum()) if cap_on else 0.0

    with c2:
        k = st.columns(4)
        if tgt.any():
            kpi(k[0], "Targeted hit rate", pct(p1[tgt.values].mean()), f"+{100 * (p1 - p0)[tgt.values].mean():.1f}pp modelled", "good")
        else:
            kpi(k[0], "Targeted hit rate", "—", "no rows targeted", "good", GREY)
        kpi(k[1], "Overall hit rate", pct(p1.mean()), f"+{100 * (p1.mean() - p0.mean()):.1f}pp", "good")
        kpi(k[2], "Extra crew-days on target", f"{(p1 - p0).sum():,.0f}", f"over {len(base):,} days in view", "good")
        kpi(k[3], "Overtime cost released", f"£{saving / 1e3:,.0f}k", f"{ot_saved_hrs.sum():,.0f} hrs avoided", "gold", TEAL)

        wf = go.Figure(go.Waterfall(
            orientation="v", measure=["absolute", "relative", "relative", "relative", "total"],
            x=["Baseline (model)", "Rework cut", "Idle eliminated", "Overtime cap", "Scenario"],
            y=[100 * p0.mean(), 100 * (p_rw.mean() - p0.mean()), 100 * (p_idle.mean() - p_rw.mean()),
               100 * (p1.mean() - p_idle.mean()), 0],
            text=[f"{100 * p0.mean():.1f}%", f"{100 * (p_rw.mean() - p0.mean()):+.1f}", f"{100 * (p_idle.mean() - p_rw.mean()):+.1f}",
                  f"{100 * (p1.mean() - p_idle.mean()):+.1f}", f"{100 * p1.mean():.1f}%"],
            textposition="outside",
            increasing=dict(marker_color=TEAL), decreasing=dict(marker_color=CORAL), totals=dict(marker_color=NAVY),
            connector=dict(line=dict(color=GREY, dash="dot")),
        ))
        wf.update_layout(title="Where the modelled improvement comes from (overall hit rate, %)",
                         yaxis_range=[max(0, 100 * p0.mean() - 15), min(100, 100 * p1.mean() + 8)], height=360, showlegend=False)
        st.plotly_chart(wf, width="stretch")
        st.caption("ℹ️ Idle time shows a near-zero *modelled* uplift because only 18 crew-days had idle events — too few for the trees to learn from. The observed gap on those days is ~19pp (28% hit rate), so treat idle-time response as a targeted fix for the affected crews rather than a portfolio-wide lever.")

    by = pd.DataFrame({"Crew_ID": base["Crew_ID"], "Before": p0, "After": p1, "ActB": a0, "ActA": a1}).groupby("Crew_ID").mean() * 100
    by = by.sort_values("Before").reset_index()
    fig = go.Figure()
    fig.add_bar(x=by["Crew_ID"], y=by["Before"], name="Baseline", marker_color="#c9d7ea")
    fig.add_bar(x=by["Crew_ID"], y=by["After"], name="With intervention",
                marker_color=[CORAL if c in crews else NAVY for c in by["Crew_ID"]])
    fig.add_hline(y=100 * BEST_RATE, line_dash="dash", line_color=TEAL, annotation_text="Top-3 crew benchmark")
    fig.update_layout(barmode="group", title="Modelled hit rate by crew — before vs after", yaxis_title="Hit rate (%)",
                      height=420, yaxis_range=[40, 100])
    st.plotly_chart(fig, width="stretch")

    st.markdown("### 🗓️ Implementation roadmap")
    start = pd.Timestamp("2027-01-04")
    plan = pd.DataFrame([
        ("Baseline & data capture", "Measure", 0, 2, "Project Controls Analyst"),
        ("Rework root-cause logging live", "Opportunity ①", 1, 3, "Continuous Improvement Manager"),
        ("Hold-points & pre-start briefings (pilot crews)", "Opportunity ①", 2, 12, "Operations Manager"),
        ("Crew buddying with top performers", "Opportunity ①", 3, 12, "Resource Planning Manager"),
        ("Idle-time 30-min escalation rule", "Opportunity ②", 2, 12, "Operations Manager"),
        ("Overtime cap & sign-off", "Opportunity ②", 2, 12, "Project Delivery Lead"),
        ("Weekly SPC review", "Measure", 2, 14, "Project Controls Analyst"),
        ("Pilot evaluation & go/no-go", "Decision", 12, 14, "Senior Sponsor"),
        ("Scale to all crews", "Decision", 14, 26, "Senior Sponsor"),
    ], columns=["Task", "Stream", "s", "e", "Owner"])
    plan["Start"] = start + pd.to_timedelta(plan["s"] * 7, unit="D")
    plan["Finish"] = start + pd.to_timedelta(plan["e"] * 7, unit="D")
    fig = px.timeline(plan, x_start="Start", x_end="Finish", y="Task", color="Stream", hover_data=["Owner"],
                      color_discrete_map={"Measure": SKY, "Opportunity ①": CORAL, "Opportunity ②": GOLD, "Decision": NAVY},
                      title="26-week plan (illustrative start date)")
    fig.update_yaxes(autorange="reversed", title="")
    fig.update_layout(height=420)
    st.plotly_chart(fig, width="stretch")

    with st.expander("🚫 What we recommend *not* doing (and why)"):
        st.markdown("""
- **Buying more overtime to recover targets** — productivity is flat across overtime bands; it only adds cost.
- **Rolling out bigger bonuses on the strength of the incentive correlation** — likely reverse causality; test first.
- **Prioritising design-change control for productivity** — no measurable daily productivity effect in this data.
- **Re-planning by region / weather** — group differences are small (±2pp) compared with crew and rework effects.
""")


def page_measure() -> None:
    hero("Q4 · How should management measure the results?",
         "A KPI framework, statistical process control and a properly sized pilot.")
    pilot_crews = WORST_CREWS
    pc = DF[DF["Crew_ID"].isin(pilot_crews)]
    days = DF["Observation_Date"].nunique()

    st.markdown("### 📊 KPI framework")
    kpis = pd.DataFrame([
        ("Outcome (lagging)", "Target hit rate", "% crew-days with actual ≥ planned productivity", pct(pc["Met_Target"].mean()),
         f"≥ {pct(BEST_RATE, 0)} (top-3 crew level)", "Weekly", "Project Delivery Lead"),
        ("Outcome (lagging)", "Mean productivity variance", "Actual − planned (pp)", f"{pc['Variance_pp'].mean():+.1f}pp", "≥ 0pp", "Weekly", "Project Controls Analyst"),
        ("Driver (leading)", "Rework hours / crew-day", "Mean Rework_Hours", f"{pc['Rework_Hours'].mean():.2f}", f"≤ {0.7 * pc['Rework_Hours'].mean():.2f} (−30%)", "Daily", "Continuous Improvement Manager"),
        ("Driver (leading)", "Idle-time events", "Crew-days with idle time > 0", f"{int(pc['Had_Idle'].sum())} in {days} days", "0; each event root-caused in 24h", "Daily", "Operations Manager"),
        ("Cost guardrail", "Overtime £ per unit", "Overtime cost ÷ completed quantity (within unit type)", f"{pc['Overtime_Ratio'].mean():.0%} OT/planned hrs", "−15% with no fall in hit rate", "Monthly", "Resource Planning Manager"),
        ("Balancing", "Quality pass rate", "Quality_Pass_Rate_Percent", f"{pc['Quality_Pass_Rate_Percent'].mean():.1f}%", "No decrease", "Weekly", "Continuous Improvement Manager"),
        ("Balancing", "Safety stand-down hours", "Safety_Standdown_Hours", f"{pc['Safety_Standdown_Hours'].mean():.2f}/day", "No increase", "Weekly", "Operations Manager"),
        ("Benefit", "Value realised", "Extra on-target days + overtime £ released", "—", "Reported to sponsor", "Monthly", "Senior Sponsor"),
    ], columns=["Type", "KPI", "Definition", f"Baseline ({', '.join(pilot_crews)})", "Target", "Frequency", "Owner"])
    st.dataframe(kpis, hide_index=True, width="stretch")

    st.markdown("### 📈 Statistical process control (p-chart)")
    c1, c2 = st.columns([3, 1])
    with c2:
        freq = st.radio("Granularity", ["Daily", "Weekly"], horizontal=True)
        scope = st.radio("Scope", ["Filtered view", "Pilot crews"], horizontal=False)
    data = D if scope == "Filtered view" else pc
    pch = A.p_chart(data, "D" if freq == "Daily" else "W")
    with c1:
        fig = go.Figure()
        fig.add_scatter(x=pch["Observation_Date"], y=100 * pch["UCL"], line=dict(color=GREY, dash="dash", width=1), name="UCL")
        fig.add_scatter(x=pch["Observation_Date"], y=100 * pch["LCL"], line=dict(color=GREY, dash="dash", width=1), name="LCL",
                        fill="tonexty", fillcolor="rgba(58,134,255,0.07)")
        fig.add_scatter(x=pch["Observation_Date"], y=100 * pch["CL"], line=dict(color=NAVY, width=2), name="Centre line")
        fig.add_scatter(x=pch["Observation_Date"], y=100 * pch["p"], mode="lines+markers", name="Hit rate",
                        line=dict(color=SKY, width=2), marker=dict(size=7, color=np.where(pch["Signal"], CORAL, SKY)))
        fig.update_layout(title="Hit rate with 3σ control limits — red points are special-cause signals",
                          yaxis_title="Hit rate (%)", height=420, yaxis_range=[-2, 102])
        st.plotly_chart(fig, width="stretch")
    c2.metric("Signals", int(pch["Signal"].sum()))
    c2.caption("After the intervention, look for a sustained shift (e.g. 8+ points above the old centre line) "
               "rather than reacting to single good or bad days.")

    st.markdown("### 🧮 Pilot design & sample size")
    c1, c2 = st.columns([1, 1.6])
    with c1:
        with st.container(border=True):
            p_base = st.slider("Baseline hit rate (pilot crews)", 0.3, 0.95, float(round(pc["Met_Target"].mean(), 2)), 0.01)
            p_goal = st.slider("Target hit rate", 0.3, 0.99, float(round(min(0.99, p_base + 0.10), 2)), 0.01)
            alpha = st.select_slider("Significance (α)", [0.01, 0.05, 0.1], 0.05)
            power = st.select_slider("Power", [0.7, 0.8, 0.9], 0.8)
            n_crews = st.number_input("Crews per arm", 1, 6, len(pilot_crews))
        n = A.two_proportion_sample_size(p_base, p_goal, alpha, power)
        per_crew_day = len(DF) / (DF["Crew_ID"].nunique() * days)
        weeks = n / (n_crews * per_crew_day * 6) if n else 0
    with c2:
        k = st.columns(2)
        kpi(k[0], "Crew-days needed per arm", f"{n:,}", f"to detect {100 * (p_goal - p_base):+.0f}pp", "navy", GREY)
        kpi(k[1], "Estimated pilot length", f"{weeks:.0f} weeks", f"{n_crews} crews/arm · ~{per_crew_day:.1f} obs/crew/day", "gold", GREY)
        grid = np.arange(0.03, 0.31, 0.01)
        sizes = [A.two_proportion_sample_size(p_base, min(0.99, p_base + g), alpha, power) for g in grid]
        fig = px.area(x=100 * grid, y=sizes, labels={"x": "Uplift to detect (pp)", "y": "Crew-days per arm"},
                      title="Smaller improvements need much longer pilots")
        fig.update_traces(line_color=NAVY, fillcolor="rgba(22,54,92,0.12)")
        fig.add_vline(x=100 * (p_goal - p_base), line_dash="dash", line_color=CORAL)
        fig.update_layout(height=300)
        st.plotly_chart(fig, width="stretch")
    st.markdown(f"""
**Evaluation approach (difference-in-differences):** run the intervention on **{', '.join(pilot_crews)}** and compare
their change in hit rate against matched comparison crews over the same weeks. This controls for seasonal, weather and
project-mix changes that would fool a simple before/after comparison. **Go/no-go rule:** scale if the pilot arm improves
by ≥ {100 * (p_goal - p_base):.0f}pp more than comparison crews, with quality and safety balancing KPIs no worse.
""")
    st.download_button("⬇️ Download KPI baseline (CSV)", kpis.to_csv(index=False).encode(), "kpi_baseline.csv", "text/csv")


PERSONAS = {
    "Project Delivery Lead": ("🎯", "Accountable for delivery against schedule and targets.",
        ["Hit rate is {met} overall; {worst} are the priority crews.",
         "Overtime is not recovering targets — cap it and use the saving to fund rework reduction.",
         "Approve a 12-week pilot with a clear go/no-go threshold."], "crew"),
    "Operations Manager": ("🛠️", "Responsible for day-to-day productivity and resource allocation.",
        ["Introduce a 30-minute escalation rule for idle crews.",
         "Daily rework-hours tracking at crew level is your best leading indicator.",
         "Fit-out & finishing has the widest downside — prioritise supervision there."], "idle"),
    "Project Controls Analyst": ("📈", "Monitors performance, trends and deviations from plan.",
        ["Track hit rate on a p-chart; react to signals, not noise.",
         "Use difference-in-differences against comparison crews to evaluate the pilot.",
         "Quarter5 and idle-time days are small samples — report confidence."], "spc"),
    "Resource Planning Manager": ("👷", "Allocates people and capacity.",
        ["Buddy lowest crews with top crews ({best}).",
         "Overtime scales cost ~4× across bands with flat productivity.",
         "Crew size bands show a mixed picture — don't resize crews on this evidence alone."], "overtime"),
    "Continuous Improvement Manager": ("♻️", "Identifies process improvements.",
        ["Rework is the #1 driver — run first-time-right with hold-points and 5-whys logging.",
         "The first hour of rework removed gives the biggest gain.",
         "Effect holds in every work package — a genuine, transferable opportunity."], "rework"),
    "Data Analyst": ("🧠", "Investigates performance patterns and presents findings.",
        ["Model AUC {auc:.2f}; leakage fields and incentive excluded by design.",
         "Overtime shows Simpson's paradox across work packages.",
         "All findings are associations on synthetic data — validate on real data."], "importance"),
    "Senior Sponsor": ("🏛️", "Approves interventions and monitors benefits.",
        ["Two focused opportunities: rework reduction and idle/overtime control.",
         "Benefits: extra on-target days plus overtime cost released.",
         "Decision point at week 12 with statistical evidence."], "benefit"),
}


def page_personas() -> None:
    hero("Persona lens", "The same evidence, tailored to each stakeholder in the brief.")
    who = st.selectbox("I am a…", list(PERSONAS))
    icon, role, points, view = PERSONAS[who]
    fmt = dict(met=pct(D["Met_Target"].mean()), worst=", ".join(WORST_CREWS), best=", ".join(CREWS.tail(3)["Crew_ID"]), auc=MODELS.auc)
    c1, c2 = st.columns([1, 1.6])
    with c1:
        st.markdown(f"## {icon} {who}")
        st.caption(role)
        for p in points:
            qcard("→", p.format(**fmt))
    with c2:
        if view == "crew":
            cs = A.group_summary(D, "Crew_ID").sort_values("Met_Rate_pct")
            fig = px.bar(cs, x="Crew_ID", y="Met_Rate_pct", color="Met_Rate_pct", color_continuous_scale=[CORAL, GOLD, TEAL],
                         title="Hit rate by crew", labels={"Met_Rate_pct": "Hit rate (%)", "Crew_ID": ""})
        elif view == "idle":
            fig = px.box(D, x="Work_Package", y="Variance_pp", color="Had_Idle", color_discrete_map={True: CORAL, False: NAVY},
                         title="Variance by package, idle vs non-idle days", labels={"Variance_pp": "pp", "Had_Idle": "Idle day", "Work_Package": ""})
        elif view == "spc":
            pch = A.p_chart(D, "W")
            fig = px.line(pch, x="Observation_Date", y=[pch["p"] * 100, pch["UCL"] * 100, pch["LCL"] * 100], markers=True,
                          title="Weekly hit rate with control limits", labels={"value": "%", "Observation_Date": ""})
        elif view == "overtime":
            ob = D.groupby("Overtime_Band", observed=True).agg(cost=("Estimated_Labour_Cost_GBP", "mean"), actual=("Actual_pct", "mean")).reset_index()
            fig = px.bar(ob, x="Overtime_Band", y="cost", color="actual", color_continuous_scale=[CORAL, TEAL], range_color=[60, 85],
                         title="Labour cost by overtime band (colour = productivity %)", labels={"cost": "£/crew-day", "Overtime_Band": ""})
        elif view == "rework":
            fig = px.histogram(D, x="Rework_Hours", color="Met_Target", barmode="overlay", nbins=40, opacity=.7,
                               color_discrete_map={True: TEAL, False: CORAL}, title="Rework hours: hit vs missed days",
                               labels={"Met_Target": "Met target"})
        elif view == "importance":
            imp = MODELS.importance.head(8).iloc[::-1]
            fig = px.bar(imp, x="Importance", y="Label", orientation="h", color_discrete_sequence=[NAVY], title="Driver importance")
        else:
            p0 = MODELS.predict_met(D)
            p1 = MODELS.predict_met(A.scenario(D, 30, True, 110, WORST_CREWS, None))
            fig = go.Figure(go.Indicator(mode="number+delta", value=100 * p1.mean(), number=dict(suffix="%"),
                                         delta=dict(reference=100 * p0.mean(), suffix="pp"),
                                         title=dict(text="Modelled hit rate with recommended package")))
        fig.update_layout(height=430)
        st.plotly_chart(fig, width="stretch")


def page_data() -> None:
    hero("Data, method & limitations", "Transparency for non-technical decision-makers.")
    c1, c2 = st.columns([1, 1.3])
    with c1:
        st.markdown("#### ✅ Data quality audit")
        st.dataframe(AUDIT, hide_index=True, width="stretch")
        st.markdown("#### ℹ️ About the data pack")
        st.dataframe(README, hide_index=True, width="stretch")
        st.dataframe(NOTES, hide_index=True, width="stretch")
    with c2:
        st.markdown("#### ⚠️ Limitations & guardrails")
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
        st.markdown("#### 🔧 Method")
        st.markdown("""
1. **Clean** — trim text, type-cast, check duplicates/ranges/variance arithmetic. No rows dropped or imputed.
2. **Describe** — hit rate, variance and lost productivity (sum of pp below plan) by crew, package, period, context.
3. **Explain** — rank correlations, within-package checks (to catch Simpson's paradox), gradient-boosted model with
   permutation importance on a hold-out set.
4. **Recommend** — focus on 1–2 genuine, actionable drivers; simulate with the model.
5. **Measure** — KPI tree, SPC chart, powered pilot with comparison crews.
""")
    st.markdown("#### 🗂️ Explore the cleaned data")
    cols = st.multiselect("Columns", list(A.NUMERIC_COLUMNS) + A.MODEL_CATEGORICAL,
                          default=["Crew_ID", "Work_Package", "Planned_Productivity", "Actual_Productivity",
                                   "Productivity_Variance", "Rework_Hours", "Idle_Time_Hours", "Overtime_Hours", "Incentive_GBP"])
    view = D[["Record_ID", "Observation_Date"] + cols]
    st.dataframe(view, hide_index=True, width="stretch", height=360)
    st.download_button("⬇️ Download filtered data (CSV)", view.to_csv(index=False).encode(), "productivity_filtered.csv", "text/csv")
    with st.expander("📖 Data dictionary"):
        st.dataframe(DICTIONARY, hide_index=True, width="stretch")


def days_span() -> str:
    return f"{(DF['Observation_Date'].max() - DF['Observation_Date'].min()).days + 1} days"


pg = st.navigation([
    st.Page(page_summary, title="Executive summary", icon="🏠", default=True),
    st.Page(page_gaps, title="Q1 · Where are the gaps?", icon="🔍", url_path="gaps"),
    st.Page(page_drivers, title="Q2 · What drives them?", icon="🧪", url_path="drivers"),
    st.Page(page_recommend, title="Q3 · Recommendations", icon="💡", url_path="recommendations"),
    st.Page(page_measure, title="Q4 · Measuring success", icon="📏", url_path="measure"),
    st.Page(page_personas, title="Persona lens", icon="👥", url_path="personas"),
    st.Page(page_data, title="Data & limitations", icon="📋", url_path="data"),
])
pg.run()
