"""Data loading, cleaning and modelling helpers for the Productivity Detective dashboard."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

DATA_DIR = Path(__file__).resolve().parent / "Data"
RAW_PATH = DATA_DIR / "construction_workplace_productivity_pack.xlsx"

# A record "meets target" when actual productivity is at or above plan (0.1pp tolerance for rounding).
MET_TOLERANCE = 0.001

NUMERIC_COLUMNS = [
    "Crew_Size", "Planned_Productivity", "Actual_Productivity", "Productivity_Variance",
    "Planned_Labour_Hours", "Overtime_Hours", "Estimated_Labour_Cost_GBP", "Incentive_GBP",
    "Idle_Time_Hours", "Idle_Workers", "Standard_Labour_Minutes_Per_Unit", "Work_In_Progress_Units",
    "Planned_Quantity", "Completed_Quantity", "Progress_Percent", "Material_Availability_Percent",
    "Equipment_Availability_Percent", "Equipment_Downtime_Hours", "Temperature_C", "Rainfall_mm",
    "Wind_Speed_kmh", "Safety_Incidents", "Safety_Standdown_Hours", "Rework_Hours",
    "Rework_Rate_Percent", "Workforce_Absence_Rate_Percent", "Average_Workforce_Experience_Years",
    "Training_Hours", "Design_Change_Count", "Supervisor_Experience_Years",
    "Quality_Pass_Rate_Percent", "Delay_Risk_Score",
]

# Operational drivers used by the model. Outcome-derived fields (actual, variance, completed quantity,
# progress %) are excluded to prevent leakage. Incentive is excluded because it may be paid *because*
# a target was hit (reverse causality); Delay_Risk_Score is illustrative only.
MODEL_NUMERIC = [
    "Crew_Size", "Planned_Productivity", "Planned_Labour_Hours", "Overtime_Hours",
    "Idle_Time_Hours", "Idle_Workers", "Standard_Labour_Minutes_Per_Unit",
    "Material_Availability_Percent", "Equipment_Availability_Percent", "Equipment_Downtime_Hours",
    "Temperature_C", "Rainfall_mm", "Wind_Speed_kmh", "Safety_Incidents", "Safety_Standdown_Hours",
    "Rework_Hours", "Rework_Rate_Percent", "Workforce_Absence_Rate_Percent",
    "Average_Workforce_Experience_Years", "Training_Hours", "Design_Change_Count",
    "Supervisor_Experience_Years", "Quality_Pass_Rate_Percent",
]
MODEL_CATEGORICAL = [
    "Crew_ID", "Work_Package", "Work_Package_Complexity", "Quarter", "Day_of_Week",
    "Site_Region", "Project_Type", "Project_Phase", "Weather_Conditions", "Site_Constraints",
    "Site_Access_Condition",
]

PRETTY = {
    "Rework_Hours": "Rework hours",
    "Rework_Rate_Percent": "Rework rate (%)",
    "Idle_Time_Hours": "Idle time (hrs)",
    "Idle_Workers": "Idle workers",
    "Overtime_Hours": "Overtime (hrs)",
    "Incentive_GBP": "Incentive (£)",
    "Design_Change_Count": "Design changes",
    "Crew_Size": "Crew size",
    "Planned_Productivity": "Planned productivity",
    "Planned_Labour_Hours": "Planned labour hours",
    "Standard_Labour_Minutes_Per_Unit": "Std labour mins / unit",
    "Material_Availability_Percent": "Material availability (%)",
    "Equipment_Availability_Percent": "Equipment availability (%)",
    "Equipment_Downtime_Hours": "Equipment downtime (hrs)",
    "Temperature_C": "Temperature (°C)",
    "Rainfall_mm": "Rainfall (mm)",
    "Wind_Speed_kmh": "Wind speed (km/h)",
    "Safety_Incidents": "Safety incidents",
    "Safety_Standdown_Hours": "Safety stand-down (hrs)",
    "Workforce_Absence_Rate_Percent": "Absence rate (%)",
    "Average_Workforce_Experience_Years": "Workforce experience (yrs)",
    "Training_Hours": "Training hours",
    "Supervisor_Experience_Years": "Supervisor experience (yrs)",
    "Quality_Pass_Rate_Percent": "Quality pass rate (%)",
    "Estimated_Labour_Cost_GBP": "Labour cost (£)",
    "Crew_ID": "Crew",
    "Work_Package": "Work package",
    "Work_Package_Complexity": "Complexity",
    "Quarter": "Period",
    "Day_of_Week": "Day of week",
    "Site_Region": "Region",
    "Project_Type": "Project type",
    "Project_Phase": "Project phase",
    "Weather_Conditions": "Weather",
    "Site_Constraints": "Site constraints",
    "Site_Access_Condition": "Site access",
    "Delay_Risk_Score": "Delay risk score",
}


def pretty(name: str) -> str:
    return PRETTY.get(name, name.replace("_", " "))


def load_workbook() -> dict[str, pd.DataFrame]:
    return pd.read_excel(RAW_PATH, sheet_name=None, engine="openpyxl")


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Row-preserving clean mirroring the notebook; returns (clean frame, audit table)."""
    df = raw.copy(deep=True)
    df.columns = df.columns.astype(str).str.strip()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("string").str.strip().replace("", pd.NA)
    df["Observation_Date"] = pd.to_datetime(df["Observation_Date"], errors="coerce")
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    expected = df["Actual_Productivity"] - df["Planned_Productivity"]
    variance_mismatch = ~np.isclose(df["Productivity_Variance"], expected, atol=0.001)
    out_of_range = ~df["Actual_Productivity"].between(0, 1) | ~df["Planned_Productivity"].between(0, 1)

    audit = pd.DataFrame(
        [
            ("Rows in source", len(raw)),
            ("Rows after cleaning (none dropped)", len(df)),
            ("Duplicate Record_IDs", int(df["Record_ID"].duplicated().sum())),
            ("Exact duplicate rows", int(df.duplicated().sum())),
            ("Productivity outside 0–1", int(out_of_range.sum())),
            ("Variance ≠ actual − planned", int(variance_mismatch.sum())),
            ("Missing Work_In_Progress_Units (non-structural packages)", int(df["Work_In_Progress_Units"].isna().sum())),
        ],
        columns=["Check", "Count"],
    )

    df["Met_Target"] = df["Productivity_Variance"] >= -MET_TOLERANCE
    df["Missed_Target"] = ~df["Met_Target"]
    df["Variance_pp"] = 100 * df["Productivity_Variance"]
    df["Actual_pct"] = 100 * df["Actual_Productivity"]
    df["Planned_pct"] = 100 * df["Planned_Productivity"]
    # Shortfall = percentage points below plan on days that missed (0 otherwise).
    df["Shortfall_pp"] = (-df["Variance_pp"]).clip(lower=0)
    df["Had_Idle"] = df["Idle_Time_Hours"] > 0
    df["Has_Incentive"] = df["Incentive_GBP"] > 0
    df["Overtime_Ratio"] = df["Overtime_Hours"] / df["Planned_Labour_Hours"].where(df["Planned_Labour_Hours"] > 0)
    df["Cost_per_Unit_GBP"] = df["Estimated_Labour_Cost_GBP"] / df["Completed_Quantity"].where(df["Completed_Quantity"] > 0)
    df["Crew_Size_Band"] = pd.cut(
        df["Crew_Size"], bins=[0, 10, 20, 35, 50, np.inf],
        labels=["1–10", "11–20", "21–35", "36–50", "51+"], include_lowest=True,
    ).astype(str)
    df["Rework_Band"] = pd.qcut(df["Rework_Hours"], 5, labels=["Very low", "Low", "Medium", "High", "Very high"]).astype(str)
    df["Overtime_Band"] = pd.qcut(df["Overtime_Hours"], 5, labels=["Q1 (least)", "Q2", "Q3", "Q4", "Q5 (most)"], duplicates="drop").astype(str)
    df["Week"] = df["Observation_Date"].dt.to_period("W").dt.start_time
    return df, audit


def group_summary(df: pd.DataFrame, by: str | list[str], min_n: int = 1) -> pd.DataFrame:
    g = df.groupby(by, observed=True, dropna=False).agg(
        Observations=("Record_ID", "count"),
        Met_Rate=("Met_Target", "mean"),
        Mean_Actual_pct=("Actual_pct", "mean"),
        Mean_Planned_pct=("Planned_pct", "mean"),
        Mean_Variance_pp=("Variance_pp", "mean"),
        Total_Shortfall_pp=("Shortfall_pp", "sum"),
        Mean_Rework_Hours=("Rework_Hours", "mean"),
        Idle_Hours=("Idle_Time_Hours", "sum"),
        Mean_Overtime=("Overtime_Hours", "mean"),
        Labour_Cost=("Estimated_Labour_Cost_GBP", "sum"),
    ).reset_index()
    g["Met_Rate_pct"] = 100 * g["Met_Rate"]
    g["Missed_Days"] = (g["Observations"] * (1 - g["Met_Rate"])).round().astype(int)
    total = g["Total_Shortfall_pp"].sum()
    g["Share_of_Shortfall_pct"] = 100 * g["Total_Shortfall_pp"] / total if total else 0.0
    return g[g["Observations"] >= min_n]


def _features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[MODEL_NUMERIC + MODEL_CATEGORICAL].copy()
    for c in MODEL_CATEGORICAL:
        X[c] = X[c].astype("category")
    return X


class DriverModels:
    """Gradient-boosted models of target achievement and actual productivity from operational drivers."""

    def __init__(self, df: pd.DataFrame, seed: int = 42):
        # Unbounded OpenMP threads make these small fits ~10x slower on many-core machines.
        with threadpool_limits(4):
            self._fit(df, seed)

    def _fit(self, df: pd.DataFrame, seed: int) -> None:
        X = _features(df)
        self.categories = {c: X[c].cat.categories for c in MODEL_CATEGORICAL}
        y_cls = df["Met_Target"].astype(int)
        y_reg = df["Actual_Productivity"]
        X_tr, X_te, yc_tr, yc_te, yr_tr, yr_te = train_test_split(
            X, y_cls, y_reg, test_size=0.25, random_state=seed, stratify=y_cls
        )
        params = dict(categorical_features="from_dtype", max_iter=300, learning_rate=0.05,
                      max_leaf_nodes=15, l2_regularization=1.0, random_state=seed)
        self.clf = HistGradientBoostingClassifier(**params).fit(X_tr, yc_tr)
        self.reg = HistGradientBoostingRegressor(**params).fit(X_tr, yr_tr)
        self.auc = roc_auc_score(yc_te, self.clf.predict_proba(X_te)[:, 1])
        self.mae_pp = 100 * mean_absolute_error(yr_te, self.reg.predict(X_te))
        self.baseline_mae_pp = 100 * mean_absolute_error(yr_te, np.full(len(yr_te), yr_tr.mean()))
        self.test_size = len(X_te)

        imp = permutation_importance(self.clf, X_te, yc_te, scoring="roc_auc", n_repeats=10, random_state=seed)
        self.importance = (
            pd.DataFrame({"Feature": X.columns, "Importance": imp.importances_mean, "Std": imp.importances_std})
            .assign(Label=lambda t: t["Feature"].map(pretty))
            .sort_values("Importance", ascending=False)
            .reset_index(drop=True)
        )

    def _prep(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[MODEL_NUMERIC + MODEL_CATEGORICAL].copy()
        for c in MODEL_CATEGORICAL:
            X[c] = pd.Categorical(X[c].astype(str), categories=self.categories[c])
        return X

    def predict_met(self, df: pd.DataFrame) -> np.ndarray:
        with threadpool_limits(4):
            return self.clf.predict_proba(self._prep(df))[:, 1]

    def predict_actual(self, df: pd.DataFrame) -> np.ndarray:
        with threadpool_limits(4):
            return self.reg.predict(self._prep(df))

    def response_curve(self, df: pd.DataFrame, feature: str, grid: np.ndarray) -> pd.DataFrame:
        """Average predicted met-rate when `feature` is set to each grid value (partial dependence)."""
        rows = []
        for v in grid:
            scenario = df.copy()
            scenario[feature] = v
            rows.append({"value": v, "met_rate": 100 * self.predict_met(scenario).mean()})
        return pd.DataFrame(rows)


def scenario(df: pd.DataFrame, rework_cut_pct: float, remove_idle: bool, overtime_cap: float | None,
             crews: list[str] | None, packages: list[str] | None) -> pd.DataFrame:
    """Apply hypothetical adjustments to selected rows and return the modified copy."""
    s = df.copy()
    mask = pd.Series(True, index=s.index)
    if crews:
        mask &= s["Crew_ID"].isin(crews)
    if packages:
        mask &= s["Work_Package"].isin(packages)
    factor = 1 - rework_cut_pct / 100
    s.loc[mask, "Rework_Hours"] *= factor
    s.loc[mask, "Rework_Rate_Percent"] *= factor
    if remove_idle:
        s.loc[mask, ["Idle_Time_Hours", "Idle_Workers"]] = 0
    if overtime_cap is not None:
        s.loc[mask, "Overtime_Hours"] = s.loc[mask, "Overtime_Hours"].clip(upper=overtime_cap)
    s["_targeted"] = mask
    return s


def p_chart(df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """p-chart limits for the share of observations meeting target per period."""
    g = df.set_index("Observation_Date").resample(freq).agg(n=("Met_Target", "size"), met=("Met_Target", "sum"))
    g = g[g["n"] > 0].reset_index()
    p_bar = g["met"].sum() / g["n"].sum()
    g["p"] = g["met"] / g["n"]
    sigma = np.sqrt(p_bar * (1 - p_bar) / g["n"])
    g["CL"] = p_bar
    g["UCL"] = np.minimum(1, p_bar + 3 * sigma)
    g["LCL"] = np.maximum(0, p_bar - 3 * sigma)
    g["Signal"] = (g["p"] > g["UCL"]) | (g["p"] < g["LCL"])
    return g


def two_proportion_sample_size(p1: float, p2: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Observations needed per arm to detect a change from p1 to p2 (two-sided)."""
    from scipy.stats import norm

    if p1 == p2:
        return 0
    z_a, z_b = norm.ppf(1 - alpha / 2), norm.ppf(power)
    p_bar = (p1 + p2) / 2
    n = ((z_a * np.sqrt(2 * p_bar * (1 - p_bar)) + z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / (p1 - p2) ** 2
    return int(np.ceil(n))
