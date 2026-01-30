import pandas as pd
import numpy as np
from patsy.highlevel import dmatrix
import statsmodels.api as sm
from lifelines import CoxPHFitter
from scipy.stats import gamma
from typing import List


def _poisson_exact_ci(n, pt, alpha=0.05):
    """
    Exact Poisson 2-sided CI for rate per unit time.
    """
    if n == 0:
        lower = 0.0
        upper = gamma.ppf(1 - alpha / 2, 1) / pt  # = -ln(alpha/2)/pt
    else:
        lower = gamma.ppf(alpha / 2, n) / pt
        upper = gamma.ppf(1 - alpha / 2, n + 1) / pt
    return lower, upper


def preprocess(df: pd.DataFrame):
    """
    Perform conversions and create derived variables similar to the R do_preprocessing().

    NOTE on missing values:
    In real extracts (especially anonymized data), missingness may be encoded as string tokens
    such as "NULL", "*", "NA", etc. In the original R workflow these were explicitly converted
    to NA before any summaries/models. This reference implementation assumes inputs are already
    cleaned; if not, add a global replacement step before calling `preprocess`.
    """
    df = df.copy()

    df["vte_index_dt"] = pd.to_datetime(df["vte_index_dt"])
    df["birth_qtr"] = pd.to_datetime(df["birth_qtr"])

    df["study_entry_dt"] = df["vte_index_dt"] + pd.Timedelta(days=15)
    df["study_entry_yr"] = df["study_entry_dt"].dt.year
    df["study_entry_dt"] = df["study_entry_dt"].dt.date

    # oac_grp
    df["oac_grp"] = df["oac_flg"].apply(lambda x: "DOAC" if x == 1 else "VKA")
    # PID
    df = df.reset_index(drop=True)
    df["PID"] = np.arange(1, len(df) + 1)

    df = calc_py_and_fu_end_dt(df, "bleeding_fu_length", "bleeding")
    df = calc_py_and_fu_end_dt(df, "death_fu_length", "death")
    return df


def calc_py_and_fu_end_dt(df, length_col, name):
    """
    In anonymized data many dates were transformed to day offset attributes and then generalized,
    which need to be transformed back to concrete dates. This might need to be adapted to actual data.
    """
    df[name + "_fu_end_dt"] = (pd.to_datetime(df["study_entry_dt"]) + pd.to_timedelta(df[length_col], unit="D")
                               - pd.Timedelta(days=1))
    df[name + "_fu_end_dt"] = df[name + "_fu_end_dt"].dt.date
    df[name + "_py"] = df[length_col] / 365.25
    return df


def compare_covar(df: pd.DataFrame, threshold=5.0):
    """
    Build a descriptive table analogous to the R compare_covar() -> t1
    """
    dataset = df.copy()
    covars = [c for c in dataset.columns if c.startswith("covar_") and c.endswith("_flg")]

    groups = {
        "Total": df,
        "DOAC": df[df["oac_grp"] == "DOAC"],
        "VKA": df[df["oac_grp"] == "VKA"]
    }

    results = []

    for cov in covars:
        cov_info = {"variable": cov}
        for grp_name, grp_df in groups.items():
            cov_info[f"{grp_name}_n"] = int((grp_df[cov] == 1).sum())
            cov_info[f"{grp_name}_pct"] = (cov_info[f"{grp_name}_n"] / grp_df.shape[0]) * 100

        # Flag based on TOTAL population
        cov_info["include"] = "include" if (
                (cov_info["Total_pct"] >= threshold) and (cov_info["DOAC_pct"] >= threshold) and (
                cov_info["VKA_pct"] >= threshold)) else "not include"

        results.append(cov_info)

    out = pd.DataFrame(results)

    return out


def calc_ps(df: pd.DataFrame):
    """
    Calculate propensity scores with logistic regression and IPTW weights.
    """
    dataset = df.copy()

    # Covariates used for the PS model (mirrors the R intent: covar_*_flg and covar_*_grp)
    covar_flg = [c for c in dataset.columns if c.startswith("covar_") and c.endswith("_flg")]
    covar_grp = [c for c in dataset.columns if c.startswith("covar_") and c.endswith("_grp")]

    # Age nonlinearity: original analysis used restricted cubic splines (rcs(age, 3)).
    # For portability we use a B-spline basis with 3 df.
    bs_age = dmatrix("bs(age, df=3, include_intercept=False)", dataset, return_type="dataframe")
    bs_age.columns = [f"spl_age_{i}" for i in range(bs_age.shape[1])]

    # Encode categorical *_grp covariates
    grp_dummies = pd.get_dummies(dataset[covar_grp], prefix=covar_grp, dummy_na=True) if covar_grp else pd.DataFrame(index=dataset.index)

    # PS design matrix
    X = pd.concat([
        bs_age,
        dataset[["Geschlecht"]],
        dataset[covar_flg] if covar_flg else pd.DataFrame(index=dataset.index),
        grp_dummies,
    ], axis=1)

    valid = X.dropna().index.intersection(dataset[dataset["oac_flg"].notna()].index)

    X = X.loc[valid]
    y = dataset.loc[valid, "oac_flg"].astype(int)
    model = sm.Logit(y, sm.add_constant(X))
    ps = model.fit(disp=False).predict(sm.add_constant(X))

    # Assign PS back to full dataset
    dataset["ps"] = np.nan
    dataset.loc[valid, "ps"] = ps

    p_DOAC = (dataset["oac_grp"] == "DOAC").mean()
    p_VKA = 1 - p_DOAC

    def iptw(row):
        if pd.isna(row["ps"]) or not (0 < row["ps"] < 1):
            return np.nan
        return p_DOAC / row["ps"] if row["oac_grp"] == "DOAC" else p_VKA / (1 - row["ps"])

    dataset["iptw"] = dataset.apply(iptw, axis=1)

    return dataset


def _hr_analysis_for_outcome(dataset: pd.DataFrame, outcome: str):
    event = f"fu_{outcome}_flg"
    duration = f"{outcome}_py"

    return_values = {}

    return_values["outcome"] = outcome
    return_values["n_doac"] = dataset[dataset["oac_grp"] == "DOAC"]["PID"].nunique()
    return_values["n_events_doac"] = dataset[dataset["oac_grp"] == "DOAC"][event].sum()
    py_sum_doac = dataset[dataset["oac_grp"] == "DOAC"][duration].sum()

    return_values["n_vka"] = dataset[dataset["oac_grp"] == "VKA"]["PID"].nunique()
    return_values["n_events_vka"] = dataset[dataset["oac_grp"] == "VKA"][event].sum()
    py_sum_vka = dataset[dataset["oac_grp"] == "VKA"][duration].sum()

    # Incidence rates are descriptive and computed on the *unweighted* cohort:
    # IR = events / total person-years * 100
    return_values["inci_rate_doac"] = (return_values["n_events_doac"] / py_sum_doac) * 100 if py_sum_doac else np.nan
    lower, upper = _poisson_exact_ci(return_values["n_events_doac"], py_sum_doac)
    return_values["CI_inci_rate_doac"] = f"({(lower * 100):0.2f} - {(upper * 100):0.2f})"

    return_values["inci_rate_vka"] = (return_values["n_events_vka"] / py_sum_vka) * 100 if py_sum_vka else np.nan
    lower, upper = _poisson_exact_ci(return_values["n_events_vka"], py_sum_vka)
    return_values["CI_inci_rate_vka"] = f"({(lower * 100):0.2f} - {(upper * 100):0.2f})"

    cph = CoxPHFitter()

    # unadjusted crude HR calculation
    cph.fit(dataset, duration_col=duration, event_col=event, formula="oac_flg")
    ci_l = cph.confidence_intervals_.loc["oac_flg", "95% lower-bound"]
    ci_u = cph.confidence_intervals_.loc["oac_flg", "95% upper-bound"]

    return_values["crude_hr"] = cph.hazard_ratios_.get("oac_flg", np.nan)
    return_values["CI_crude_hr"] = f"({ci_l:0.2f} - {ci_u:0.2f})"

    # Adjusted Cox model (illustrative implementation):
    # We use linear age + sex adjustment for stability in demo / dummy data.
    # NOTE: In the original analysis, age was modeled using restricted cubic splines (rcs(age, 3)).
    cph.fit(dataset, duration_col=duration, event_col=event, formula="age + Geschlecht + oac_flg")
    ci_l = cph.confidence_intervals_.loc["oac_flg", "95% lower-bound"]
    ci_u = cph.confidence_intervals_.loc["oac_flg", "95% upper-bound"]

    return_values["adj_hr"] = cph.hazard_ratios_.get("oac_flg", np.nan)
    return_values["CI_adj_hr"] = f"({ci_l:0.2f} - {ci_u:0.2f})"

    # iptw weighted HRs
    cph.fit(dataset, duration_col=duration, event_col=event, weights_col="iptw", formula="oac_flg")
    ci_l = cph.confidence_intervals_.loc["oac_flg", "95% lower-bound"]
    ci_u = cph.confidence_intervals_.loc["oac_flg", "95% upper-bound"]

    return_values["iptw_hr"] = cph.hazard_ratios_.get("oac_flg", np.nan)
    return_values["CI_iptw_hr"] = f"({ci_l:0.2f} - {ci_u:0.2f})"

    return pd.Series(return_values)


def calc_hazard_ratios(dataset_ps: pd.DataFrame):
    """
    Calculate incidence rates and Cox crude/adjusted HRs (age+sex) as in R calc_hazard_ratios().
    """

    hr_results = pd.DataFrame()
    hr_results["major bleeding"] = _hr_analysis_for_outcome(dataset_ps, "bleeding")
    hr_results["all-cause mortality"] = _hr_analysis_for_outcome(dataset_ps, "death")

    return hr_results.transpose()


def compare_smd(dataset_ps: pd.DataFrame, all_covars: List[str]):
    """
    Compute unweighted standardized mean differences (SMD).
      """
    ds = dataset_ps.copy()
    # Prepare columns: continuous vs binary vs categorical levels
    # binary flags end with _flg; categorical groups end with _grp
    bin_cols = [c for c in all_covars if c.endswith("_flg")]
    if "oac_flg" in bin_cols:
        bin_cols.remove("oac_flg")
    cont_cols = ["age"]

    # compute unweighted SMD for continuous
    rows = []
    for var in cont_cols:
        # weighted/unweighted means by treatment
        group_stats = ds.groupby("oac_flg")[var].agg(["mean", "std", "count"]).reset_index()
        if group_stats.shape[0] < 2:
            continue
        mean1 = group_stats.loc[group_stats["oac_flg"] == 1, "mean"].values[0]
        mean0 = group_stats.loc[group_stats["oac_flg"] == 0, "mean"].values[0]
        sd1 = group_stats.loc[group_stats["oac_flg"] == 1, "std"].values[0]
        sd0 = group_stats.loc[group_stats["oac_flg"] == 0, "std"].values[0]
        pooled = np.sqrt((sd1 ** 2 + sd0 ** 2) / 2)
        smd = (mean1 - mean0) / pooled if pooled > 0 else np.nan
        rows.append({"variable": var, "value": "", "n_mean_doac": mean1, "pct_sd_doac": sd1,
                     "n_mean_vka": mean0, "pct_sd_vka": sd0, "smd": smd})
    # binary variables
    for var in bin_cols:
        if var not in ds.columns:
            continue

        # compute prevalences 0/1
        p1 = ds.loc[ds["oac_flg"] == 1, var].mean()
        p0 = ds.loc[ds["oac_flg"] == 0, var].mean()
        smd = (p1 - p0) / np.sqrt(((p1 * (1 - p1)) + (p0 * (1 - p0))) / 2)
        rows.append({"variable": var.replace("covar_", "").replace("_flg", ""), "value": "", "n_mean_doac": p1,
                     "pct_sd_doac": p1 * 100,
                     "n_mean_vka": p0, "pct_sd_vka": p0 * 100, "smd": smd})

    t4 = pd.DataFrame(rows)

    return t4
