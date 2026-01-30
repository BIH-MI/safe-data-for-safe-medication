import pandas as pd

from datetime import datetime

date_string_today = datetime.today().strftime('%Y-%m-%d')

def descriptive_stats(df: pd.DataFrame, reference_count:int, prefix: str = None) -> pd.Series:
    stat_names = ["count", "mean", "std", "min", "max"]

    def format_number(num):
        if isinstance(num, float):
            return float(f"{num:.4g}")
        return num

    _stats = df.select_dtypes(include="number").agg(stat_names).T
    _stats["missing_n"] = reference_count - _stats["count"]
    _stats["missing_perc"] = (_stats["missing_n"] / reference_count) * 100
    _stats = _stats.applymap(format_number)

    if prefix:
        _stats.columns = [f"{prefix}_{col}" for col in _stats.columns]

    return _stats

def fidelity_analysis(df_orig, df_synth, df_anon_cd, df_anon_ci):
    reference_count_orig = len(df_orig)

    general_stats = pd.concat([descriptive_stats(df_orig, reference_count_orig, "Original"),
                               descriptive_stats(df_synth, reference_count_orig,"Synthetic"),
                               descriptive_stats(df_anon_cd, reference_count_orig,"Dependent Anon"),
                               descriptive_stats(df_anon_ci, reference_count_orig,"Independent Anon")], axis=1)

    general_stats["Synthetic_abs_diff"] = general_stats["Original_mean"] - general_stats["Synthetic_mean"]
    general_stats["Dependent Anon_abs_diff"] = general_stats["Original_mean"] - general_stats["Dependent Anon_mean"]
    general_stats["Independent Anon_abs_diff"] = general_stats["Original_mean"] - general_stats["Independent Anon_mean"]

    return general_stats
