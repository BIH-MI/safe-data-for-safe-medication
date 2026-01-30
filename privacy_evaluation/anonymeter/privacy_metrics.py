from typing import List, Tuple

import pandas as pd
from anonymeter.evaluators import SinglingOutEvaluator, InferenceEvaluator, LinkabilityEvaluator


def run_anonymeter_linkability(data_origin: pd.DataFrame, data_processed: pd.DataFrame, data_control: pd.DataFrame,
                               aux_columns: Tuple[List[str], List[str]], n_attacks: int = 400) -> pd.DataFrame:
    evaluator = LinkabilityEvaluator(ori=data_origin,
                                     syn=data_processed,
                                     control=data_control,
                                     n_attacks=n_attacks,
                                     aux_cols=aux_columns,
                                     n_neighbors=1)

    evaluator.evaluate(n_jobs=-2)

    risk = evaluator.risk(confidence_level=0.95)
    results = evaluator.results()
    results_df = pd.DataFrame()
    results_df["attack_rate success_rate"] = pd.Series(results.attack_rate.value)
    results_df["attack_rate error"] = pd.Series(results.attack_rate.error)
    results_df["baseline_rate success_rate"] = pd.Series(results.baseline_rate.value)
    results_df["baseline_rate error"] = pd.Series(results.baseline_rate.error)
    results_df["control_rate success_rate"] = pd.Series(results.control_rate.value)
    results_df["control_rate error"] = pd.Series(results.control_rate.error)
    results_df["n_attacks"] = pd.Series(results.n_attacks)
    results_df["n_baseline"] = pd.Series(results.n_baseline)
    results_df["n_control"] = pd.Series(results.n_control)
    results_df["n_success"] = pd.Series(results.n_success)
    results_df["priv_risk"] = pd.Series(risk.value)
    results_df["priv_risk_ci"] = pd.Series(risk.ci)
    results_df["can_be_used"] = results_df["baseline_rate success_rate"] < results_df["attack_rate success_rate"]
    return results_df.transpose()


def run_anonymeter_attribute_inference(data_origin: pd.DataFrame, data_processed: pd.DataFrame,
                                       data_control: pd.DataFrame,
                                       n_attacks: int = 400) -> pd.DataFrame:
    columns = list(set(data_origin.columns).intersection(data_processed.columns))
    results = []
    risks = []
    for secret in columns:
        aux_cols = [col for col in columns if col != secret]

        evaluator = InferenceEvaluator(ori=data_origin,
                                       syn=data_processed,
                                       control=data_control,
                                       aux_cols=aux_cols,
                                       secret=secret,
                                       n_attacks=n_attacks)
        evaluator.evaluate(n_jobs=-2)
        results.append((secret, evaluator.results()))
        risks.append((secret, evaluator.risk()))

    results_df = pd.DataFrame({k: __to_series(v, "inference") for k, v in dict(results).items()}).add_prefix(
        "inference_")
    results_df = results_df.transpose()
    results_df[["attack_rate success_rate", "attack_rate error"]] = pd.DataFrame(results_df['attack_rate'].tolist(),
                                                                                 index=results_df.index)
    results_df[["baseline_rate success_rate", "baseline_rate error"]] = pd.DataFrame(
        results_df['baseline_rate'].tolist(), index=results_df.index)
    results_df[["control_rate success_rate", "control_rate error"]] = pd.DataFrame(results_df['control_rate'].tolist(),
                                                                                   index=results_df.index)
    results_df.drop(["attack_rate", "baseline_rate", "control_rate"], inplace=True, axis=1)

    risks_df = pd.DataFrame({k: v for k, v in dict(risks).items()}).add_prefix(
        "inference_")
    risks_df.index = ["priv_risk", "priv_risk_ci"]
    risks_df = risks_df.transpose()
    results_df[["priv_risk", "priv_risk_ci"]] = risks_df[["priv_risk", "priv_risk_ci"]]

    results_df["can_be_used"] = results_df["baseline_rate success_rate"] < results_df["attack_rate success_rate"]

    return results_df.transpose()


def run_anonymeter_singlingout(data_origin: pd.DataFrame, data_processed: pd.DataFrame, data_control: pd.DataFrame,
                               mode: str = "univariate", n_attacks: int = 400) -> pd.DataFrame:
    evaluator = SinglingOutEvaluator(ori=data_origin,
                                     syn=data_processed,
                                     control=data_control,
                                     n_attacks=n_attacks,
                                     max_attempts=10000)
    evaluator.evaluate(mode=mode)

    risk = evaluator.risk(confidence_level=0.95)

    results = evaluator.results()
    results_df = pd.DataFrame()
    results_df["attack_rate success_rate"] = pd.Series(results.attack_rate.value)
    results_df["attack_rate error"] = pd.Series(results.attack_rate.error)
    results_df["baseline_rate success_rate"] = pd.Series(results.baseline_rate.value)
    results_df["baseline_rate error"] = pd.Series(results.baseline_rate.error)
    results_df["control_rate success_rate"] = pd.Series(results.control_rate.value)
    results_df["control_rate error"] = pd.Series(results.control_rate.error)
    results_df["n_attacks"] = pd.Series(results.n_attacks)
    results_df["n_baseline"] = pd.Series(results.n_baseline)
    results_df["n_control"] = pd.Series(results.n_control)
    results_df["n_success"] = pd.Series(results.n_success)
    results_df["priv_risk"] = pd.Series(risk.value)
    results_df["priv_risk_ci"] = pd.Series(risk.ci)
    results_df["can_be_used"] = results_df["baseline_rate success_rate"] < results_df["attack_rate success_rate"]
    return results_df.transpose()


def __to_series(class_object, name: str = "0"):
    series = pd.Series(class_object.__dict__, name=name)
    return series


def __format_results(res_link, res_inf, res_SO_uni, res_SO_multi, priv_mostly):
    results = res_inf
    results.insert(0, "inference average", results.transpose().mean(skipna=True).transpose())
    results.insert(0, "linkage", res_link[0])

    results.insert(1, "singling out_univariate", res_SO_uni[0])
    results.insert(2, "singling out_multivariate", res_SO_multi[0])

    holdout_res = priv_mostly.loc[0]
    return results.transpose(), holdout_res


def compare_dtypes(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    df_dtypes = pd.DataFrame()
    df_dtypes["dtypes 1"] = df1.dtypes
    df_dtypes["dtypes 2"] = df2.dtypes
    df_dtypes["equal"] = df_dtypes["dtypes 1"] == df_dtypes["dtypes 2"]
    return df_dtypes


def unify_columns(data_origin, data_processed, data_control):
    diff = set(data_origin.columns) - set(data_processed.columns)
    column_set = list(set(data_origin.columns) - diff)
    return data_origin[column_set], data_processed[column_set], data_control[column_set]


def unify_dtypes(data_origin, data_processed, data_control):
    df_dtypes = compare_dtypes(data_origin, data_processed)
    df_dtypes_control = compare_dtypes(data_origin, data_control)
    df_dtypes["dtypes 3"] = df_dtypes_control["dtypes 2"]
    df_dtypes["equal 3"] = df_dtypes_control["equal"]

    for attribute in df_dtypes[df_dtypes["equal"] == False].index:
        if attribute not in data_processed.columns:
            data_origin.drop(attribute, axis=1, inplace=True)
            data_control.drop(attribute, axis=1, inplace=True)
            continue
        try:
            if data_processed[attribute].isna().any():
                data_origin[attribute] = data_origin[attribute].astype(data_processed[attribute].dtype)
            else:
                data_processed[attribute] = data_processed[attribute].astype(data_origin[attribute].dtype)

            if data_origin[attribute].dtype == "datetime64[ns]":
                data_processed[attribute] = pd.to_datetime(data_processed[attribute])

        except:
            print(
                f"Failed to cast {data_processed[attribute].dtype} to {data_origin[attribute].dtype} for attribute {attribute}. "
                f"Trying other way around.")
            data_origin[attribute] = data_origin[attribute].astype(data_processed[attribute].dtype)
            data_control[attribute] = data_control[attribute].astype(data_processed[attribute].dtype)

    return data_origin, data_processed, data_control


def __summarize_results(res_link, res_inf, res_SO_uni, res_SO_multi):
    results = res_inf

    numeric = (results.transpose()
               .apply(pd.to_numeric, errors="coerce"))

    # skip NaNs so one missing value doesn't kill the whole mean
    average_inference = numeric.mean(skipna=True)

    results.insert(0, "inference average", average_inference)
    results.insert(0, "linkage", res_link[0])
    results.insert(1, "singling out_univariate", res_SO_uni[0])
    results.insert(2, "singling out_multivariate", res_SO_multi[0])

    return results.transpose()


def anonymeter_evaluation(data_origin, data_processed, data_control, aux_columns):
    data_origin, data_processed, data_control = unify_dtypes(data_origin, data_processed, data_control)

    for col in aux_columns[0].copy():
        if col not in data_processed.columns:
            aux_columns[0].remove(col)
            print(f"removed {col} from known aux columns for linkage.")
    for col in aux_columns[1].copy():
        if col not in data_processed.columns:
            aux_columns[1].remove(col)
            print(f"removed {col} from unknown aux columns for linkage.")

    res_Link = run_anonymeter_linkability(data_origin, data_processed, data_control, aux_columns, n_attacks=5)
    res_Inf = run_anonymeter_attribute_inference(data_origin, data_processed, data_control, n_attacks=5)
    res_SO_uni = run_anonymeter_singlingout(data_origin, data_processed, data_control, "univariate", n_attacks=5)
    res_SO_multi = run_anonymeter_singlingout(data_origin, data_processed, data_control, "multivariate", n_attacks=5)

    return __summarize_results(res_Link, res_Inf, res_SO_uni, res_SO_multi)
