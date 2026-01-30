import os
import pandas as pd

from privacy_evaluation.anonymeter.privacy_metrics import anonymeter_evaluation
from datetime import datetime

cwd = os.path.dirname(__file__)

VERBOSE = False
DATE_TODAY = datetime.now().date()

# *************** global configuration ************** #

ORIGINAL_FILE = r'../../data/dummy_data_original.csv'
HOLDOUT_FILE = r'../../data/dummy_data_test.csv'
TRAIN_FILE = r'../../data/dummy_data_train.csv'
SYNTHETIC_FILE = r'../../data/dummy_data_synth.csv'
DEPENDENT_ANONYM_FILE = r'../../data/dummy_data_anonymized.csv'
INDEPENDENT_ANONYM_FILE = r'../../data/dummy_data_anonymized.csv'
ANON_TYPE = "K2 with t-closeness"
OUTPUT_PATH = r'results'

SLICE_1 = [
    "fu_bleeding_flg", "fu_death_flg", "fu_rec_vte_flg",
    "covar_CKD at least 2 codes_flg", "covar_Obesity _flg",
    "covar_Varicose veins/post-thrombotic syndrome_flg", "covar_Arterial hypertension_flg",
    "covar_Diabetes mellitus_flg", "covar_Moderate/severe liver disease_flg", "covar_IBD_flg",
    "covar_Cancer excl. non-melanoma skin cancer_flg", "covar_Bleeding_flg", "covar_Fracture_flg",
    "covar_Major surgery_flg", "covar_Oral contraceptives_flg",
    "covar_Tamoxifen_flg", "covar_Systemic corticosteroids_flg", "covar_PPIs_flg",
    "covar_NSAIDs_flg"
]

SLICE_2 = [
    "covar_SSRIs_flg", "sex_flg", "oac_flg",
    "covar_Congestive heart failure_flg",
    "covar_MI_flg", "covar_Stroke_flg", "covar_Hormone replacement therapy_flg"]

LINKAGE_CONFIG = [
    SLICE_1,
    SLICE_2
]


# **************************************************** #


def export_iterated_results(iterated_results: list[dict], folder_output):
    results_simplified = pd.DataFrame()

    def summarize(anonymeter_results: pd.DataFrame, iteration: str, protection: str):
        cols = [
            "attack_rate success_rate", "attack_rate error",
            "baseline_rate success_rate", "baseline_rate error",
            "control_rate success_rate", "control_rate error",
            "priv_risk", "priv_risk_ci", "can_be_used",
        ]
        _ret_val = anonymeter_results.loc[:, [c for c in cols if c in anonymeter_results.columns]].copy()
        _ret_val["iteration"] = iteration
        _ret_val["modality"] = protection
        return _ret_val

    for result in iterated_results:
        results_ori = summarize(result["anonymeter_origin"], result["iteration"], "original")
        results_anon_dep = summarize(result["anonymeter_anon_dep"], result["iteration"], "anonymized_dependent")
        results_anon_indep = summarize(result["anonymeter_anon_indep"], result["iteration"], "anonymized_independent")
        results_synth = summarize(result["anonymeter_synth"], result["iteration"], "synthetic")

        results_simplified = pd.concat(
            [results_simplified,
             pd.concat([results_ori, results_anon_dep, results_anon_indep, results_synth])])

    _result_path = os.path.join(folder_output, f"{DATE_TODAY}_anonymeter_summary.csv")
    results_simplified.to_csv(_result_path)
    return _result_path


def run_anonymeter_eval(df_orig, df_train, df_holdout, df_synth, df_anon_dep, df_anon_indep,
                        iterations: int = 10) -> str:
    iterated_results = []
    for n in range(0, iterations):
        print(f"Started iteration {n}")
        anonymeter_results_origin = anonymeter_evaluation(df_orig,
                                                          df_train,
                                                          df_holdout,
                                                          LINKAGE_CONFIG)

        anonymeter_results_synth = anonymeter_evaluation(df_orig,
                                                         df_synth,
                                                         df_holdout,
                                                         LINKAGE_CONFIG)

        anonymeter_results_anon_dep = anonymeter_evaluation(df_orig,
                                                            df_anon_dep,
                                                            df_holdout,
                                                            LINKAGE_CONFIG)

        anonymeter_results_anon_indep = anonymeter_evaluation(df_orig,
                                                              df_anon_indep,
                                                              df_holdout,
                                                              LINKAGE_CONFIG)

        iterated_results.append({
            "anonymeter_origin": anonymeter_results_origin,
            "anonymeter_anon_dep": anonymeter_results_anon_dep,
            "anonymeter_anon_indep": anonymeter_results_anon_indep,
            "anonymeter_synth": anonymeter_results_synth,
            "iteration": n
        })

    _result_path = export_iterated_results(iterated_results, OUTPUT_PATH)
    print("*** Anonymeter risk evaluation finished and results exported ***")
    print(f"Export path: {_result_path}")
    return _result_path


if __name__ == '__main__':
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Note: Please make sure that all datasets have an identical structure and attribute names.
    # The anonymized dataset columns should have identical datatypes as the original columns

    df_orig = pd.read_csv(ORIGINAL_FILE)
    df_train = pd.read_csv(TRAIN_FILE)
    df_holdout = pd.read_csv(HOLDOUT_FILE)
    df_anon_dep = pd.read_csv(DEPENDENT_ANONYM_FILE)
    df_anon_indep = pd.read_csv(INDEPENDENT_ANONYM_FILE)
    df_synth = pd.read_csv(SYNTHETIC_FILE)

    print("* start anonymeter risk evaluation *")
    _result_path = run_anonymeter_eval(df_orig, df_train, df_holdout, df_synth, df_anon_dep, df_anon_indep, iterations=2)
