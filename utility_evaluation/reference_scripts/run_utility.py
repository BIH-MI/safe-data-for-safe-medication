import os
from datetime import datetime
import pandas as pd

from utility_evaluation.reference_scripts.fidelity_metrics import fidelity_analysis
from utility_evaluation.reference_scripts.utility_metrics import preprocess, compare_covar, calc_ps, \
    calc_hazard_ratios, compare_smd

# *************** global configuration ************** #
ORIGINAL_FILE = r'../../data/dummy_data_original.csv'
SYNTHETIC_FILE = r'../../data/dummy_data_synth.csv'
DEPENDENT_ANONYM_FILE = r'../../data/dummy_data_anonymized.csv'
INDEPENDENT_ANONYM_FILE = r'../../data/dummy_data_anonymized.csv'
OUTPUT_PATH = r'results'

DATE_TODAY = datetime.now().date()
# **************************************************** #


def utility_analysis(df: pd.DataFrame, name: str):
    # Missingness note:
    # In some datasets (esp. anonymized exports) missing values may be encoded as strings like
    # "NULL", "*", "NA", etc. The original R workflow normalized these to NA before analysis.
    # For reproducible behavior, you may want to normalize such tokens here.
    df = df.replace({"*": pd.NA, "NULL": pd.NA, "nan": pd.NA, "NaN": pd.NA, "NA": pd.NA})

    df_clean = preprocess(df)
    covar_selection = compare_covar(df_clean)

    selected_variables = list(set(df_clean.columns) - set(
        covar_selection[covar_selection["include"] == "not include"]["variable"]))
    # selected_variables.remove("PID")
    # Keep Geschlecht in the PS model (as in the original R implementation)

    ps_data = calc_ps(df_clean[selected_variables])

    hazard_ratio_results = calc_hazard_ratios(ps_data)
    hazard_ratio_results.to_csv(fr"{OUTPUT_PATH}/{DATE_TODAY}_HR_{name}.csv", index=False)

    smd_variables = ['age'] + [var for var in df_clean.columns if var.endswith("_flg")]
    smd_results = compare_smd(ps_data, smd_variables)
    smd_results.to_csv(fr"{OUTPUT_PATH}/{DATE_TODAY}_SMD_{name}.csv", index=False)
    return hazard_ratio_results, smd_results


if __name__ == '__main__':
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH, exist_ok=True)

    df_orig = pd.read_csv(ORIGINAL_FILE)
    df_synth = pd.read_csv(SYNTHETIC_FILE)
    df_anon_cd = pd.read_csv(DEPENDENT_ANONYM_FILE)
    df_anon_ci = pd.read_csv(INDEPENDENT_ANONYM_FILE)

    # Full dataset
    fidelity_results = fidelity_analysis(df_orig, df_synth, df_anon_cd, df_anon_ci)
    fidelity_results.to_csv(os.path.join(OUTPUT_PATH, f"{DATE_TODAY}_fidelity_summary_full.csv"))

    # DOAC subgroup
    fidelity_results_doac = fidelity_analysis(df_orig[df_orig["oac_flg"] == 1],
                                              df_synth[df_synth["oac_flg"] == 1],
                                              df_anon_cd[df_anon_cd["oac_flg"] == 1],
                                              df_anon_ci[df_anon_ci["oac_flg"] == 1])
    fidelity_results_doac.to_csv(os.path.join(OUTPUT_PATH, f"{DATE_TODAY}_fidelity_summary_DOAC.csv"))

    # VKA subgroup
    fidelity_results_vka = fidelity_analysis(df_orig[df_orig["oac_flg"] == 0],
                                             df_synth[df_synth["oac_flg"] == 0],
                                             df_anon_cd[df_anon_cd["oac_flg"] == 0],
                                             df_anon_ci[df_anon_ci["oac_flg"] == 0])
    fidelity_results_vka.to_csv(os.path.join(OUTPUT_PATH, f"{DATE_TODAY}_fidelity_summary_VKA.csv"))

    hr_original, smd_original = utility_analysis(df_orig, "orig")
    hr_synth, smd_synth = utility_analysis(df_synth, "synth")
    hr_anon_cd, smd_anon_cd = utility_analysis(df_anon_cd, "anon context dependent")
    hr_anon_ci, smd_anon_ci = utility_analysis(df_anon_ci, "anon context independent")
