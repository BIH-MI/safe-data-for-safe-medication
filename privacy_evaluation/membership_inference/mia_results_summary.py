import json
import os
import warnings
from glob import glob

import numpy as np
import yaml
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

CWD = os.path.dirname(__file__)

"""
This script assumes that shadow-model and Phantom Anonymization evaluations have already been executed using the 
respective upstream frameworks.

It then shows how the membership inference results of the two frameworks are unified.
"""

# *************** global configuration ************** #

PA_FOLDER = r"path/to/PhantomAnonymization/results"
SM_FOLDER = r".path/to/ShadowModel/results"
OUTPUT_PATH = r'./results'

TARGET_MODELS = {
    'IDENTITY': "Original",
    'K2-CD': "50% max. Risk  (context-dependent)",
    'K2-CI': "50% max. Risk  (context-independent)",
    'Mostly': "MostlyAI"
}

# **************************************************** #

LABEL_IN = 1
LABEL_OUT = 0


def get_mia_advantage(tp_rate, fp_rate):
    return tp_rate - fp_rate


def get_tp_fp_rates(guesses, labels):
    targetIn = np.where(labels == LABEL_IN)[0]
    targetOut = np.where(labels == LABEL_OUT)[0]
    return sum(guesses.iloc[targetIn] == LABEL_IN) / len(targetIn), sum(guesses.iloc[targetOut] == LABEL_IN) / len(
        targetOut)


def load_summary_file(summary_file):
    return pd.read_csv(summary_file, sep=";")


def load_config_file(config_file):
    with open(config_file) as f:
        config = yaml.safe_load(f)
    return config


def load_shadow_model_results(dirname):
    """
    Helper function to load results of privacy evaluation under risk of linkability
    :param dirname: str: Directory that contains results files
    :return: results: DataFrame: Results of privacy evaluation
    """

    files = glob(os.path.join(dirname, f'*ResultsMIA_*.json'))

    index = 0
    resList = []
    for fpath in files:
        index += 1
        with open(fpath) as f:
            resDict = json.load(f)

        dataset = fpath.split('.json')[0].split("ResultsMIA_")[1].split('_')[0]

        for tid, tres in resDict.items():
            for gm, gmDict in tres.items():
                for nr, nrDict in gmDict.items():
                    for fset, fsetDict in nrDict.items():
                        df = pd.DataFrame(fsetDict)

                        df['Run'] = f"{index}_{nr}"
                        df['FeatureSet'] = fset
                        df['TargetModel'] = gm
                        df['TargetID'] = tid
                        df['Dataset'] = dataset

                        resList.append(df)

    results = pd.concat(resList)

    return aggregate_mia_results(results)


def aggregate_mia_results(results):
    resAgg = []
    games = results.groupby(['TargetID', 'TargetModel', 'FeatureSet', 'Run'])
    for gameParams, gameRes in games:
        tpSyn, fpSyn = get_tp_fp_rates(gameRes['AttackerGuess'], gameRes['Secret'])
        advantageSyn = get_mia_advantage(tpSyn, fpSyn)
        advantageRaw = 1

        resAgg.append(gameParams + (tpSyn, fpSyn, advantageSyn, advantageRaw))
    resAgg = pd.DataFrame(resAgg)
    resAgg.columns = ['TargetID', 'TargetModel', 'FeatureSet', 'Run', 'TPSyn', 'FPSyn', 'AdvantageSyn', 'AdvantageRaw']
    resAgg['PrivacyGain'] = resAgg['AdvantageRaw'] - resAgg['AdvantageSyn']
    return resAgg


def load_phantom_anon_results(dirname):
    """
    Helper function to load results of privacy evaluation under risk of linkability
    :param dirname: str: Directory that contains results files
    :return: results: DataFrame: Results of privacy evaluation
    """

    files = glob(os.path.join(dirname, f'*_log.csv'))

    resList = []
    for fpath in files:
        summary_file = fpath.replace("_log.csv", "_summary.txt")
        summary = load_summary_file(summary_file)

        config_file = fpath.replace("_log.csv", "_cfgs.yml")
        config = load_config_file(config_file)

        with open(fpath) as f:
            _temp = pd.read_csv(f, sep=";")
            _temp["FeatureSet"] = ", ".join(config["riskAssessmentConfig"]["featureTypes"])
            _temp["TargetModel"] = config["anonymizationConfig"]["name"]
            _temp["PrivacyModel"] = str(config["anonymizationConfig"]["privacyModelList"])
            resList.append(_temp)

    results = pd.concat(resList)
    results.rename(columns={
        "TargetId": "TargetID",
        "TestRun": "Run",
        "PredictedLabel": "AttackerGuess",
        "TrueLabel": "Secret"
    }, inplace=True)
    results["TargetID"] = "ID" + results["TargetID"].astype(str)
    return aggregate_mia_results(results)


def set_target_type(targetIds: pd.Series, target_dict: dict):
    """ Returns the target type (outlier or average) as a pandas data series from a given data series containing
    record IDs and the target dictionary"""

    reverse_dict = {id_: category for category, ids in target_dict.items() for id_ in ids}
    return targetIds.map(reverse_dict)


def filter_and_merge_results(phantom_anon_results: pd.DataFrame, shadow_model_results: pd.DataFrame,
                             target_dictionary=None):
    mia_results = pd.concat([phantom_anon_results, shadow_model_results], axis=0)
    mia_results["TargetModel"] = mia_results["TargetModel"].map(TARGET_MODELS)

    # filter mia_results to just the ones in the target_dictionary
    if target_dictionary:
        targets = [id for ids in target_dictionary.values() for id in ids]
        mia_results = mia_results[mia_results['TargetID'].isin(targets)]
        mia_results["TargetType"] = set_target_type(mia_results["TargetID"], target_dictionary)

    ## format
    mia_results["FeatureSet"] = mia_results["FeatureSet"].str.upper().replace("CORRELATIONS", "CORRELATION")
    return mia_results


def summarize_mia_results(mia_results: pd.DataFrame):
    return mia_results.groupby(["TargetID", "TargetModel", "FeatureSet"]).describe()


if __name__ == "__main__":
    _target_dictionary = {
        "Average": ["ID001", "ID002", "..."],
        "Outlier": ["ID101", "ID102", "..."]
    }

    results_mia_synth = load_shadow_model_results(SM_FOLDER)
    results_mia_anon = load_phantom_anon_results(PA_FOLDER)

    results_mia_clean = filter_and_merge_results(results_mia_anon, results_mia_synth)

    results_summary = summarize_mia_results(results_mia_clean)
    results_summary.to_csv(f"MIA_Statistics.xlsx")
