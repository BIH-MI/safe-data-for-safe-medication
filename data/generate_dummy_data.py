import uuid
from random import randrange
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def random_date(start_date, end_date):
    delta = (end_date - start_date) / np.timedelta64(1, 'D')
    random_days = randrange(int(delta))
    return start_date + np.timedelta64(random_days, "D")


def create_dataset(n_records: int) -> pd.DataFrame:
    """
        Creation of a dummy dataset.

        Notice that we use 4 derived attributes that are not part of the original dataset.
        These should be generated in preprocessing after the protected datasets were created.
    """
    # Dataset attributes
    df_dummy = pd.DataFrame()
    df_dummy["PID"] = [uuid.uuid4() for _ in range(n_records)]
    df_dummy["age"] = np.random.normal(75, 11, n_records)
    birth_years = (2020 - df_dummy["age"]).astype(int)
    df_dummy["birth_qtr"] = [np.datetime64(f"{year}-01-01") for year in birth_years]
    df_dummy["sex_flg"] = np.random.choice([0, 1], size=(n_records,), p=[0.52, 0.48])
    df_dummy["oac_grp"] = np.random.choice(["DOAC", "VKA"], size=(n_records,), p=[0.65, 0.35])
    df_dummy["index_med"] = np.random.choice(["Apixaban", "Dabigatran", "Edoxaban", "Phenprocoumon", "Rivaroxaban"], size=(n_records,), p=[0.25, 0.05, 0.05, 0.30, 0.35])
    df_dummy["baseline_strt_dt"] = pd.to_datetime([random_date(np.datetime64('2011-01-01'), np.datetime64('2019-01-01')) for _ in range(n_records)]).date
    df_dummy["death_max_dt"] = df_dummy["baseline_strt_dt"] + pd.to_timedelta(np.random.normal(365, 200, n_records))
    df_dummy["vte_index_dt"] = df_dummy["baseline_strt_dt"]
    df_dummy["baseline_end_dt"] = df_dummy["baseline_strt_dt"] + pd.to_timedelta(np.random.normal(365, 200, n_records))
    df_dummy["fu_end_dt"] = df_dummy["baseline_strt_dt"] + pd.to_timedelta(np.random.normal(365, 200, n_records))
    df_dummy["bleeding_fu_end_dt"] = df_dummy["baseline_strt_dt"] + pd.to_timedelta(np.random.normal(365, 200, n_records))
    df_dummy["death_fu_end_dt"] = df_dummy["death_max_dt"]
    df_dummy["fu_bleeding_flg"] = np.random.choice([0, 1], size=(n_records,), p=[0.8, 0.2])
    df_dummy["fu_death_flg"] = np.random.choice([0, 1], size=(n_records,), p=[0.8, 0.2])
    df_dummy["fu_rec_vte_flg"] = np.random.choice([0, 1], size=(n_records,), p=[0.8, 0.2])
    df_dummy["covar_CKD (at least 2 codes)_flg"] = np.random.choice([0, 1], size=(n_records,), p=[0.7, 0.3])
    df_dummy["covar_Obesity _flg"] = np.random.choice([0, 1], size=(n_records,), p=[0.9, 0.1])
    df_dummy["covar_Varicose veins/post-thrombotic syndrome_flg"] = np.random.choice([0, 1], size=(n_records,), p=[0.8, 0.2])
    df_dummy["covar_Arterial hypertension_flg"] = np.random.choice([0, 1], size=(n_records,), p=[0.2, 0.8])
    df_dummy["covar_Congestive heart failure_flg"] = np.random.choice([0, 1], size=(n_records,), p=[0.6, 0.4])
    df_dummy["covar_MI_flg"] = np.random.choice([0, 1], size=(n_records,), p=[0.8, 0.2])
    df_dummy["covar_Stroke_flg"] = np.random.choice([0, 1], size=(n_records,), p=[0.8, 0.2])
    df_dummy["covar_Diabetes mellitus_flg"] = np.random.choice([0, 1], size=(n_records,), p=[0.6, 0.4])
    df_dummy["covar_Moderate/severe liver disease_flg"] = np.random.choice([0, 1], size=(n_records,), p=[0.9, 0.1])
    df_dummy["covar_IBD_flg"] = np.random.choice([0, 1], size=(n_records,), p=[0.9, 0.1])
    df_dummy["covar_Cancer (excl. non-melanoma skin cancer)_flg"] = np.random.choice([0, 1], size=(n_records,), p=[0.8, 0.2])
    df_dummy["covar_Bleeding_flg"] = np.random.choice([0, 1], size=(n_records,), p=[0.8, 0.2])
    df_dummy["covar_Fracture_flg"] = np.random.choice([0, 1], size=(n_records,), p=[0.8, 0.2])
    df_dummy["covar_Major surgery_flg"] = np.random.choice([0, 1], size=(n_records,), p=[0.9, 0.1])
    df_dummy["covar_Oral contraceptives_flg"] = np.random.choice([0, 1], size=(n_records,), p=[0.9, 0.1])
    df_dummy["covar_Hormone replacement therapy_flg"] = np.random.choice([0, 1], size=(n_records,), p=[1, 0])
    df_dummy["covar_Tamoxifen_flg"] = np.random.choice([0, 1], size=(n_records,), p=[0.8, 0.2])
    df_dummy["covar_Systemic corticosteroids_flg"] = np.random.choice([0, 1], size=(n_records,), p=[0.8, 0.2])
    df_dummy["covar_SSRIs_flg"] = np.random.choice([0, 1], size=(n_records,), p=[0.8, 0.2])
    df_dummy["covar_PPIs_flg"] = np.random.choice([0, 1], size=(n_records,), p=[0.8, 0.2])
    df_dummy["covar_NSAIDs_flg"] = np.random.choice([0, 1], size=(n_records,), p=[0.6, 0.4])
    df_dummy["covar_Overall number of hospitalizations"] = np.random.choice([0, 1], size=(n_records,), p=[0.6, 0.4])
    df_dummy["covar_Overall number of non-antithrombotic medication (All drugs excep B01A)"] = np.random.choice([0, 1], size=(n_records,), p=[0.3, 0.7])

    # Derived attributes
    df_dummy["Geschlecht"] = df_dummy["sex_flg"]
    df_dummy["bleeding_fu_length"] = np.random.normal(75, 75, n_records)
    df_dummy["death_fu_length"] = np.random.normal(75, 75, n_records)
    df_dummy["oac_flg"] = df_dummy["oac_grp"] .map({"DOAC": 1, "VKA": 0})
    return df_dummy


if __name__ == "__main__":
    """
        Generation of dummy data.
        
        The dummy_data_anonymized and dummy_data_synth are random dummy data, 
        with no relation to dummy_data_original. For your actual analysis,
        anonymize your data with the respective tools. 
    """
    n_records = 1800

    df_dummy_full = create_dataset(n_records)
    df_dummy_full.to_csv("dummy_data_original.csv", index=None)

    df_train, df_test = train_test_split(df_dummy_full, test_size=0.2)
    df_train.to_csv("dummy_data_train.csv", index=None)
    df_test.to_csv("dummy_data_test.csv", index=None)

    create_dataset(len(df_train)).to_csv("dummy_data_synth.csv", index=None)
    create_dataset(len(df_train)).to_csv("dummy_data_anonymized.csv", index=None)

