# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#   kernelspec:
#     display_name: play-aiml-edit
#     language: python
#     name: play-aiml-edit
# ---

# ## Testing the HB method and the accumulation error method from 
# ## SSB Vaskify to identify accumulation errors.

# #### The methods were applied to two-month periods (i.e., January–February, February–March, etc.) for the years 2018–2023 in order to determine the optimal parameter values for each method.
# #### These parameter values were then used to detect accumulation errors in data from January 2024 to April 2025 using both the HB method and the accumulation error method.  
# #### Four NACE groups at the 2-digit level were considered: 00, 45, 46, and 47. For NACE 45 and 46, data are only available for 2021–2025, while for NACE 00 and 47, data are available for 2018–2025.  
# #### The NACE group used in the two methods was defined based on the NACE classification in the second month of each two-month period.  
# #### In particular, many units are classified under NACE 00 in January and February; from March onwards, these are largely reassigned to the other three NACE groups.
#

# + [markdown] jp-MarkdownHeadingCollapsed=true
# #### HB METHOD:
# #### The HB method was applied stratified by NACE × two-month period.  
# #### Optimal parameter values were selected based on testing three different values of U and C.  
# #### The metric Mean_f1 was used to determine the best parameter combination:  
#
# U_VALUES = [0.2, 0.5, 0.9]  
# C_VALUES = [4.0, 7.0, 10.0]  
# A_VALUES = [0.05]
#
# -

# #### ACCUMULATION ERROR METHOD:
# #### The accumulation error method was also applied to two-month periods within each NACE group (defined by the NACE classification in the second month of the period).  
# #### The metric Mean_f1 was used to determine the best parameter value:  
#
# ERROR_VALUES = [0.5, 0.8]

# ## Dependencies
#
# The following custom modules are required:
# - base_model.py
# - create_accumulation_error.py
# - load_data.py
# - time_features.py
# - config.py
#
# These are available in the accompanying repository.
#
# The HB and accumulation error methods rely on the vaskify package, which is available at: https://github.com/statisticsnorway/ssb-vaskify

# +
from pathlib import Path
import sys

repo_root = Path.cwd()

while not (repo_root / "src").exists():
    repo_root = repo_root.parent

sys.path.append(str(repo_root))

# -

import lightgbm

import time
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# +
from pathlib import Path
import sys

from src.models.base_model import BaseModel

from src.data.create_synthetic_data import make_synthetic_df
from src.accumulation_error.create_accumulation_error import AccumulationErrors
from src.data.load_data import get_all_data
from src.data.time_features import create_features, prepare_data
from config.config import Config  # pyright: ignore[reportAttributeAccessIssue]

from vaskify import Detect
# -

import pandas as pd
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score

from IPython.display import HTML

import warnings 
warnings.filterwarnings("ignore")

# ## Input Data

# Due to data confidentiality, the original dataset cannot be shared. However, the methods can be applied to any dataset with a similar structure. The required input is a panel dataset containing:

# - **unit_id** (*string*): Unique identifier for each unit (e.g., firm or entity)
# - **period** (*datetime*): Main time variable used in the analysis (e.g., 2018-01-01)
# - **time_period** (*string or datetime*): String representation of `period`, used by the detection methods
# - **value** (*float*): Variable of interest (e.g., turnover)
# - **nace** (*string*): Industry classification code
# - **nace2** (*string*): Industry classification at the 2-digit level
# - **true_error** (*binary, optional*): Indicator of true errors (1 = error, 0 = non-error), used for evaluation
# - **year_month** (*string*): Time variable formatted as year-month

# ### Data Requirements

# - The data must be in long format (one row per unit per time period)
# - `value` must be numeric
# - `true_error` is optional for detection, but required for evaluation


# #### Construction of Two-Month Datasets

# The data are split into overlapping two-month periods (e.g., January-February, February-March).
# Each period is identified by a key of the form "YYYYMM_YYYYMM".

# For each pair of consecutive months, a subset of the data is created and stored in a dictionary:
# - key: period identifier (e.g., "201801_201802")
# - value: corresponding DataFrame


# #### Grouping of Periods

# Two-month periods are grouped across years based on their position within the calendar year 
# (e.g., all January-February periods in group 1, all February-March periods in group 2 etc.). This allows parameter performance to be
# evaluated across comparable seasonal patterns.


# #### Data Structures

# The following data structures are used throughout the analysis:

#  - **group_keys** (*dict*): Maps each group to corresponding dataset keys (e.g., "201801_201802")
#  - **new_dfs** (*dict*): Maps each group index to a list of DataFrames representing the same
#    two-month period across different years
#  - **selected_keys** (*list*): List of dataset keys included in the evaluation period (e.g., 2024-2025)
#  - **filtered_dfs** (*dict*): Subset of `new_dfs` restricted to the evaluation period
#


# #### Generating synthetic data:

# df_synthetic = make_synthetic_df(n_orgs=150, years=range(2018, 2026), seed=1)

# print("Generating accumulation errors")
# make_errors = AccumulationErrors(
#    cfg=Config,
#    years=Config.years,
#    type_of_errors=Config.acc_errors,
#    total_error_prct= 0.30,
#        seed=Config.seed
#    )      
# df_org = make_errors.create_accumulation_errors(df_synthetic)

# df_errors = df_org.copy()

# Henter data og lager akkumuleringsfeil. Legger alt i df
if __name__ == "__main__":
    time_start = time.time()

    print("Henter ut data fra VHI")
    hent_data = get_all_data(Config)
    
    print("Lager akkumuleringsfeil")
    make_errors = AccumulationErrors(
        cfg=Config,
        years=Config.years,
        type_of_errors=Config.acc_errors,
        total_error_prct=Config.bedrifter_med_feil,
        seed=Config.seed
    )      
    df_org = make_errors.create_accumulation_errors(hent_data)

#Lager kopi av df. Kaller den df_errors
df_errors = df_org.copy()


df_errors["periode_streng"] = df_errors["periode"].dt.strftime("%Y-%m-%d")

# ### Set the NACE level, create new variables, and split the data into two-month dataframes:

# +

df_errors["orgnrb"] = df_errors["orgnrb"].astype(str)
# %%
df_errors['aar'] = df_errors['periode'].dt.strftime('%Y')
df_errors['maned'] = df_errors['periode'].dt.strftime('%m')
df_errors['aar_maaned'] = df_errors['periode'].dt.strftime('%Y%m')
df_errors["nace2"] = df_errors["nace"].str.slice(0, 2)

df_subset = df_errors[["orgnrb","periode","periode_streng","aar_maaned","oms","nace","nace2","har_feil"]]
df_renamed = df_subset.rename(columns={
    "orgnrb": "unit_id",
    "periode": "period",
    "periode_streng" : "time_period",
    "aar_maaned" : "year_month",
    "oms" : "value",
    "nace" : "nace",
    "nace2" : "nace2",
    "har_feil" : "true_error"})
# %%
df_errors = df_renamed.copy()
mnd = sorted(df_errors['year_month'].unique())

# %%
mnd_par = [(mnd[i], mnd[i+1]) for i in range(len(mnd)-1)]

splitte_dfs = {}
for start, slutt in mnd_par:
    subset = df_errors[df_errors['year_month'].isin([start, slutt])]
    splitte_dfs[f"{start}_{slutt}"] = subset
# -

# ### Groups the two-month dataframes by month pair and selects the datasets used for parameter tuning.
# ### These groups are later used to evaluate parameter performance and identify the best-performing parameter values.
# #### Group 1 contains all January-February dataframes, Group 2 contains all February-March dataframes, and so on.

# +

group_keys = {
    1:  ["201801_201802", "201901_201902", "202001_202002", "202101_202102", "202201_202202", "202301_202302"],
    2:  ["201802_201803", "201902_201903", "202002_202003", "202102_202103", "202202_202203", "202302_202303"],
    3:  ["201803_201804", "201903_201904", "202003_202004", "202103_202104", "202203_202204", "202303_202304"],
    4:  ["201804_201805", "201904_201905", "202004_202005", "202104_202105", "202204_202205", "202304_202305"],
    5:  ["201805_201806", "201905_201906", "202005_202006", "202105_202106", "202205_202206", "202305_202306"],
    6:  ["201806_201807", "201906_201907", "202006_202007", "202106_202107", "202206_202207", "202306_202307"],
    7:  ["201807_201808", "201907_201908", "202007_202008", "202107_202108", "202207_202208", "202307_202308"],
    8:  ["201808_201809", "201908_201909", "202008_202009", "202108_202109", "202208_202209", "202308_202309"],
    9:  ["201809_201810", "201909_201910", "202009_202010", "202109_202110", "202209_202210", "202309_202310"],
    10: ["201810_201811", "201910_201911", "202010_202011", "202110_202111", "202210_202211", "202310_202311"],
    11: ["201811_201812", "201911_201912", "202011_202012", "202111_202112", "202211_202212", "202311_202312"],
   
}



new_dfs = {i: [] for i in range(1, 12)}


for gid in range(1, 12):
    for key in group_keys[gid]:
        df = splitte_dfs[key].copy()
        new_dfs[gid].append(df)

# -

#  ### Creation of functions for running the HB method, computing metrics, and applying the accumulation error method:

# +

def run_hb(df, pu, pc, pa, group_var):
    """
    Applies the Hidiroglou–Berthelot (HB) method for outlier detection
    within groups of longitudinal data.

    Parameters
        ----------
        df : pandas.DataFrame
            Input dataset containing at least:
            - 'unit_id' (identifier)
            - 'time_period' (time variable, exactly two periods per group)
            - 'value' (variable of interest)
            - grouping variable (e.g. industry)

        pu, pc, pa : float
            Parameters for the detection method.

        group_var : str
            Column used to define groups (e.g. industry classification).

        Returns
        -------
        pandas.DataFrame or None
            Concatenated results across groups, or None if no valid results.

    """
    results = []
    df = df.copy()
    
    for group in df[group_var].unique():
    
        subset = df[df[group_var] == group].copy()
    
        # Require exactly two time periods
        periods = subset["period"].unique()
        if len(periods) != 2:
            continue
        enddate = pd.to_datetime(subset["time_period"]).max()
        
        # Run detection (assumes Detect class is available)
        detector = Detect(subset, "unit_id")
    
        res = detector.hb(
            y_var="value",
            time_var="time_period",
            pu=pu,
            pc=pc,
            pa=pa,
            strata_var=group_var
        ).copy()
    
    # Basic validation
        if res is None or "ratio" not in res.columns:
            continue

        # Identify time columns and rename to generic format
        time_cols = sorted([c for c in res.columns if c.startswith("20")])
        if len(time_cols) != 2:
            continue

        res = res.rename(columns={
            time_cols[0]: "x1",
            time_cols[1]: "x2"
        })

        # Minimal derived variables
        res["max_value"] = res[["x1", "x2"]].max(axis=1)
        res["median_ratio"] = res["ratio"].median()
        res["period"] = enddate
        results.append(res)

    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return None


def run_accumulation_error(df, error_threshold):
    """
    Applies an accumulation error detection method to longitudinal data.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset containing at least:
        - 'unit_id' (identifier)
        - 'time_period' (time variable)
        - 'value' (variable of interest)
        - error_threshold : float (Threshold parameter controlling the sensitivity of the method.)

    Returns
    -------
    pandas.DataFrame or None
        DataFrame with unit identifiers, time periods, and binary error flags,
        or None if no valid results are produced.
    """

    if df is None or df.empty:
        return None

    detector = Detect(df, "unit_id")

    res = detector.accumulation_error(
        y_var="value",
        time_var="time_period",
        output_format="data",
        error=error_threshold,
    )

    if res is None or res.empty:
        return None

    return (
        res[["unit_id", "period", "flag_accumulation", "true_error"]]
        # Create binary version of flag (NaN -> 0) for evaluation
        .assign(flag_accumulation_clean=lambda x: x["flag_accumulation"].fillna(0).astype(int))
    )



def metrics(y_true, y_pred, beta=2.0):
    """
        Computes standard classification metrics.

         Returns
        -------
        dict
            Dictionary containing precision, recall, F1, and F-beta scores.
    """

    result = {
        "f1_score": f1_score(y_true, y_pred, zero_division=0.0), 
        "f2_score": fbeta_score(y_true, y_pred, beta=2.0, zero_division=0.0),  
        f"f{beta}_score": fbeta_score(y_true, y_pred, beta=beta, zero_division=0.0), 
        "precision": precision_score(y_true, y_pred, zero_division=0.0),  
        "recall": recall_score(y_true, y_pred, zero_division=0.0),  
   }

    return result



# +
#metrics_dfs = {}
metrics_hb = {}
metrics_accumulation = {}

#global_df_combined = pd.DataFrame()
global_df_accumulation = pd.DataFrame()
global_df_hb = pd.DataFrame()

df_metrics_hb_final = pd.DataFrame()
df_metrics_accumulation_final = pd.DataFrame()
# -

# ### Specification of Parameter Values for Model Training
# #### The cells below, until the section titled "Computation of overall performance metrics for the HB method and the accumulation error method", need to be executed separately for each nace2_t2 category (45, 46, 47, and 00).

# +
# =================================
# Configuration of model parameters
# =================================

# Grouping variable
group_var = "nace2_t2"

# Optional subset selection (e.g. specific industry)
selected_group = "45"

# Parameter grids
U_VALUES = [0.2, 0.5, 0.9]
C_VALUES = [4.0, 7.0, 10.0]
A_VALUES = [0.05]

ERROR_VALUES = [0.5, 0.8]
# -

# #### Application of the accumulation error method and computation of performance metrics:
#

# +
accumulation_results = []

for error_value in ERROR_VALUES:
    for group_id, dataframes in new_dfs.items():
        for idx, df_group in enumerate(dataframes):

            dataset_key = group_keys[group_id][idx]

            # Identify second-period industry classification (t2)
            end_period = pd.to_datetime(df_group["period"]).max()

            nace2_t2 = (
                df_group[pd.to_datetime(df_group["period"]) == end_period]
                [["unit_id", "nace2"]]
                .rename(columns={"nace2": "nace2_t2"})
            )

            df_group = df_group.merge(nace2_t2, on="unit_id", how="left")
            
            # Filter on selected industry
            df_group = df_group[df_group["nace2_t2"] == selected_group]

            if df_group.empty:
                continue

            # Run accumulation error detection
            try:
                df_accu = run_accumulation_error(df_group, error_threshold=error_value)

            except Exception:
                continue
          
            if df_accu is None or df_accu.empty:
                continue

            # Add metadata
            df_accu = df_accu.assign(
                group_id=group_id,
                dataset_key=dataset_key,
                industry=selected_group,
                error_threshold=error_value,
            )

            accumulation_results.append(df_accu)

# Combine all results once at the end
df_accumulation_all = (
    pd.concat(accumulation_results, ignore_index=True)
    if accumulation_results else None
)


# +
accumulation_metrics = []

if df_accumulation_all is not None:
    for (group_id, dataset_key, error_threshold), group_df in df_accumulation_all.groupby(
        ["group_id", "dataset_key", "error_threshold"]
    ):
        group_df = group_df.copy()

        m = metrics(
            group_df["true_error"],
            group_df["flag_accumulation_clean"],
            beta=2.0
        )

        accumulation_metrics.append({
            "industry": group_df["industry"].iloc[0],
            "group_id": group_id,
            "dataset_key": dataset_key,
            "U": np.nan,
            "C": np.nan,
            "A": np.nan,
            "precision_accumulation": m["precision"],
            "recall_accumulation": m["recall"],
            "f_beta_accumulation": m["f2.0_score"],
            "f2_accumulation": m["f2_score"],
            "f1_accumulation": m["f1_score"],
            "error_threshold": error_threshold,
            "n_true_errors": int((group_df["true_error"] == 1).sum()),
            "n_flagged": int((group_df["flag_accumulation_clean"] == 1).sum()),
        })
df_metrics_accumulation = pd.DataFrame(accumulation_metrics)



# -

# #### Identification of optimal parameter values for each group (e.g., January–February, etc.):

# +
best_accumulation_per_group = (
    df_metrics_accumulation
    .groupby(["group_id", "industry", "error_threshold"])
    .agg(
        mean_precision=("precision_accumulation", "mean"),
        mean_recall=("recall_accumulation", "mean"),
        mean_f_beta=("f_beta_accumulation", "mean"),
        mean_f1=("f1_accumulation", "mean"),
        n_datasets=("dataset_key", "nunique"),
    )
    .reset_index()
)

# Select best error threshold per group 
best_accumulation_per_group = (
    best_accumulation_per_group
    .sort_values("mean_f1", ascending=False)
    .groupby("group_id", as_index=False)
    .first()
)

# -

# #### Application of the HB method and computation of performance metrics:

# +
hb_results = []

for U in U_VALUES:
    for C in C_VALUES:
        for A in A_VALUES:
    
            for group_id, dataframes in new_dfs.items():
                for idx, df_group in enumerate(dataframes):
    
                    dataset_key = group_keys[group_id][idx]
                    df_group = df_group.copy()
                    df_group["period"] = pd.to_datetime(df_group["period"])
                    end_period = df_group["period"].max()
    
                    # Identify industry in period t2
                    nace2_t2 = (
                        df_group[df_group["period"] == end_period]
                        [["unit_id", "nace2"]]
                        .rename(columns={"nace2": "nace2_t2"})
                    )
    
                    df_group = df_group.merge(nace2_t2, on="unit_id", how="left")
    
                    # Filter to selected industry (defined by t2 classification)
                    df_group = df_group[df_group["nace2_t2"] == selected_group]
                    if df_group.empty:
                        continue
                      
                        
                     # HB is applied within a single industry; group_var is kept for consistency
                    df_hb = run_hb(df_group, pu=U, pc=C, pa=A, group_var = "nace2_t2")
                    if df_hb is None:
                        continue
                       
                    merged = df_hb.merge(
                        df_group,
                        on=["unit_id", "period"],
                        how="left"
                    )
                     
                    # Adjust flags
                    merged["flag_hb_adjusted"] = 0
                    merged.loc[merged["ratio"] >= 1, "flag_hb_adjusted"] = merged["flag_hb"]
    
                    # Add metadata
                    merged = merged.assign(
                        group_id=group_id,
                        dataset_key=dataset_key,
                        industry=selected_group,
                        U=U,
                        C=C,
                        A=A,
                    )
                        
                    hb_results.append(merged)

df_hb_all = pd.concat(hb_results, ignore_index=True) if hb_results else None

# +
hb_metrics = []

if df_hb_all is not None:

    for (group_id, dataset_key, U, C, A), group_df in df_hb_all.groupby(
        ["group_id", "dataset_key", "U", "C", "A"]
    ):
        group_df = group_df.copy()
    
        m = metrics(
            group_df["true_error"],
            group_df["flag_hb_adjusted"],
            beta=2.0
        )
    
        hb_metrics.append({
            "industry": group_df["industry"].iat[0],
            "group_id": group_id,
            "dataset_key": dataset_key,
            "U": U,
            "C": C,
            "A": A,
            "precision_hb": m["precision"],
            "recall_hb": m["recall"],
            "f_beta_hb": m["f2.0_score"],
            "f1_hb": m["f1_score"],
            "error_threshold": np.nan,
            "n_true_errors": int((group_df["true_error"] == 1).sum()),
            "n_flagged": int((group_df["flag_hb_adjusted"] == 1).sum()),
        })

df_metrics_hb = pd.DataFrame(hb_metrics)
# -

# #### Identification of optimal parameter values for each group (e.g., January–February, etc.):

# +
best_hb_per_group = (
    df_metrics_hb
    .groupby(["group_id", "industry", "U", "C", "A"])
    .agg(
        mean_precision=("precision_hb", "mean"),
        mean_recall=("recall_hb", "mean"),
        mean_f_beta=("f_beta_hb", "mean"),
        std_f_beta=("f_beta_hb", "std"),
        min_f_beta=("f_beta_hb", "min"),
        mean_f1=("f1_hb", "mean"),
        n_datasets=("dataset_key", "nunique"),
    )
    .reset_index()
)

# Select best parameter combination per group
best_hb_per_group = (
    best_hb_per_group
    .sort_values("mean_f1", ascending=False)
    .groupby("group_id", as_index=False)
    .first()
)


# -

# # Application of the optimal parameter values to the evaluation period:
# ## i.e., from January 2024 to April 2025.

# #### Definition of a function to map period keys to group numbers:

def key_to_group_index(dataset_key: str) -> int:
    """
    Maps a two-period dataset key to a sequential group index.

    Examples
    --------
    202401_202402 -> 1  (Jan–Feb)
    202402_202403 -> 2  (Feb–Mar)
    202403_202404 -> 3  (Mar–Apr)

    Parameters
    ----------
    dataset_key : str
        String representing two consecutive periods in the format YYYYMM_YYYYMM.

    Returns
    -------
    int
        Sequential group index based on the second period.
    """

    _, end_period = dataset_key.split("_")

    end_month = int(end_period[-2:])  # Extract month (e.g. "202402" -> 2)

    # January–February corresponds to index 1
    group_index = end_month - 1

    return group_index

# ### Selection and Filtering of Two-Month Periods for the Evaluation Dataset
#
#

# A subset of two-month periods is selected to define the evaluation dataset. These periods correspond to the time span from January 2024 to April 2025.
#
# A list of selected period keys is defined manually, where each key represents a pair of consecutive months (e.g., "202401_202402").
#
# The dictionary `split_dfs`, which contains all available two-month subsets, is then filtered to include only the selected periods. The resulting dictionary, `filtered_dfs`, contains only the DataFrames relevant for the evaluation phase.

# #### Application of the HB method across all groups:
# #### For each group, the optimal parameter values are identified and the HB method is applied accordingly.


# ## Applies the best-performing configuration to the new dataframes:
# ### Covering the period from January 2024 to April 2025.

selected_keys = [
    "202401_202402",
    "202402_202403",
    "202403_202404",
    "202404_202405",
    "202405_202406",
    "202406_202407",
    "202407_202408",
    "202408_202409",
    "202409_202410",
    "202410_202411",
    "202411_202412",
    "202501_202502",
    "202502_202503",
    "202503_202504",
]


filtered_dfs = {
    key: splitte_dfs[key]
    for key in selected_keys
    if key in splitte_dfs
}

# +
hb_method_metrics = []

y_true_all = []
y_pred_all = []

for dataset_key, df_group in filtered_dfs.items():

    if dataset_key not in selected_keys:
        continue

    group_index = key_to_group_index(dataset_key)

    df_group = df_group.copy()
   
    #    print(df_group[df_group["true_error"] ==1])
    # ======================
    # Preprocessing
    # ======================
    df_group["period"] = pd.to_datetime(df_group["period"])

    df_group["row_id"] = (
        df_group["unit_id"].astype(str) + "_" +
        df_group["time_period"].astype(str)
    )

    start_period = df_group["period"].min()
    end_period = df_group["period"].max()

    # Identify industry in second period (t2)
    nace2_t2 = (
        df_group.loc[
            df_group["period"] == end_period,
            ["unit_id", "nace2"]
        ]
        .rename(columns={"nace2": "nace2_t2"})
    )

    df_group = df_group.merge(nace2_t2, on="unit_id", how="left")

    # Filter on selected industry
    df_group = df_group[df_group["nace2_t2"] == selected_group]
    if df_group.empty:
        continue

    n_obs_t1 = len(df_group[df_group["period"] == start_period])
    n_obs_t2 = len(df_group[df_group["period"] == end_period])

    # ======================
    # Apply HB
    # ======================
    row = best_hb_per_group.loc[
        best_hb_per_group["group_id"] == group_index
    ]

    if len(row) != 1:
        raise ValueError(f"Invalid HB parameters for group {group_index}")

    row = row.iloc[0]

    df_flags = run_hb(
        df_group,
        pu=row["U"],
        pc=row["C"],
        pa=row["A"],
        group_var = "nace2_t2",
    )

    if df_flags is None or df_flags.empty:
        continue
    
    merged = df_flags.merge(
        df_group,
        on=["unit_id", "period"],
        how="left"
    )

    # Adjust flag definition
    merged["flag_pred"] = 0
    merged.loc[merged["ratio"] >= 1, "flag_pred"] = merged["flag_hb"]

    # ======================
    # Metrics per dataset
    # ======================
    m = metrics(
        merged["true_error"],
        merged["flag_pred"],
        beta=2.0
    )

    y_true_all.extend(merged["true_error"].astype(int))
    y_pred_all.extend(merged["flag_pred"].astype(int))

    hb_method_metrics.append({
        "dataset_key": dataset_key,
        "industry": selected_group,
        "method": "hb",
        "precision": m["precision"],
        "recall": m["recall"],
        "f1": m["f1_score"],
        "n_true_errors": int((merged["true_error"] == 1).sum()),
        "n_flagged": int((merged["flag_pred"] == 1).sum()),
        "n_obs_t1": n_obs_t1,
        "n_obs_t2": n_obs_t2,
        "n_obs_final": len(merged),
    })

tmp_metrics = pd.DataFrame(hb_method_metrics)

# ======================
# Combine predictions
# ======================
tmp_df = pd.DataFrame({
    "y_true": y_true_all,
    "y_pred": y_pred_all,
    "industry": selected_group,
    "method": "hb",
})

global_df_hb = pd.concat([global_df_hb, tmp_df], ignore_index=True)

# ======================
# Metrics dataframe
# ======================
df_metrics_hb_final = pd.concat([df_metrics_hb_final, tmp_metrics], ignore_index=True
)


# +
accumulation_method_metrics = []

y_true_all = []
y_pred_all = []

for dataset_key, df_group in filtered_dfs.items():

    if dataset_key not in selected_keys:
        continue

    group_index = key_to_group_index(dataset_key)

    df_group = df_group.copy()

    # ======================
    # Preprocessing
    # ======================
    df_group["period"] = pd.to_datetime(df_group["period"])
    end_period = df_group["period"].max()

    # Identify industry in second period (t2)
    nace2_t2 = (
        df_group.loc[
            df_group["period"] == end_period,
            ["unit_id", "nace2"]
        ]
        .rename(columns={"nace2": "nace2_t2"})
    )

    df_group = df_group.merge(nace2_t2, on="unit_id", how="left")

    # Filter on selected industry
    df_group = df_group[df_group["nace2_t2"] == selected_group]
    if df_group.empty:
        continue

    # ======================
    # Apply accumulation method
    # ======================
    row = best_accumulation_per_group.loc[
        best_accumulation_per_group["group_id"] == group_index
    ]

    if len(row) != 1:
        raise ValueError(f"Invalid accumulation parameters for group {group_index}")

    row = row.iloc[0]

    merged = run_accumulation_error(
        df_group,
        error_threshold=row["error_threshold"],
    )

    if merged is None or merged.empty:
        continue

    # Keep only second period
    merged = merged[merged["period"] == end_period]

    merged["flag_pred"] = merged["flag_accumulation_clean"]

    # ======================
    # Metrics per dataset
    # ======================
    m = metrics(
        merged["true_error"],
        merged["flag_pred"],
        beta=2.0
    )

    y_true_all.extend(merged["true_error"].astype(int))
    y_pred_all.extend(merged["flag_pred"].astype(int))

    accumulation_method_metrics.append({
        "dataset_key": dataset_key,
        "industry": selected_group,
        "method": "accumulation",
        "precision": m["precision"],
        "recall": m["recall"],
        "f1": m["f1_score"],
        "n_true_errors": int((merged["true_error"] == 1).sum()),
        "n_flagged": int((merged["flag_pred"] == 1).sum()),
        "n_obs_final": len(merged),
    })
tmp_metrics = pd.DataFrame(accumulation_method_metrics)
# ======================
# Combine predictions
# ======================
tmp_df = pd.DataFrame({
    "y_true": y_true_all,
    "y_pred": y_pred_all,
    "industry": selected_group,
    "method": "accumulation",
})

global_df_accumulation = pd.concat([global_df_accumulation, tmp_df], ignore_index=True)

# ======================
# Metrics dataframe
# ======================
df_metrics_accumulation_final = pd.concat([df_metrics_accumulation_final, tmp_metrics], ignore_index = True)
# -

# ## Computation of overall performance metrics for the HB method and the accumulation error method:

# +

if not global_df_hb.empty:
    global_metrics_hb = metrics(
        global_df_hb["y_true"],
        global_df_hb["y_pred"]
    )

if not global_df_accumulation.empty:
    global_metrics_accumulation = metrics(
        global_df_accumulation["y_true"],
        global_df_accumulation["y_pred"]
    )

# Convert to DataFrame for reporting
df_global_metrics_hb = pd.DataFrame([global_metrics_hb]) if not global_df_hb.empty else pd.DataFrame()
df_global_metrics_accumulation = pd.DataFrame([global_metrics_accumulation]) if not global_df_accumulation.empty else pd.DataFrame()

# -

df_global_metrics_hb

df_global_metrics_accumulation




