import pandas as pd
from sklearn.metrics import (
    f1_score,
    fbeta_score,
    precision_score,
    recall_score
)

from vaskify import Detect


def compute_metrics(y_true, y_pred, beta=2.0):
    return {
        "f1_score": f1_score(y_true, y_pred, zero_division=0.0),
        "f2_score": fbeta_score(y_true, y_pred, beta=2.0, zero_division=0.0),
        f"f{beta}_score": fbeta_score(y_true, y_pred, beta=beta, zero_division=0.0),
        "precision": precision_score(y_true, y_pred, zero_division=0.0),
        "recall": recall_score(y_true, y_pred, zero_division=0.0),
    }


def prepare_long_format(df, cfg):
    df = df.copy()
    df[cfg.bedrift] = df[cfg.bedrift].astype(str)
    df[cfg.dato] = pd.to_datetime(df[cfg.dato])

    df["time_period"] = df[cfg.dato].dt.strftime("%Y-%m-%d")
    df["year_month"] = df[cfg.dato].dt.strftime("%Y%m")
    df["nace2"] = df["nace"].str.slice(0, 2)

    renamed = df[
        [cfg.bedrift, cfg.dato, "time_period", "year_month", cfg.omsetning, "nace", "nace2", cfg.error_col]
    ].rename(columns={
        cfg.bedrift: "unit_id",
        cfg.dato: "period",
        cfg.omsetning: "value",
        cfg.error_col: "true_error",
    })
    return renamed


def build_two_month_datasets(df):
    months = sorted(df["year_month"].unique())
    month_pairs = [(months[i], months[i + 1]) for i in range(len(months) - 1)]

    two_month_dfs = {}
    for start, end in month_pairs:
        subset = df[df["year_month"].isin([start, end])]
        two_month_dfs[f"{start}_{end}"] = subset

    return two_month_dfs


def key_to_group_index(dataset_key):
    _, end_period = dataset_key.split("_")
    end_month = int(end_period[-2:])
    return end_month - 1


def group_keys_by_period(two_month_keys, start_date=None, end_date=None):
    start = pd.to_datetime(start_date) if start_date is not None else None
    end = pd.to_datetime(end_date) if end_date is not None else None

    groups = {}
    for key in sorted(two_month_keys):
        end_period = pd.to_datetime(key.split("_")[1], format="%Y%m")
        if start is not None and end_period < start:
            continue
        if end is not None and end_period >= end:
            continue

        group_id = key_to_group_index(key)
        groups.setdefault(group_id, []).append(key)

    return groups


def filter_keys_by_period(two_month_keys, start_date=None, end_date=None):
    start = pd.to_datetime(start_date) if start_date is not None else None
    end = pd.to_datetime(end_date) if end_date is not None else None

    selected = []
    for key in sorted(two_month_keys):
        end_period = pd.to_datetime(key.split("_")[1], format="%Y%m")
        if start is not None and end_period < start:
            continue
        if end is not None and end_period >= end:
            continue
        selected.append(key)

    return selected


def with_nace2_t2(df_group, selected_group=None):
    df_group = df_group.copy()
    df_group["period"] = pd.to_datetime(df_group["period"])
    end_period = df_group["period"].max()

    nace2_t2 = (
        df_group.loc[df_group["period"] == end_period, ["unit_id", "nace2"]]
        .rename(columns={"nace2": "nace2_t2"})
    )
    df_group = df_group.merge(nace2_t2, on="unit_id", how="left")

    if selected_group is not None:
        df_group = df_group[df_group["nace2_t2"] == selected_group]

    return df_group


def run_hb(df, pu, pc, pa, group_var):
    results = []
    df = df.copy()

    for group in df[group_var].unique():
        subset = df[df[group_var] == group].copy()

        periods = subset["period"].unique()
        if len(periods) != 2:
            continue
        enddate = pd.to_datetime(subset["time_period"]).max()

        detector = Detect(subset, "unit_id")

        res = detector.hb(
            y_var="value",
            time_var="time_period",
            pu=pu,
            pc=pc,
            pa=pa,
            strata_var=group_var,
        ).copy()

        if res is None or "ratio" not in res.columns:
            continue

        time_cols = sorted([c for c in res.columns if c.startswith("20")])
        if len(time_cols) != 2:
            continue

        res = res.rename(columns={time_cols[0]: "x1", time_cols[1]: "x2"})

        res["max_value"] = res[["x1", "x2"]].max(axis=1)
        res["median_ratio"] = res["ratio"].median()
        res["period"] = enddate
        results.append(res)

    if results:
        return pd.concat(results, ignore_index=True)
    return None


def run_accumulation_error(df, error_threshold):
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
        .assign(flag_accumulation_clean=lambda x: x["flag_accumulation"].fillna(0).astype(int))
    )
