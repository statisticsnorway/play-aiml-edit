import pandas as pd

from config.config import Config
from src.HB_and_accumulation_models.hb_accumulation_common import (
    compute_metrics,
    prepare_long_format,
    build_two_month_datasets,
    group_keys_by_period,
    with_nace2_t2,
    run_hb,
    run_accumulation_error,
)

DEFAULT_INDUSTRIES = ["00", "45", "46", "47"]
DEFAULT_U_VALUES = [0.2, 0.5, 0.9]
DEFAULT_C_VALUES = [4.0, 7.0, 10.0]
DEFAULT_A_VALUES = [0.05]
DEFAULT_ERROR_VALUES = [0.5, 0.8]


def _tune_hb(two_month_dfs, group_keys, industries, u_values, c_values, a_values):
    hb_results = []

    for selected_group in industries:
        for U in u_values:
            for C in c_values:
                for A in a_values:
                    for group_id, keys in group_keys.items():
                        for key in keys:
                            df_group = with_nace2_t2(two_month_dfs[key], selected_group=selected_group)
                            if df_group.empty:
                                continue

                            df_hb = run_hb(df_group, pu=U, pc=C, pa=A, group_var="nace2_t2")
                            if df_hb is None:
                                continue

                            merged = df_hb.merge(df_group, on=["unit_id", "period"], how="left")
                            merged["flag_hb_adjusted"] = 0
                            merged.loc[merged["ratio"] >= 1, "flag_hb_adjusted"] = merged["flag_hb"]

                            merged = merged.assign(
                                group_id=group_id, dataset_key=key, industry=selected_group, U=U, C=C, A=A,
                            )
                            hb_results.append(merged)

    if not hb_results:
        return pd.DataFrame()

    df_hb_all = pd.concat(hb_results, ignore_index=True)

    hb_metrics = []
    for (group_id, dataset_key, industry, U, C, A), group_df in df_hb_all.groupby(
        ["group_id", "dataset_key", "industry", "U", "C", "A"]
    ):
        m = compute_metrics(group_df["true_error"], group_df["flag_hb_adjusted"], beta=2.0)
        hb_metrics.append({
            "industry": industry, "group_id": group_id, "dataset_key": dataset_key,
            "U": U, "C": C, "A": A,
            "precision_hb": m["precision"], "recall_hb": m["recall"],
            "f_beta_hb": m["f2.0_score"], "f1_hb": m["f1_score"],
        })

    df_metrics_hb = pd.DataFrame(hb_metrics)

    best_hb_per_group = (
        df_metrics_hb
        .groupby(["group_id", "industry", "U", "C", "A"])
        .agg(
            mean_precision=("precision_hb", "mean"),
            mean_recall=("recall_hb", "mean"),
            mean_f_beta=("f_beta_hb", "mean"),
            mean_f1=("f1_hb", "mean"),
            n_datasets=("dataset_key", "nunique"),
        )
        .reset_index()
        .sort_values("mean_f1", ascending=False)
        .groupby(["group_id", "industry"], as_index=False)
        .first()
    )
    return best_hb_per_group


def _tune_accumulation(two_month_dfs, group_keys, industries, error_values):
    accumulation_results = []

    for selected_group in industries:
        for error_value in error_values:
            for group_id, keys in group_keys.items():
                for key in keys:
                    df_group = with_nace2_t2(two_month_dfs[key], selected_group=selected_group)
                    if df_group.empty:
                        continue

                    try:
                        df_accu = run_accumulation_error(df_group, error_threshold=error_value)
                    except Exception:
                        continue

                    if df_accu is None or df_accu.empty:
                        continue

                    df_accu = df_accu.assign(
                        group_id=group_id, dataset_key=key, industry=selected_group, error_threshold=error_value,
                    )
                    accumulation_results.append(df_accu)

    if not accumulation_results:
        return pd.DataFrame()

    df_accumulation_all = pd.concat(accumulation_results, ignore_index=True)

    accumulation_metrics = []
    for (group_id, dataset_key, industry, error_threshold), group_df in df_accumulation_all.groupby(
        ["group_id", "dataset_key", "industry", "error_threshold"]
    ):
        m = compute_metrics(group_df["true_error"], group_df["flag_accumulation_clean"], beta=2.0)
        accumulation_metrics.append({
            "industry": industry, "group_id": group_id, "dataset_key": dataset_key,
            "error_threshold": error_threshold,
            "precision_accumulation": m["precision"], "recall_accumulation": m["recall"],
            "f_beta_accumulation": m["f2.0_score"], "f1_accumulation": m["f1_score"],
        })

    df_metrics_accumulation = pd.DataFrame(accumulation_metrics)

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
        .sort_values("mean_f1", ascending=False)
        .groupby(["group_id", "industry"], as_index=False)
        .first()
    )
    return best_accumulation_per_group


def find_optimal_parameters(
    df,
    cfg=Config,
    industries=DEFAULT_INDUSTRIES,
    u_values=DEFAULT_U_VALUES,
    c_values=DEFAULT_C_VALUES,
    a_values=DEFAULT_A_VALUES,
    error_values=DEFAULT_ERROR_VALUES,
):
    long_df = prepare_long_format(df, cfg)
    two_month_dfs = build_two_month_datasets(long_df)

    train_group_keys = group_keys_by_period(two_month_dfs.keys(), end_date=cfg.split_date)

    best_hb = _tune_hb(two_month_dfs, train_group_keys, industries, u_values, c_values, a_values)
    best_accumulation = _tune_accumulation(two_month_dfs, train_group_keys, industries, error_values)

    return {"hb": best_hb, "accumulation": best_accumulation}