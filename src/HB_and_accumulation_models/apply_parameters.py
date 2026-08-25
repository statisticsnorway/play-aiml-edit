import pandas as pd

from config.config import Config
from src.HB_and_accumulation_models.hb_accumulation_common import (
    compute_metrics,
    prepare_long_format,
    build_two_month_datasets,
    key_to_group_index,
    filter_keys_by_period,
    with_nace2_t2,
    run_hb,
    run_accumulation_error,
)
from src.HB_and_accumulation_models.optimize_parameters import DEFAULT_INDUSTRIES


def _apply_hb(two_month_dfs, keys, best_hb_per_group, industries):
    per_dataset = []
    y_true_all, y_pred_all = [], []

    for key in keys:
        group_id = key_to_group_index(key)

        for selected_group in industries:
            df_group = with_nace2_t2(two_month_dfs[key], selected_group=selected_group)
            if df_group.empty:
                continue

            row = best_hb_per_group.loc[
                (best_hb_per_group["group_id"] == group_id)
                & (best_hb_per_group["industry"] == selected_group)
            ]
            if row.empty:
                continue
            row = row.iloc[0]

            df_flags = run_hb(df_group, pu=row["U"], pc=row["C"], pa=row["A"], group_var="nace2_t2")
            if df_flags is None or df_flags.empty:
                continue

            merged = df_flags.merge(df_group, on=["unit_id", "period"], how="left")
            merged["flag_pred"] = 0
            merged.loc[merged["ratio"] >= 1, "flag_pred"] = merged["flag_hb"]

            m = compute_metrics(merged["true_error"], merged["flag_pred"], beta=2.0)
            y_true_all.extend(merged["true_error"].astype(int))
            y_pred_all.extend(merged["flag_pred"].astype(int))

            per_dataset.append({
                "dataset_key": key, "group_id": group_id, "industry": selected_group, "method": "hb",
                "precision": m["precision"], "recall": m["recall"], "f1": m["f1_score"],
                "n_true_errors": int((merged["true_error"] == 1).sum()),
                "n_flagged": int((merged["flag_pred"] == 1).sum()),
            })

    overall = compute_metrics(y_true_all, y_pred_all) if y_true_all else {}
    return pd.DataFrame(per_dataset), overall


def _apply_accumulation(two_month_dfs, keys, best_accumulation_per_group, industries):
    per_dataset = []
    y_true_all, y_pred_all = [], []

    for key in keys:
        group_id = key_to_group_index(key)

        for selected_group in industries:
            df_group = with_nace2_t2(two_month_dfs[key], selected_group=selected_group)
            if df_group.empty:
                continue

            row = best_accumulation_per_group.loc[
                (best_accumulation_per_group["group_id"] == group_id)
                & (best_accumulation_per_group["industry"] == selected_group)
            ]
            if row.empty:
                continue
            row = row.iloc[0]

            merged = run_accumulation_error(df_group, error_threshold=row["error_threshold"])
            if merged is None or merged.empty:
                continue

            end_period = df_group["period"].max()
            merged = merged[merged["period"] == end_period]
            merged["flag_pred"] = merged["flag_accumulation_clean"]

            m = compute_metrics(merged["true_error"], merged["flag_pred"], beta=2.0)
            y_true_all.extend(merged["true_error"].astype(int))
            y_pred_all.extend(merged["flag_pred"].astype(int))

            per_dataset.append({
                "dataset_key": key, "group_id": group_id, "industry": selected_group, "method": "accumulation",
                "precision": m["precision"], "recall": m["recall"], "f1": m["f1_score"],
                "n_true_errors": int((merged["true_error"] == 1).sum()),
                "n_flagged": int((merged["flag_pred"] == 1).sum()),
            })

    overall = compute_metrics(y_true_all, y_pred_all) if y_true_all else {}
    return pd.DataFrame(per_dataset), overall


def evaluate_hb_and_accumulation(
    df,
    best_hb_per_group,
    best_accumulation_per_group,
    cfg=Config,
    industries=DEFAULT_INDUSTRIES,
):
    long_df = prepare_long_format(df, cfg)
    two_month_dfs = build_two_month_datasets(long_df)

    splits = {
        "valid": filter_keys_by_period(two_month_dfs.keys(), start_date=cfg.split_date, end_date=cfg.test_date),
        "test": filter_keys_by_period(two_month_dfs.keys(), start_date=cfg.test_date),
    }

    results = {"hb": {}, "accumulation": {}}
    for split_name, keys in splits.items():
        if not keys:
            continue

        hb_per_dataset, hb_overall = _apply_hb(two_month_dfs, keys, best_hb_per_group, industries)
        results["hb"][split_name] = {"per_dataset": hb_per_dataset, "overall": hb_overall}

        acc_per_dataset, acc_overall = _apply_accumulation(
            two_month_dfs, keys, best_accumulation_per_group, industries
        )
        results["accumulation"][split_name] = {"per_dataset": acc_per_dataset, "overall": acc_overall}

    return results
