import pandas as pd
import numpy as np

def postprocess_consecutive(
    df, y_pred, cfg, min_duration, max_duration,
):
    result = y_pred.copy()

    working = df[[cfg.bedrift, cfg.dato]].copy().reset_index(drop=True)
    working["pred"] = result
    working[cfg.dato] = pd.to_datetime(working[cfg.dato])
    working = working.sort_values([cfg.bedrift, cfg.dato])

    for _, group in working.groupby(cfg.bedrift):
        filtered = _filter_runs(
            preds=group["pred"].values,
            min_duration=min_duration,
            max_duration=max_duration,
        )
        working.loc[group.index, "pred"] = filtered

    result = working["pred"].values
    return result


def _filter_runs(
    preds, min_duration, max_duration,
):
    result = preds.copy()
    n = len(preds)
    i = 0

    while i < n:
        if preds[i] == 0:
            i += 1
            continue

        run_start = i
        while i < n and preds[i] == 1:
            i += 1
        run_end = i

        run_length = run_end - run_start

        if run_length < min_duration:
            result[run_start:run_end] = 0
        elif run_length > max_duration:
            result[run_start + max_duration:run_end] = 0

    return result