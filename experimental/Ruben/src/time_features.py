import pandas as pd
import numpy as np

def create_features(df, bedrift="orgnrb", omsetning="oms", dato="periode", min_periods=1, months=[3, 6, 9, 12]):
    df[dato] = pd.to_datetime(df[dato])
    df = df.sort_values([bedrift, dato]).reset_index(drop=True)
    
    df["month"] = df[dato].dt.month
    df["year"] = df[dato].dt.year
    
    grouped = df.groupby(bedrift, group_keys=False)[omsetning]
    
    df["mom_change"] = grouped.diff()
    df["mom_pct_change"] = grouped.pct_change(fill_method=None)
    df["turnover_last_year"] = grouped.shift(12)
    df["yoy_change"] = grouped.diff(12)
    df["yoy_pct_change"] = grouped.pct_change(12, fill_method=None)
    df["ratio_to_last_year"] = df[omsetning] / (df["turnover_last_year"] + 1e-6)
    
    for lag in [1, 2, 3, 6, 12]:
        df[f"turnover_lag_{lag}"] = grouped.shift(lag)
    
    for window in months:
        rolling_obj = grouped.rolling(window, min_periods=min_periods)
        
        df[f"rolling_mean_{window}m"] = rolling_obj.mean().reset_index(level=0, drop=True)
        df[f"rolling_median_{window}m"] = rolling_obj.median().reset_index(level=0, drop=True)
        df[f"rolling_std_{window}m"] = rolling_obj.std().reset_index(level=0, drop=True)
        
        df[f"ratio_to_{window}m_mean"] = df[omsetning] / (df[f"rolling_mean_{window}m"] + 1e-6)
    
    df["zscore_3m"] = (df[omsetning] - df["rolling_mean_3m"]) / (df["rolling_std_3m"] + 1e-6)
    df["zscore_6m"] = (df[omsetning] - df["rolling_mean_6m"]) / (df["rolling_std_6m"] + 1e-6)
    df["cv_3m"] = df["rolling_std_3m"] / (df["rolling_mean_3m"] + 1e-6)
    df["cv_12m"] = df["rolling_std_12m"] / (df["rolling_mean_12m"] + 1e-6)
    
    return df

def prepare_data(df, cfg):
    """
    Lager trening- og valideringssett
    """
    feature_cols = [
        cfg.omsetning,
        "mom_change", "mom_pct_change",
        "yoy_change", "yoy_pct_change", "ratio_to_last_year",
        "rolling_mean_3m", "rolling_mean_6m", "rolling_mean_9m", "rolling_mean_12m",
        "rolling_median_3m", "rolling_median_6m", "rolling_median_9m", "rolling_median_12m",
        # "rolling_std_3m", "rolling_std_6m", "rolling_std_9m", "rolling_std_12m",
        # "rolling_min_3m", "rolling_max_3m",
        "ratio_to_3m_mean", "ratio_to_6m_mean", "ratio_to_12m_mean",
        "zscore_3m", "zscore_6m",
        "turnover_lag_1", "turnover_lag_2", "turnover_lag_3", "turnover_lag_6", "turnover_lag_12",
        "cv_3m", "cv_12m", "month", 
    ]
    
    X = df[feature_cols].copy()
    X = X.fillna(0).replace([np.inf, -np.inf], 0)
    Y = df[cfg.error_col]
    dates = df[cfg.dato]
    

    split = pd.to_datetime(cfg.split_date)
    train_mask = dates < split
    valid_mask = dates >= split
    
    X_train = X[train_mask]
    X_valid = X[valid_mask]
    y_train = Y[train_mask]
    y_valid = Y[valid_mask]

    return X_train, X_valid, y_train, y_valid, feature_cols

def normalize_by_company(df_train, df_valid, bedrift, omsetning):
    absolute_features = [
        omsetning,
        "mom_change", "yoy_change",
        "rolling_mean_3m", "rolling_mean_6m", "rolling_mean_9m", "rolling_mean_12m",
        "rolling_median_3m", "rolling_median_6m", "rolling_median_9m", "rolling_median_12m",
        "turnover_lag_1", "turnover_lag_2", "turnover_lag_3", "turnover_lag_6", "turnover_lag_12",
        "turnover_last_year",
    ]
    
    company_scale = (
        df_train
        .groupby(bedrift)[omsetning]
        .median()
        .abs()
        .rename("company_scale")
    )
    global_scale = df_train[omsetning].median()
    
    def _apply_scale(df):
        df = df.copy()
        scale = df[bedrift].map(company_scale).fillna(global_scale).clip(lower=1e-6)
        cols_present = [c for c in absolute_features if c in df.columns]
        df[cols_present] = df[cols_present].div(scale, axis=0)
        return df
    
    return _apply_scale(df_train), _apply_scale(df_valid)