import pandas as pd
import numpy as np

def create_features(df, bedrift="orgnrb", omsetning="oms", dato="periode", min_periods=1, months=[3, 6, 9, 12]):
    df[dato] = pd.to_datetime(df[dato])

    # Tidsvariabler (måned, år)
    df["month"] = df[dato].dt.month
    df["year"] = df[dato].dt.year

    for company in df[bedrift].unique():
        mask = df[bedrift] == company
        turnover = df.loc[mask, omsetning]
        
        # Differanse mellom forrige måned
        df.loc[mask, "mom_change"] = turnover.diff()
        df.loc[mask, "mom_pct_change"] = turnover.pct_change(fill_method=None)
        
        # Årsforskjeller (mellom tidligere år)
        df.loc[mask, "turnover_last_year"] = turnover.shift(12)
        df.loc[mask, "yoy_change"] = turnover.diff(12)
        df.loc[mask, "yoy_pct_change"] = turnover.pct_change(12, fill_method=None)
        df.loc[mask, "ratio_to_last_year"] = turnover / (df.loc[mask, "turnover_last_year"] + 1e-6)
        
        # 3. Flytende/glidende gjenomsnitt
        for window in months:
            df.loc[mask, f"rolling_mean_{window}m"] = turnover.rolling(window, min_periods=min_periods).mean()
            df.loc[mask, f"rolling_median_{window}m"] = turnover.rolling(window, min_periods=min_periods).median()
            df.loc[mask, f"rolling_std_{window}m"] = turnover.rolling(window, min_periods=min_periods).std()
            # df.loc[mask, f"rolling_min_{window}m"] = turnover.rolling(window, min_periods=min_periods).min()
            # df.loc[mask, f"rolling_max_{window}m"] = turnover.rolling(window, min_periods=min_periods).max()
        
        # Forhold mellom omsetning og flytende gjennomsnitt
        for window in months:
            df.loc[mask, f"ratio_to_{window}m_mean"] = turnover / (df.loc[mask, f"rolling_mean_{window}m"] + 1e-6)
        
        # z-score
        df.loc[mask, "zscore_3m"] = (turnover - df.loc[mask, "rolling_mean_3m"]) / (df.loc[mask, "rolling_std_3m"] + 1e-6)
        df.loc[mask, "zscore_6m"] = (turnover - df.loc[mask, "rolling_mean_6m"]) / (df.loc[mask, "rolling_std_6m"] + 1e-6)
        
        # lag-features
        for lag in [1, 2, 3, 6, 12]:
            df.loc[mask, f"turnover_lag_{lag}"] = turnover.shift(lag)
        
        # Grad av variasjon (variasjonskoeffisient, std / mean)
        df.loc[mask, "cv_3m"] = df.loc[mask, "rolling_std_3m"] / (df.loc[mask, "rolling_mean_3m"] + 1e-6)
        df.loc[mask, "cv_12m"] = df.loc[mask, "rolling_std_12m"] / (df.loc[mask, "rolling_mean_12m"] + 1e-6) 
    
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