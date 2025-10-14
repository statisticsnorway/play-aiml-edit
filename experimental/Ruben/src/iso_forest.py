import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score, accuracy_score

from time_features import create_features, prepare_data
from config import Config  # pyright: ignore[reportAttributeAccessIssue]

class AccumulationIsoFor:
    """
    Simple data preparation for detecting accumulation errors with Isolation Forest.
    Focuses on the most important features only.
    """
    
    def __init__(
        self, df, cfg, date_col="date", company_col="company_name",
        turnover_col="turnover", error_col="contains_error"
    ):
        self.df = df.copy()
        self.cfg = cfg
        self.date_col = date_col
        self.company_col = company_col
        self.turnover_col = turnover_col
        self.error_col = error_col

        self.df[date_col] = pd.to_datetime(self.df[date_col])
        self.df = self.df.sort_values([company_col, date_col]).reset_index(drop=True)
    
    def fit_predict(self, feature_cols=None, contamination=0.02, random_state=42):
        X_train, X_valid, y_train, y_valid, feature_cols = prepare_data(
            df=df,
            cfg=self.cfg,
            feature_cols=feature_cols
        )
        
        self.model = IsolationForest(
            contamination=contamination,  # pyright: ignore[reportArgumentType]
            random_state=random_state,
            n_estimators=100
        )
        self.model.fit(X=X_train)
        
        predictions = self.model.predict(X=X_valid)
        scores = self.model.score_samples(X=X_valid)
        
        y_pred = (predictions == -1).astype(int)

        print(f"Training set: {len(X_train)} samples, {y_train.sum()} errors ({y_train.mean():.3%})")
        print(f"Validation set: {len(X_valid)} samples, {y_valid.sum()} errors ({y_valid.mean():.3%})")
        return {
            "y_true": y_valid.values,  # pyright: ignore[reportAttributeAccessIssue]
            "y_pred": y_pred,
            "anomaly_scores": scores,
            "X_valid": X_valid
        }
    
    def metrics(self, y_true, y_pred, beta=2.0):
        metrics = {
            "f1_score": f1_score(y_true, y_pred, zero_division=0.0),  # pyright: ignore[reportArgumentType]
            "f2_score": fbeta_score(y_true, y_pred, beta=2.0, zero_division=0.0),  # pyright: ignore[reportArgumentType]
            f"f{beta}_score": fbeta_score(y_true, y_pred, beta=beta, zero_division=0.0),  # pyright: ignore[reportArgumentType]
            "precision": precision_score(y_true, y_pred, zero_division=0.0),  # pyright: ignore[reportArgumentType]
            "recall": recall_score(y_true, y_pred, zero_division=0.0),  # pyright: ignore[reportArgumentType]
        }

        return metrics
    
    def evaluate(self, split_date, contamination=0.02, random_state=42, beta=2.0):
        results = self.fit_predict(
            contamination=contamination, 
            random_state=random_state
        )
    
        metrics = self.metrics(
            y_true=results["y_true"], 
            y_pred=results["y_pred"],
            beta=beta
        )

        return metrics


if __name__ == "__main__":
    df = pd.read_csv("experimental/Ruben/src/accumulation_error.csv")
    
    print("Lager tidsvariabler:")
    df = create_features(
        df=df,
        company_col=Config.company_col,
        turnover_col=Config.turnover_col,
        date_col=Config.date_col,
    )
    prep = AccumulationIsoFor(
        df=df,
        cfg=Config,
        date_col=Config.date_col,
        company_col=Config.company_col,
        turnover_col=Config.turnover_col,
        error_col=Config.error_col
    )

    
    contamination = 0.035
    print(contamination)
    split_date = "2024-01-01"
    metrics = prep.evaluate(contamination=contamination, split_date=split_date, beta=0.5)

    print(metrics)