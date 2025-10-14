import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score

from time_features import create_features, prepare_data
from config import Config  # pyright: ignore[reportAttributeAccessIssue]

class AccumulationLGBM:
    
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
    
    def fit_predict(
        self, learning_rate=0.05, num_leaves=31, max_depth=-1, min_child_samples=20,
        n_estimators=100, scale_pos_weight=None, feature_cols=None, verbose=-1
    ):

        X_train, X_valid, y_train, y_valid, feature_cols = prepare_data(
            df=df,
            cfg=self.cfg,
            feature_cols=feature_cols
        )
        
        if scale_pos_weight is None:
            n_negative = (y_train == 0).sum()
            n_positive = (y_train == 1).sum()
            scale_pos_weight = n_negative / (n_positive + 1e-6)
            print(f"Auto-calculated scale_pos_weight: {scale_pos_weight:.2f}")
        
        print(f"Training set: {len(X_train)} samples, {y_train.sum()} errors ({y_train.mean():.3%})")
        print(f"Validation set: {len(X_valid)} samples, {y_valid.sum()} errors ({y_valid.mean():.3%})")
        
        self.model = lgb.LGBMClassifier(
            objective='binary',
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            n_estimators=n_estimators,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbose=verbose,
            n_jobs=-1,
            colsample_bytree=0.8,
            subsample=0.8,
            subsample_freq=1,
            reg_alpha=0.1,
            reg_lambda=0.1
        )

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )

        
        y_pred = self.model.predict(X_valid)
        
        self.feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            "y_true": y_valid.values,  # pyright: ignore[reportAttributeAccessIssue]
            "y_pred": y_pred,
            "X_valid": X_valid,
            "feature_importance": self.feature_importance
        }
    
    def metrics(self, y_true, y_pred, beta=2.0):
        """Calculate evaluation metrics."""
        metrics = {
            "f1_score": f1_score(y_true, y_pred, zero_division=0.0),  # pyright: ignore[reportArgumentType]
            "f2_score": fbeta_score(y_true, y_pred, beta=2.0, zero_division=0.0),  # pyright: ignore[reportArgumentType]
            f"f{beta}_score": fbeta_score(y_true, y_pred, beta=beta, zero_division=0.0),  # pyright: ignore[reportArgumentType]
            "precision": precision_score(y_true, y_pred, zero_division=0.0),  # pyright: ignore[reportArgumentType]
            "recall": recall_score(y_true, y_pred, zero_division=0.0),  # pyright: ignore[reportArgumentType]
        }
        
        return metrics
    
    def evaluate(self, beta=0.5, show_feature_importance=True, top_n=15, **kwargs):
        """Fit, predict, and evaluate."""
        results = self.fit_predict(**kwargs)
        
        metrics = self.metrics(
            y_true=results["y_true"], 
            y_pred=results["y_pred"],
            beta=beta
        )
        
        pred_rate = np.sum(results["y_pred"]) / len(results["y_pred"])
        true_rate = np.sum(results["y_true"]) / len(results["y_true"])
        print(f"\nPredicted anomaly rate: {pred_rate:.3f}, True rate: {true_rate:.3f}")
        
        if show_feature_importance and self.feature_importance is not None:
            print(f"\nTop {top_n} Most Important Features:")
            print(self.feature_importance.head(n=top_n).to_string(index=False))
        
        return metrics, results


if __name__ == "__main__":
    df = pd.read_csv("experimental/Ruben/src/accumulation_error.csv")
    
    print("Lager tidsvariabler:")
    df = create_features(
        df=df,
        company_col=Config.company_col,
        turnover_col=Config.turnover_col,
        date_col=Config.date_col,
    )
    
    lgbm_detector = AccumulationLGBM(
        df=df,
        cfg=Config,
        date_col="periode", 
        company_col="orgnrb", 
        turnover_col="oms", 
        error_col="contains_error"
    )

    print("Trener modell:")
    metrics, results = lgbm_detector.evaluate(
        beta=0.5,
        n_estimators=500,
        learning_rate=0.01,
        num_leaves=31,
        max_depth=7,
        top_n=36
    )

    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")