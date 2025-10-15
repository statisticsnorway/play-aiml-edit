import time
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import f1_score

from base_model import BaseModel
from create_accumulation_error import AccumulationErrors
from load_data import get_all_data
from time_features import create_features, prepare_data
from config import Config  # pyright: ignore[reportAttributeAccessIssue]

def f1_eval_metric(preds, train_data):
    labels = train_data.get_label()
    binary_preds = (preds > 0.5).astype(int)
    f1 = f1_score(y_true=labels, y_pred=binary_preds, average='binary')
    return 'f1_score', f1, True

class AccumulationLGBM(BaseModel):
    """
    Testing av LightGBM for akkumuleringsfeil.

    Forbedringer
    - kryssvalidering
    - parametertuning
    - test ut en annen metric
    """
    def fit_predict(
        self, learning_rate=0.05, num_leaves=31, max_depth=-1, min_child_samples=20, 
        n_estimators=100, scale_pos_weight=None, statistics=False
    ):

        X_train, X_valid, y_train, y_valid, feature_cols = prepare_data(
            df=df,
            cfg=self.cfg,
        )

        if statistics:
            self._statistics(
                X_train, y_train, X_valid, y_valid
            )
        
        if scale_pos_weight is None:
            n_negative = (y_train == 0).sum()
            n_positive = (y_train == 1).sum()
            scale_pos_weight = n_negative / n_positive
        
        
        self.model = lgb.LGBMClassifier(
            objective='binary',
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            n_estimators=n_estimators,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
            colsample_bytree=0.8,
            subsample=0.8,
            subsample_freq=1,
            reg_alpha=0.1,
            reg_lambda=0.1
        )

        self.model.fit(
            X=X_train, y=y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )

        y_pred = self.model.predict(X=X_valid)
        
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
    
    def evaluate(self, beta=0.5, show_feature_importance=True, top_n=15, stastics=False, **kwargs):
        results = self.fit_predict(**kwargs)
        
        metrics = self._metrics(
            y_true=results["y_true"], 
            y_pred=results["y_pred"],
            beta=beta
        )
        
        pred_rate = np.sum(results["y_pred"]) / len(results["y_pred"])
        true_rate = np.sum(results["y_true"]) / len(results["y_true"])
        print(f"\nPredikert feil: {pred_rate:.3f}, Faktisk feil: {true_rate:.3f}")
        

        if show_feature_importance and self.feature_importance is not None:
            print(f"\nTopp {top_n} variabler:")
            print(self.feature_importance.head(n=top_n).to_string(index=False))
        
        return metrics, results

if __name__ == "__main__":
    time_start = time.time()

    print("Henter ut data fra VHI")
    hent_data = get_all_data(cfg=Config)
    
    print("Lager akkumuleringsfeil")
    make_errors = AccumulationErrors(
        cfg=Config,
        years=Config.years,
        type_of_errors=Config.acc_errors,
        total_error_prct=0.25 # må se nærmere på denne variabelen, introduserer rundt 1-5% med total_error_prct=0.05-0.30 
    )
    
    df = make_errors.create_accumulation_errors(df=hent_data)
    
    print("Lager tidsvariabler")
    df = create_features(
        df=df,
        bedrift=Config.bedrift,
        omsetning=Config.omsetning,
        dato=Config.dato,
        min_periods=Config.min_periods
    )
    
    lgbm_detector = AccumulationLGBM(df=df, cfg=Config)

    print("Trener LightGBM")
    metrics, results = lgbm_detector.evaluate(
        beta=0.5,
        n_estimators=500,
        learning_rate=0.01,
        num_leaves=31,
        max_depth=7,
        scale_pos_weight=None,
        top_n=10
    )

    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    time_end = time.time()
    print(f"Tid: {(time_end - time_start) / 60} minutter")