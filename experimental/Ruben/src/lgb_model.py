import time
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import f1_score
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval


from base_model import BaseModel
from create_accumulation_error import AccumulationErrors
from load_data import get_all_data
from time_features import create_features, prepare_data
from config import Config  # pyright: ignore[reportAttributeAccessIssue]
from post_processing import postprocess_consecutive

def f1_eval_metric(preds, train_data):
    labels = train_data.get_label()
    binary_preds = (preds > 0.5).astype(int)
    f1 = f1_score(y_true=labels, y_pred=binary_preds, average='binary')
    return 'f1_score', f1, True


class AccumulationLGBM(BaseModel):
    """Testing av LightGBM for akkumuleringsfeil."""
    def optimize_hyperparameters(self, max_evals=100, beta=0.5):
        space = {
            'n_estimators': hp.choice('n_estimators', range(500, 1001)),
            'learning_rate': hp.uniform('learning_rate', 0.001, 0.10),
            'max_depth': hp.choice('max_depth', range(5, 9)),
            'num_leaves': hp.choice('num_leaves', range(5, 127)),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
            'reg_alpha': hp.uniform('reg_alpha', 0, 10),
            'reg_lambda' : hp.uniform('reg_lambda', 0, 10),
            'subsample': hp.uniform('subsample', 0.5, 1),
            'subsample_freq': hp.choice('subsample_freq', range(1,10)),
            'min_child_samples': hp.choice('min_data_in_leaf', range(20,100)), 
        }

        def objective(params):
            try:
                results = self.fit_predict(**params)
                
                metrics = self._metrics(
                    y_true=results["y_true"],
                    y_pred=results["y_pred"],
                    beta=beta
                )

                if self.greater_is_better:
                    loss = -(metrics[self.eval_metric])
                else:
                    loss = metrics[self.eval_metric]
                
                return {
                    'loss': loss,
                    'status': STATUS_OK,
                    'metrics': metrics,
                    "params": params,
                }
            except Exception as e:
                print(f"Feil med parametere: {params}")
                print(f"Feilmelding: {e}")
                return {'loss': 1.0, 'status': STATUS_OK}
        
        trials = Trials()
        best_idx = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            verbose=1
        )
        best_params = space_eval(space, hp_assignment=best_idx)
        self.best_params = best_params
        
        return best_params
    
    def fit_predict(
        self, scale_pos_weight=None, **kwargs,
    ):

        X_train, X_valid, y_train, y_valid, feature_cols = prepare_data(
            df=self.df,
            cfg=self.cfg,
        )

        if self.call:
            self._statistics(X_train, y_train, X_valid, y_valid)
            self.call = False
        
        if scale_pos_weight is None:
            n_negative = (y_train == 0).sum()
            n_positive = (y_train == 1).sum()
            scale_pos_weight = n_negative / n_positive

        self.model = lgb.LGBMClassifier(
            objective='binary',
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
            **kwargs
        )

        self.model.fit(
            X=X_train, y=y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric='average_precision',
        )

        y_pred = self.model.predict(X=X_valid)
        
        self.feature_importance = pd.DataFrame(data={
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            "y_true": y_valid.values,
            "y_pred": y_pred,
            "X_valid": X_valid,
            "feature_importance": self.feature_importance
        }
    
    def evaluate(self, beta=0.5, show_feature_importance=True, top_n=15, max_evals=100, **kwargs):

        best_params = self.optimize_hyperparameters(max_evals=max_evals, beta=beta)
        kwargs.update(best_params)
        
        results = self.fit_predict(**kwargs)
        
        metrics = self._metrics(
            y_true=results["y_true"], 
            y_pred=results["y_pred"],
            beta=beta
        )
        
        y_pred_filtered = postprocess_consecutive(
            df=self.df[self.df[self.cfg.dato] >= pd.to_datetime(self.cfg.split_date)].reset_index(drop=True),
            y_pred=results["y_pred"],
            cfg=self.cfg,
            min_duration=self.cfg.min_duration,
            max_duration=self.cfg.max_duration,
        )

        metrics_filtered = self._metrics(
            y_true=results["y_true"],
            y_pred=y_pred_filtered,
            beta=beta,
        )

        
        if show_feature_importance and self.feature_importance is not None:
            print(f"\nTopp {top_n} variabler:")
            print(self.feature_importance.head(n=top_n).to_string(index=False))

        print("Resultater før post-processing")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        print("Resultater etter post-processing:")
        for k, v in metrics_filtered.items():
            print(f"  {k}: {v:.4f}")

        # n_removed = int(results["y_pred"].sum() - y_pred_filtered.sum())
            
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
        total_error_prct=Config.bedrifter_med_feil,
        seed=Config.seed
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
    metrics, results = lgbm_detector.evaluate()

    time_end = time.time()
    print(f"Tid: {(time_end - time_start) / 60} minutter")