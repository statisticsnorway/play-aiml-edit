import time
from sklearn.ensemble import IsolationForest

from src.models.base_model import BaseModel
from src.data.time_features import prepare_data, normalize_by_company

class AccumulationIsoFor(BaseModel):
    """
    Testing av Isolation forest for akkumuleringsfeil.

    Forbedringer
    - normalisering på bedriftsnivå
    - parametertuning
    - se på anomaly scores
    """
    def fit_predict(self, n_estimators=100, contamination=None, random_state=42):
        X_train, X_valid, y_train, y_valid, _ = prepare_data(
            df=self.df,
            cfg=self.cfg,
        )

        self._statistics(X_train, y_train, X_valid, y_valid)
        
        train_with_id = X_train.copy()
        valid_with_id = X_valid.copy()
        train_with_id[self.cfg.bedrift] = self.df.loc[X_train.index, self.cfg.bedrift].values
        valid_with_id[self.cfg.bedrift] = self.df.loc[X_valid.index, self.cfg.bedrift].values

        train_scaled, valid_scaled = normalize_by_company(
            df_train=train_with_id,
            df_valid=valid_with_id,
            bedrift=self.cfg.bedrift,
            omsetning=self.cfg.omsetning,
        )
        X_train = train_scaled.drop(columns=[self.cfg.bedrift])
        X_valid = valid_scaled.drop(columns=[self.cfg.bedrift])

        if contamination is None or contamination == "train":
            contamination = self.train_error_rate

        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )
        self.model.fit(X=X_train)

        predictions = self.model.predict(X=X_valid)
        scores = self.model.score_samples(X=X_valid)
        y_pred = (predictions == -1).astype(int)

        return {
            "y_true": y_valid.values,
            "y_pred": y_pred,
            "anomaly_scores": scores,
            "X_valid": X_valid,
        }
    
    def evaluate(self, contamination=0.02, random_state=42, beta=2.0):
        results = self.fit_predict(
            contamination=contamination, 
            random_state=random_state
        )
    
        metrics = self._metrics(
            y_true=results["y_true"], 
            y_pred=results["y_pred"],
            beta=beta
        )

        return metrics