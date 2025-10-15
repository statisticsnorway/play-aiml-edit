import time
from sklearn.ensemble import IsolationForest

from base_model import BaseModel
from create_accumulation_error import AccumulationErrors
from load_data import get_all_data
from time_features import create_features, prepare_data
from config import Config  # pyright: ignore[reportAttributeAccessIssue]

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
            df=df,
            cfg=self.cfg,
        )

        self._statistics(
            X_train, y_train, X_valid, y_valid
        )

        if contamination is None or contamination == "train":
            contamination = self.train_error_rate
        
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,  # pyright: ignore[reportArgumentType]
            random_state=random_state,
        )
        self.model.fit(X=X_train)
        
        predictions = self.model.predict(X=X_valid)
        scores = self.model.score_samples(X=X_valid)
        y_pred = (predictions == -1).astype(int)

        return {
            "y_true": y_valid.values,  # pyright: ignore[reportAttributeAccessIssue]
            "y_pred": y_pred,
            "anomaly_scores": scores,
            "X_valid": X_valid
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

if __name__ == "__main__":
    time_start = time.time()

    print("Henter ut data fra VHI")
    hent_data = get_all_data(Config)
    
    print("Lager akkumuleringsfeil")
    make_errors = AccumulationErrors(
        cfg=Config,
        years=Config.years,
        type_of_errors=Config.acc_errors,
        total_error_prct=0.25 # må se nærmere på denne variabelen, introduserer rundt 1-5% med total_error_prct=0.05-0.30 
    )
    
    df = make_errors.create_accumulation_errors(hent_data)
    
    print("Lager tidsvariabler")
    df = create_features(
        df=df,
        bedrift=Config.bedrift,
        omsetning=Config.omsetning,
        dato=Config.dato,
        min_periods=Config.min_periods
    )

    print("Trener Isolation Forest")
    prep = AccumulationIsoFor(df, Config)
    
    contamination = 0.01
    print(f"Isolation forest med contamination: {contamination}")
    metrics = prep.evaluate(contamination=contamination)
    print(metrics)

    contamination = 0.03
    print(f"Isolation forest med contamination: {contamination}")
    metrics = prep.evaluate(contamination=contamination)
    print(metrics)

    contamination = 0.05
    print(f"Isolation forest med contamination: {contamination}")
    metrics = prep.evaluate(contamination=contamination)
    print(metrics)

    contamination = "train"
    print(f"Isolation forest med contamination: {contamination}")
    metrics = prep.evaluate(contamination=contamination)  # pyright: ignore[reportArgumentType]
    print(metrics)

    time_end = time.time()
    print(f"Tid: {(time_end - time_start) / 60} minutter")