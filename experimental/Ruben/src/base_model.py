import pandas as pd
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score

class BaseModel:
    def __init__(self, df, cfg):
        self.df = df.copy()
        self.cfg = cfg
        self.dato = cfg.dato
        self.bedrift = cfg.bedrift
        self.omsetning = cfg.omsetning
        self.error_col = cfg.error_col
        
        self.df[cfg.dato] = pd.to_datetime(self.df[cfg.dato])
        self.df = self.df.sort_values([cfg.bedrift, cfg.dato]).reset_index(drop=True)

        self.feature_importance = None
        self.model = None

    def _metrics(self, y_true, y_pred, beta=2.0):
        metrics = {
            "f1_score": f1_score(y_true, y_pred, zero_division=0.0),  # pyright: ignore[reportArgumentType]
            "f2_score": fbeta_score(y_true, y_pred, beta=2.0, zero_division=0.0),  # pyright: ignore[reportArgumentType]
            f"f{beta}_score": fbeta_score(y_true, y_pred, beta=beta, zero_division=0.0),  # pyright: ignore[reportArgumentType]
            "precision": precision_score(y_true, y_pred, zero_division=0.0),  # pyright: ignore[reportArgumentType]
            "recall": recall_score(y_true, y_pred, zero_division=0.0),  # pyright: ignore[reportArgumentType]
        }

        return metrics
    
    def _statistics(self, X_train, y_train, X_valid, y_valid):
        print(f"Treningssett: {len(X_train)}. Prosent feil: {y_train.mean():.3%}")
        print(f"Valideringssett: {len(X_valid)}. Prosent feil: {y_valid.mean():.3%}")

        self.train_error_rate = y_train.mean()