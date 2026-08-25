import time
from sklearn.ensemble import IsolationForest

from src.models.base_model import BaseModel
from src.data.time_features import prepare_data_with_test, normalize_by_company
from src.models.post_processing import postprocess_consecutive

class AccumulationIsoFor(BaseModel):
    """
    Testing av Isolation forest for akkumuleringsfeil.

    Forbedringer
    - normalisering på bedriftsnivå
    - parametertuning
    - se på anomaly scores
    """
    def fit_predict(self, n_estimators=100, contamination=None, random_state=42, eval_on="valid"):
        data = prepare_data_with_test(df=self.df, cfg=self.cfg)
        X_train, y_train = data["X_train"], data["y_train"]
        X_valid, y_valid = data["X_valid"], data["y_valid"]
        X_test, y_test = data["X_test"], data["y_test"]

        if self.call:
            self._statistics(X_train, y_train, X_valid, y_valid)
            self.call = False

        train_with_id = X_train.copy()
        valid_with_id = X_valid.copy()
        test_with_id = X_test.copy()
        train_with_id[self.cfg.bedrift] = self.df.loc[X_train.index, self.cfg.bedrift].values
        valid_with_id[self.cfg.bedrift] = self.df.loc[X_valid.index, self.cfg.bedrift].values
        test_with_id[self.cfg.bedrift] = self.df.loc[X_test.index, self.cfg.bedrift].values

        train_scaled, valid_scaled = normalize_by_company(
            df_train=train_with_id,
            df_valid=valid_with_id,
            bedrift=self.cfg.bedrift,
            omsetning=self.cfg.omsetning,
        )
        _, test_scaled = normalize_by_company(
            df_train=train_with_id,
            df_valid=test_with_id,
            bedrift=self.cfg.bedrift,
            omsetning=self.cfg.omsetning,
        )

        X_train = train_scaled.drop(columns=[self.cfg.bedrift])
        X_valid = valid_scaled.drop(columns=[self.cfg.bedrift])
        X_test = test_scaled.drop(columns=[self.cfg.bedrift])

        if contamination is None or contamination == "train":
            contamination = self.train_error_rate

        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )
        self.model.fit(X=X_train)

        X_eval = {"train": X_train, "valid": X_valid, "test": X_test}[eval_on]
        y_eval = {"train": y_train, "valid": y_valid, "test": y_test}[eval_on]

        predictions = self.model.predict(X=X_eval)
        scores = self.model.score_samples(X=X_eval)
        y_pred = (predictions == -1).astype(int)

        return {
            "y_true": y_eval.values,
            "y_pred": y_pred,
            "anomaly_scores": scores,
            "X_eval": X_eval,
            "eval_mask": data["masks"][eval_on],
            "eval_on": eval_on,
        }

    def show_results(self, beta, results, eval_on):
        metrics = self._metrics(
            y_true=results["y_true"],
            y_pred=results["y_pred"],
            beta=beta
        )

        df_eval = self.df[results["eval_mask"]].reset_index(drop=True)

        y_pred_filtered = postprocess_consecutive(
            df=df_eval,
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

        # print(f"\nResultater på {eval_on}-settet:")
        # print("Før post-processing:")
        # for k, v in metrics.items():
        #     print(f"  {k}: {v:.4f}")

        # print("Etter post-processing:")
        # for k, v in metrics_filtered.items():
        #     print(f"  {k}: {v:.4f}")

        return metrics, results

    def evaluate(self, n_estimators=100, contamination=0.02, random_state=42, beta=2.0):
        evaluations = ["valid", "test"]

        scores = {}
        for eval_on in evaluations:
            results = self.fit_predict(
                n_estimators=n_estimators,
                contamination=contamination,
                random_state=random_state,
                eval_on=eval_on,
            )

            metrics, results = self.show_results(beta, results, eval_on)

            scores[eval_on] = {
                "results": results,
                "metrics": metrics,
            }

        return scores
