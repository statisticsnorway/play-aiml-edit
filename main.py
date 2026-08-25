import warnings
import pandas as pd

warnings.filterwarnings(
    "ignore",
    message="DataFrameGroupBy.apply operated on the grouping columns.*",
    category=FutureWarning,
    module="vaskify",
)
warnings.filterwarnings(
    "ignore",
    category=pd.errors.SettingWithCopyWarning,
    module="vaskify",
)

from config.config import Config

from src.data import (
    get_all_data,
    create_features,
    make_synthetic_df
)
from src.models import (
    AccumulationLGBM,
    AccumulationIsoFor
)
from src.HB_and_accumulation_models import (
    find_optimal_parameters,
    evaluate_hb_and_accumulation,
)
from src.accumulation_error import AccumulationErrors


if __name__ == "__main__":
    print("Get data:")
    try:
        hent_data = get_all_data(cfg=Config)
    except Exception as e:
        print(f"Using synthetic data as data source.")
        hent_data = make_synthetic_df(
            n_orgs=150,
            years=range(min(Config.years), max(Config.years) + 1),
            seed=Config.seed,
        )

    print("Creating accumulation errors:")
    make_errors = AccumulationErrors(
        cfg=Config,
        years=Config.years,
        type_of_errors=Config.acc_errors,
        total_error_prct=Config.bedrifter_med_feil,
        seed=Config.seed
    )   
    
    df = make_errors.create_accumulation_errors(df=hent_data)
    
    print("Making time features:")
    df = create_features(
        df=df,
        bedrift=Config.bedrift,
        omsetning=Config.omsetning,
        dato=Config.dato,
        min_periods=Config.min_periods
    )

    rows = []

    def add_rows(method, scores):
        for split, res in scores.items():
            m = res["metrics"]
            rows.append({
                "method": method,
                "split": split,
                "f1_score": m["f1_score"],
                "precision": m["precision"],
                "recall": m["recall"],
            })

    lgbm_detector = AccumulationLGBM(df=df.copy(), cfg=Config)

    print("Training LightGBM:")
    lgbm_scores = lgbm_detector.evaluate(show_feature_importance=False, max_evals=10)
    add_rows("LightGBM", lgbm_scores)

    print("Training Isolation Forest:")
    prep = AccumulationIsoFor(df=df.copy(), cfg=Config)

    contamination = [0.01, 0.03, 0.05, "train"]

    for cont in contamination:
        iso_scores = prep.evaluate(contamination=cont)
        add_rows(f"IsolationForest (contamination={cont})", iso_scores)

    print("Tuning HB and accumulation-error parameters:")
    best_params = find_optimal_parameters(df, cfg=Config)

    print("Applying HB and accumulation-error methods:")
    hb_acc_scores = evaluate_hb_and_accumulation(
        df,
        best_hb_per_group=best_params["hb"],
        best_accumulation_per_group=best_params["accumulation"],
        cfg=Config,
    )

    for method, splits in hb_acc_scores.items():
        for split, res in splits.items():
            if not res["overall"]:
                continue
            rows.append({
                "method": method,
                "split": split,
                "f1_score": res["overall"]["f1_score"],
                "precision": res["overall"]["precision"],
                "recall": res["overall"]["recall"],
            })

    # Make table results
    results_table = pd.DataFrame(rows).round(4)
    print("\nResultattabell (F1, presisjon, recall):")
    print(results_table.to_string(index=False))