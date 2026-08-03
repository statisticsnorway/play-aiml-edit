from config.config import Config

from src.data import get_all_data, create_features
from src.models import (
    AccumulationLGBM,
    AccumulationIsoFor
)
from src.accumulation_error import AccumulationErrors

# TODO: option for specific method or run all methods?

if __name__ == "__main__":
    print("Get data_")
    hent_data = get_all_data(cfg=Config)
    # TODO: if not existing, use synthetic data instead.
    
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

    lgbm_detector = AccumulationLGBM(df=df.copy(), cfg=Config)

    print("Training LightGBM:")
    lgbm_detector.evaluate()

    print("Training Isolation Forest")
    prep = AccumulationIsoFor(df=df.copy(), cfg=Config)

    contamination = [0.01, 0.03, 0.05, "train"]

    for cont in contamination:
        metrics = prep.evaluate(contamination=cont)

        print(f"Isolation forest with contamination {cont}: {metrics}")