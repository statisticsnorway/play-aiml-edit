from dynaconf import Dynaconf


class Config:
    # Mapper og filstier
    project_name = "gs://ssb-vare-tjen-korttid-data-delt-vhi-aiml-prod/vhi-data"
    mappe_utvalg = "utvalgsdata/utvalg_p"
    aiml4os_mappe = ""

    # Variabler
    years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
    bedrift = "orgnrb"
    omsetning = "oms"
    dato = "periode"
    error_col = "har_feil"
    acc_errors = ["cumulative_sum"] #["cumulative_sum", "cascading", "random"]
    bedrifter_med_feil = 0.175
    seed = 42

    # Startmåned for akkumuleringsfeil
    start_month = 1  # 0 = tilfeldig og 1 = Januar
    start_month_prob = 0.95
    max_duration = 12
    min_duration = 3

    # Mulighet for feil i flere år
    multiple_error_years = False # hvis False, kun mulighet for feil i 1 år
    error_year_probability = 0.4 # Sannsynligheten for feil i et år, ignorert hvis multiple_errors_years er False

    # Modell parametere
    min_periods = 1
    split_date = "2023-01-01"   # train < split_date <= validation
    test_date = "2024-01-01"    # validation < test_date <= test

    # Training parameters
    eval_metric = "f1_score" # precision, recall, f1_score, f_beta, f2_score, loss (what losses are possible?)
    greater_is_better = True

settings = Dynaconf(
    settings_files=["settings.toml"],
    envvar_prefix="DYNACONF",
    environments=True,
    env="default",  # Change this to gsbuckets for use with google storage buckets
)
