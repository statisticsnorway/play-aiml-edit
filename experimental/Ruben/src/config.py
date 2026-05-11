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
    split_date = "2024-01-01"

    # Training parameters
    eval_metric = "recall" # precision, recall, f1_score, f_beta, f2_score, loss (what losses are possible?)
    greater_is_better = True