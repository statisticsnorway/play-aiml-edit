class Config:
    # Mapper og filstier
    project_name = "gs://ssb-vare-tjen-korttid-data-delt-vhi-aiml-prod/vhi-data"
    mappe_utvalg = "utvalgsdata/utvalg_p"
    aiml4os_mappe = "" # mappe hvor man lagrer data

    # Variabler
    years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
    bedrift = "orgnrb"
    omsetning = "oms"
    dato = "periode"
    error_col = "har_feil"
    acc_errors = ["cumulative_sum", "cascading", "random"]

    # Startmåned for akkumuleringsfeil
    start_month = 1
    start_month_prob = 0.95

    # Modell parametere
    min_periods = 1
    split_date = "2024-01-01"