import dapla as dp
import pandas as pd

project_name = "gs://ssb-vare-tjen-korttid-data-delt-vhi-aiml-prod/vhi-data"
mappe_kjede = "kjededata/kjede_p"
mappe_transak = "transaksjonsdata/oms_transaksjon_bedrift_mmva_p"
mappe_utvalg = "utvalgsdata/utvalg_p"

def load_data_year_month(path, mappe, year, month):
    file = dp.read_pandas(
        gcs_path=f"{path}/{mappe}{year}-{month}_v1.parquet",
        file_format="parquet",
        columns=None
        )
    file["year"] = int(year)
    file["month"] = int(month)
    return file

def load_data_year(path, mappe, year):
    months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

    files = [
        dp.read_pandas(
            gcs_path=f"{path}/{mappe}{year}-{m}_v1.parquet",
            file_format="parquet",
            columns=None
            ) for m in months
    ]

    return pd.concat(files, ignore_index=True)

file = load_data_year(path=project_name, mappe=mappe_utvalg, year=2021)

print(file)