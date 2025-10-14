import dapla as dp
import pandas as pd

def load_data_year(path, mappe, year):
    months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

    if year == 2025:
        months = ["01", "02", "03", "04"]
    
    files = [
        dp.read_pandas(
            gcs_path=f"{path}/{mappe}{year}-{m}_v1.parquet",
            file_format="parquet",
            columns=None
            ) for m in months
    ]

    return pd.concat(files, ignore_index=True)