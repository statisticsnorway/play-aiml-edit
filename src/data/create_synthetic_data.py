import pandas as pd
import numpy as np

def make_synthetic_df(n_orgs=30, years=range(2019, 2023), seed=1):
    rng = np.random.default_rng(seed)
    records = []
    for org_id in range(1, n_orgs + 1):
        base_level = rng.uniform(50_000, 5_000_000)
        for year in years:
            for month in range(1, 13):
                date = pd.Timestamp(year=year, month=month, day=1)
                seasonal = 1 + 0.1 * np.sin(2 * np.pi * month / 12)
                noise = rng.normal(1, 0.05)
                value = max(0, base_level * seasonal * noise)
                records.append({
                    'bedrift_id': org_id,
                    'dato': date,
                    'omsetning': round(value, 2)
                })
    return pd.DataFrame(records)
 
df = make_synthetic_df()
print(df.head())
print(df.shape)
print(df['bedrift_id'].nunique(), "companies")