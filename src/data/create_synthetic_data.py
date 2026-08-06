import pandas as pd
import numpy as np

def make_synthetic_df(n_orgs=150, years=range(2018, 2025), seed=1):
    rng = np.random.default_rng(seed)
    records = []
    nace_options = ["45", "46", "47", "00"]
    for org_id in range(1, n_orgs + 1):
        base_level = rng.uniform(50_000, 50_000_000)
        nace = rng.choice(nace_options)
        for year in years:
            for month in range(1, 13):
                date = pd.Timestamp(year=year, month=month, day=1)
                seasonal = 1 + 0.1 * np.sin(2 * np.pi * month / 12)
                noise = rng.normal(1, 0.30)
                value = max(0, base_level * seasonal * noise)
                records.append({
                    'orgnrb': org_id,
                    'periode': date,
                    'oms': round(value, 2),
                    'nace': str(nace)
                })
    return pd.DataFrame(records)