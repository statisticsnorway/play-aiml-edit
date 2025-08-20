# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: play-aiml-edit
#     language: python
#     name: play-aiml-edit
# ---

# %%
import pandas as pd

# %%
import numpy as np

# %%
import os

# %% [markdown]
# ### Ser på en fil under området utvalgsdata:

# %%
file_path = "/buckets/shared/vare-tjen-korttid/vhi-aiml/vhi-data/utvalgsdata/utvalg_p2024-07_v1.parquet"

# %%
df = pd.read_parquet(file_path)

# %%
df

# %% [markdown]
# ### Ser på en fil under kjededata:

# %%
file_path = "/buckets/shared/vare-tjen-korttid/vhi-aiml/vhi-data/kjededata/kjede_p2024-07_v1.parquet"

# %%
df2 = pd.read_parquet(file_path)

# %%
df2

# %% [markdown]
# ### Ser op en fil under transaksjonsdata:

# %%
file_path = "/buckets/shared/vare-tjen-korttid/vhi-aiml/vhi-data/transaksjonsdata/oms_transaksjon_bedrift_mmva_p2024-07_v1.parquet"

# %%
df3 = pd.read_parquet(file_path)

# %%
df3

# %% [markdown]
# ### Ser på hvilke områder som ligger under området vhi-data:

# %%
file_path = "/buckets/shared/vare-tjen-korttid/vhi-aiml/vhi-data/"

# %%
files = os.listdir(file_path)

# %%
files

# %%
for file in files:
    print(file)

# %% [markdown]
# ### Ser på hvilke filer som ligger under området utvalgsdata:

# %%
file_path = "/buckets/shared/vare-tjen-korttid/vhi-aiml/vhi-data/utvalgsdata"

# %%
files = os.listdir(file_path)

# %%
files

# %%
for file in files:
    print(file)

# %%
