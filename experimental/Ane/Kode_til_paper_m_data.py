# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#   kernelspec:
#     display_name: play-aiml-edit
#     language: python
#     name: play-aiml-edit
# ---

# # Forsøk på å bruke HB-metoden og akkumuleringsfeilmetoden fra 
# # SSB Vaskify til å finne akkumuleringsferil.

# #### Metodene ble kjørt for tomånedersperioder (dvs. januar-februar, februar-mars osv.) i årene 2018-2023 for å finne beste valg av parameterverdier for hver metode. 
# #### Deretter ble disse parameterverdiene brukt for å finne akkumuleringsfeil i data fra januar 2024 til april 2025 ved HB-metoden og akkumuleringsfeilmetoden.
# #### Vi hadde fire NACE-grupper på 2-siffernivå: 00, 45, 46 og 47.For NACE 45 og 46 har vi bare data fra 2021-2025, mens for NACE 00 og 47 har vi data fra 2018-2025.
# #### NACE-gruppe brukt i de to metodene ble definert som NACE-tilhørigheten i måned nummer 2 i tomånedersperioden. 
# #### Det er særlig i januar og februar at det er mange som tilhører NACE 00; fra mars og utover er disse stort sett plassert i de andre tre NACEgruppene.
#
#

# + [markdown] jp-MarkdownHeadingCollapsed=true
# ### HB-METODEN:
# ##### HB-metoden ble kjørt stratifisert for NACE*tomånedersperiode. 
# ##### Beste parametervalg ble valgt utfra testing av tre ulike U-verdier og C-verdier.
# ##### Metrikken Mean_f1 ble brukt til å velge den beste:
# ##### U_VALUES = [0.2, 0.5, 0.9]
# ##### C_VALUES = [4.0, 7.0, 10.0]
# ##### A_VALUES = [0.05]
#
# -

# ### AKKUMULERINGSFEILMETODEN
# ##### Akkumuleringsfeilmetoden ble også kjørt for tomånedersperiodene innen hver NACE (definert ved NACE-tilhørigheten i måned nummer 2). 
# ##### Metrikken Mean_f1 ble brukt til å velge den beste:
# ##### Her testet vi ERROR_VALUES = [0.5, 0.8] 

import time
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# +
from base_model import BaseModel
from create_accumulation_error import AccumulationErrors
from load_data import get_all_data
from time_features import create_features, prepare_data
from config import Config  # pyright: ignore[reportAttributeAccessIssue]

from vaskify import Detect
# -

import pandas as pd
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score

import warnings
warnings.filterwarnings("ignore")


from IPython.display import HTML

# ## Henter data lager akkumuleringsfeil og bearbeider data:

# Henter data og lager akkumuleringsfeil. Legger alt i df
if __name__ == "__main__":
    time_start = time.time()

    print("Henter ut data fra VHI")
    hent_data = get_all_data(Config)
    
    print("Lager akkumuleringsfeil")
    make_errors = AccumulationErrors(
        cfg=Config,
        years=Config.years,
        type_of_errors=Config.acc_errors,
        total_error_prct=Config.bedrifter_med_feil,
        seed=Config.seed
    )      
    df_org = make_errors.create_accumulation_errors(hent_data)

#Lager kopi av df. Kaller den df_errors
df_errors = df_org.copy()
#df_org.info()

# + active=""
# len(df_errors)
# -

# Lager string av periode, til bruk i HB- og akkumuleringsfeil-funksjonen:
df_errors["periode_streng"] = df_errors["periode"].dt.strftime("%Y-%m-%d")

# + active=""
# len(df_errors["periode_streng"].unique())
#
# -

df_errors["periode_streng"].value_counts().reset_index(name="antall")

# +
df_errors["periode"] = pd.to_datetime(df_errors["periode"], errors="coerce")

df_errors["periode"].value_counts().sort_index()


# +
df = df_errors.copy()
df["periode_streng"] = pd.to_datetime(df["periode_streng"], errors="coerce")

df = df[df["periode_streng"] >= "2024-01-01"]

df["periode_streng"].value_counts().sort_index()
# -

df["periode_streng"].value_counts().sum() - 2843

# Lager en variabel, "lag1", som inneholder verdien for denne enheten fra forrige periode:
df_errors["lag1"] = df_errors.groupby("orgnrb")["oms"].shift(1)

# Lager en variabel kalt "forh" som er forholdet mellom oms i t og oms i t-1:
df_errors["forh"] = df_errors["oms"] / df_errors["lag1"]

# Lager en variabel kalt "forhold", som er forholdet mellom oms-verdi i t og editert omsverdi i t:
df_errors["forhold"] = df_errors["oms"]/df_errors["oms_original"]

# ## Velger ut de dataene som skal brukes til å finne beste parametere:

# +
# Legger inn alle aktuelle 2- og 2-måneders perioder i en liste, "aktuelle_keys":

aktuelle_keys = ["201801_201802", "201802_201803", "201803_201804", "201804_201805", "201805_201806",
                 "201806_201807", "201807_201808", "201808_201809", "201809_201810", "201810_201811",
                 "201811_201812", "201812_201901", "201901_201902", "201902_201903", "201903_201904",
                 "201904_201905", "201905_201906", "201906_201907", "201907_201908", "201908_201909",
                 "201909_201910", "201910_201911", "201911_201912", "201912_202001", "202001_202002",
                 "202002_202003", "202003_202004", "202004_202005", "202005_202006", "202006_202007",
                 "202007_202008", "202008_202009", "202009_202010", "202010_202011", "202011_202012",
                 "202012_202101", "202101_202102", "202102_202103", "202103_202104", "202104_202105",
                 "202105_202106", "202106_202107", "202107_202108", "202108_202109", "202109_202110",
                 "202110_202111", "202111_202112", "202112_202201", "202201_202202", "202202_202203",
                 "202203_202204", "202204_202205", "202205_202206", "202206_202207", "202207_202208",
                 "202208_202209", "202209_202210", "202210_202211", "202211_202212", "202212_202301",
                 "202301_202302", "202302_202303", "202303_202304", "202304_202305",
                 "202305_202306", "202306_202307", "202307_202308", "202308_202309",
                 "202309_202310", "202310_202311", "202311_202312"]
# -

# ### Setter NACE-nivå, lager nye variable og deler inn i dataframes med to og to måneder:

# Bestemmer hvilket nacenivå vi skal bruke:
ngruppe = "nace2"

# +
# Lager "periode_streng" og deler opp i 2- og 2-måneders df:
df_errors["periode_streng"] = df_errors["periode"].dt.strftime("%Y-%m-%d")


# %%
df_errors['aar'] = df_errors['periode'].dt.strftime('%Y')
df_errors['maned'] = df_errors['periode'].dt.strftime('%m')
df_errors['aar_maaned'] = df_errors['periode'].dt.strftime('%Y%m')
df_errors["nace2"] = df_errors["nace"].str.slice(0, 2)
# %%
mnd = sorted(df_errors['aar_maaned'].unique())

# %%
mnd_par = [(mnd[i], mnd[i+1]) for i in range(len(mnd)-1)]

splitte_dfs = {}
for start, slutt in mnd_par:
    subset = df_errors[df_errors['aar_maaned'].isin([start, slutt])]
    splitte_dfs[f"{start}_{slutt}"] = subset
# -

df_errors = df_errors.sort_values(["orgnrb", "periode"])

df_errors["nace2_forrige"] = df_errors.groupby("orgnrb")["nace2"].shift(1)
df_errors["periode_forrige"] = df_errors.groupby("orgnrb")["periode"].shift(1)

# +
pd.set_option("display.max_rows", None)
df = df_errors.copy()

df = df[df["periode"] >= "2024-01-01"]

#df.groupby(["periode","nace2_forrige"]).size().reset_index(name="antall")
df_filtrert = df[df["nace2"] == '45']

df_filtrert.groupby(["periode", "nace2"]).size().reset_index(name="antall")
#df_filtrert["antall"].sum()
# -

df_test =df_filtrert.groupby(["periode", "nace2"]).size().reset_index(name="antall")
antall_45 = df_test["antall"].sum()

# ### Grupperer to og to måneders-dataframene etter hvilke måneder det er. 
# #### Gruppe 1 er alle dataframes fra januar-februar, gruppe to fra februar-mars osv:

# +
# 12 grupper med keys (denne fyller du med dine keys)
grupper_keys = {
    1:  ["201801_201802", "201901_201902", "202001_202002", "202101_202102", "202201_202202", "202301_202302"],
    2:  ["201802_201803", "201902_201903", "202002_202003", "202102_202103", "202202_202203", "202302_202303"],
    3:  ["201803_201804", "201903_201904", "202003_202004", "202103_202104", "202203_202204", "202303_202304"],
    4:  ["201804_201805", "201904_201905", "202004_202005", "202104_202105", "202204_202205", "202304_202305"],
    5:  ["201805_201806", "201905_201906", "202005_202006", "202105_202106", "202205_202206", "202305_202306"],
    6:  ["201806_201807", "201906_201907", "202006_202007", "202106_202107", "202206_202207", "202306_202307"],
    7:  ["201807_201808", "201907_201908", "202007_202008", "202107_202108", "202207_202208", "202307_202308"],
    8:  ["201808_201809", "201908_201909", "202008_202009", "202108_202109", "202208_202209", "202308_202309"],
    9:  ["201809_201810", "201909_201910", "202009_202010", "202109_202110", "202209_202210", "202309_202310"],
    10: ["201810_201811", "201910_201911", "202010_202011", "202110_202111", "202210_202211", "202310_202311"],
    11: ["201811_201812", "201911_201912", "202011_202012", "202111_202112", "202211_202212", "202311_202312"],
   
}

# Lag tom struktur med 12 grupper
nye_dfs = {i: [] for i in range(1, 12)}

# Fyll hver gruppe med DataFrames
for gid in range(1, 12):
    for key in grupper_keys[gid]:
        df = splitte_dfs[key].copy()
        nye_dfs[gid].append(df)


# -

# ### Lager funksjoner for kjøring av HB-metoden, plotting, metrikker og kjøring av akkumuleringsmetoden: 

def kjør_hb_for_df(df, U, C, A):
    df_hb = None
    df = df.copy()
    assert "nace2_p2" in df.columns, "nace2_p2 mangler - må lages før kjør_hb_for_df"

    df["nace"] = df["nace"].astype(str)
    df["nace2"] = df["nace"].str.slice(0, 2)
    df["nace3"] = df["nace"].str.slice(0, 3)

    resultater_df = []

    for nacegruppe in df[ngruppe].unique():

        subset = df[df[ngruppe] == nacegruppe].copy()

        # Må ha to perioder
        unike_perioder = subset["periode_streng"].unique()
        if len(unike_perioder) != 2:
            print(f"Hopper over NACE {nacegruppe}: {unike_perioder}")
            continue

        sluttdato = pd.to_datetime(subset["periode_streng"]).max()


        det = Detect(subset, "orgnrb")

        df_hb = det.hb(
            y_var="oms",
            time_var="periode_streng",
            pu=U, pc=C, pa=A,
            strata_var="nace2_p2"
        ).copy()

        
        if df_hb is None or "ratio" not in df_hb.columns:
            print("HB er ikke gyldig for denne DF-en")
            continue
        
        # Finn periodekolonner
        periodekolonner = sorted([c for c in df_hb.columns if c.startswith("20")])
        if len(periodekolonner) != 2:
            print(f"HB returnerte {len(periodekolonner)} perioder for NACE {nacegruppe}")
            continue

        df_hb = df_hb.rename(columns={
            periodekolonner[0]: "x1",
            periodekolonner[1]: "x2"
        })
        
        df_hb["maxX"] = df_hb[["x1","x2"]].max(axis=1)
        df_hb["medrat"] = df_hb["ratio"].median()
        df_hb["colour"] = df_hb["flag_hb"].map(lambda f: "red" if f == 1 else "black")  
        df_hb["periode"] = sluttdato
        resultater_df.append(df_hb)

    if resultater_df:
        return pd.concat(resultater_df, ignore_index=True)
    else:
        return None


def metrics(y_true, y_pred, beta=2.0):
        metrics = {
            "f1_score": f1_score(y_true, y_pred, zero_division=0.0),  # pyright: ignore[reportArgumentType]
            "f2_score": fbeta_score(y_true, y_pred, beta=2.0, zero_division=0.0),  # pyright: ignore[reportArgumentType]
            f"f{beta}_score": fbeta_score(y_true, y_pred, beta=beta, zero_division=0.0),  # pyright: ignore[reportArgumentType]
            "precision": precision_score(y_true, y_pred, zero_division=0.0),  # pyright: ignore[reportArgumentType]
            "recall": recall_score(y_true, y_pred, zero_division=0.0),  # pyright: ignore[reportArgumentType]
        }

        return metrics



def run_accu(df, error):
    if df is None or df.empty:
        return None

    det = Detect(df, "orgnrb")

    df_accu = det.accumulation_error(
        y_var="oms",
        time_var="periode_streng",
        output_format="data",
        error=error,
    )

    if df_accu is None or df_accu.empty:
        return None

    return (
        df_accu[["orgnrb", "periode", "flag_accumulation", "har_feil"]]
        .assign(flag_accu=lambda x: x["flag_accumulation"].fillna(0).astype(int))
    )


# + active=""
# def globale_maal(df):
#     TP = df["tp"].sum()
#     FP = df["fp"].sum()
#     FN = df["fn"].sum()
#
#     precision = TP / (TP + FP) if (TP + FP) > 0 else np.nan
#     recall = TP / (TP + FN) if (TP + FN) > 0 else np.nan
#     f1 = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else np.nan
#
#     return {
#         "global_recall": recall,
#         "global_precision": precision,
#         "global_f1": f1
#     }
# -

metrics_dfs= {}
metrics_hb = {}
metrics_accu = {}
global_df_begge = pd.DataFrame()
global_df_accu = pd.DataFrame()
global_df_hb = pd.DataFrame()
row_ids_all_global = []

# ### Setter verdier som skal brukes i "treningen":

# +
# Setter parametere

# ================================
# ?? MANUELLE VALG
# ================================

ngruppe = "nace2_p2"
valgt_nace2 = "00"
U_VALUES = [0.2, 0.5, 0.9]
C_VALUES = [4.0, 7.0, 10.0]
A_VALUES = [0.05]
ERROR_VALUES= [0.5, 0.8]
# ================================
# ?? OPPSAMLING
# ================================

# -


# #### Kjører akkumuleringsmetoden og lager metrikker:

# +
accu_resultater = []

for errorverdi in ERROR_VALUES:
    for valgt_gid, liste_med_df in nye_dfs.items():
        for idx, df_grp in enumerate(liste_med_df):

            key_for_df = grupper_keys[valgt_gid][idx]
            n_obs = len(df_grp)
            print(
                f"? Kjører ACCU: error={errorverdi}, "
                f"gruppe={valgt_gid}, key={key_for_df}"
                f"Antall obs: {n_obs}"
            )

            # LAG nace2_p2
            sluttdato = pd.to_datetime(df_grp["periode_streng"]).max()

            nace2_p2 = (
                df_grp[pd.to_datetime(df_grp["periode_streng"]) == sluttdato]
                [["orgnrb", "nace2"]]
                .rename(columns={"nace2": "nace2_p2"})
            )
    
            df_grp2 = df_grp.merge(nace2_p2, on="orgnrb", how="left")

            # FILTRER PÅ VALGT NACE
            df_grp2 = df_grp2[df_grp2["nace2_p2"] == valgt_nace2]
            if df_grp2.empty:
                continue
            n_obs2 = len(df_grp2)
            print(
                f"Antall med {valgt_nace2}: obs: {n_obs2}"
            )
            # KJØR ACCU ÉN GANG
            try:
                df_accu = run_accu(df_grp2, error=errorverdi)
            except Exception as e:
                print(f"ACCU feilet: {e}")
                continue
            
            if df_accu is None or df_accu.empty:
                continue

            df_accu = df_accu[df_accu["flag_accumulation"].notna()]
            n_obs3 = len(df_accu)
            print(
                f"Antall til slutt : obs: {n_obs3}"
            )
            # LEGG TIL METADATA
            df_accu = df_accu.assign(
                gruppe=valgt_gid,
                key=key_for_df,
                nacegruppe=valgt_nace2,
                errorverdi=errorverdi,
            )

            accu_resultater.append(df_accu)
            df_accu_alle = pd.concat(accu_resultater, ignore_index=True)

# + active=""
# accu_resultater

# + active=""
# df_accu_alle

# +
accu_metrics = []

for (gruppe, key, errorverdi), g in df_accu_alle.groupby(
    ["gruppe", "key", "errorverdi"]
):
    g = g.copy()

    m = metrics(
        g["har_feil"],
        g["flag_accu"],
        beta=2.0
    )
    
    accu_metrics.append({
        "nacegruppe": g["nacegruppe"].iloc[0],
        "gruppe": gruppe,
        "key": key,
        "U": np.nan,
        "C": np.nan,
        "A": np.nan,
        "accu_precision": m["precision"],
        "accu_recall": m["recall"],
        "accu_fbeta": m["f2.0_score"],
        "accu_f2": m["f2_score"],
        "accu_f1": m["f1_score"],
        "errorverdi": errorverdi,   # behold hvis du vil skille 0.5 / 0.8 senere
        "n_sanne_feil": int((g["har_feil"] == 1).sum()),
        "n_flagget": int((g["flag_accu"] == 1).sum()),

    })
df_metrics_accu = pd.DataFrame(accu_metrics)
# -
df_metrics_accu

# #### Finner beste parametervalg for hver gruppe (jan-feb osv.):

beste_accu_per_gruppe = (
    df_metrics_accu
    .groupby(["gruppe", "nacegruppe", "errorverdi"])
    .agg(
        mean_precision=("accu_precision", "mean"),
        mean_recall=("accu_recall", "mean"),
        mean_f_beta=("accu_fbeta", "mean"),
        mean_f2=("accu_f2", "mean"),
        mean_f1=("accu_f1", "mean"),
        n_keys=("key", "nunique"),
    )
    .reset_index()
    .sort_values("mean_f1", ascending=False)
    .groupby("gruppe")
    .head(1)
)

# #### Kjører HB-metoden og beregner metrikker:

# +
alle_resultater = []
df_metrics_alle_hb = []

for U in U_VALUES:
    for C in C_VALUES:
        for A in A_VALUES:

            print(f"Kjører HB med U={U}, C={C}, A={A}")

            metrics_liste = []

            for valgt_gid, liste_med_df in nye_dfs.items():
                for idx, df_grp in enumerate(liste_med_df):

                    key_for_df = grupper_keys[valgt_gid][idx]

                    # LAG nace2_p2
                    sluttdato = pd.to_datetime(df_grp["periode_streng"]).max()

                    nace2_p2 = (
                        df_grp[pd.to_datetime(df_grp["periode_streng"]) == sluttdato]
                        [["orgnrb", "nace2"]]
                        .rename(columns={"nace2": "nace2_p2"})
                    )

                    df_grp2 = df_grp.merge(nace2_p2, on="orgnrb", how="left")

                    # FILTRER PÅ VALGT NACE
                    df_grp2 = df_grp2[df_grp2["nace2_p2"] == valgt_nace2]
                    if df_grp2.empty:
                        continue

                    # =========================
                    # ?? HB
                    # =========================
                  
                    df_hb = kjør_hb_for_df(df_grp2, U=U, C=C, A=A)
                    if df_hb is None:
                        continue

                    merged = df_hb.merge(
                        df_grp2,
                        on=["orgnrb", "periode"],
                        how="left"
                    )

                    merged["flag_hb_ny"] = 0
                    merged.loc[merged["ratio"] >= 1, "flag_hb_ny"] = merged["flag_hb"]
                    n_rad = len(df_hb)
                    print(f"Antall rader: {n_rad}")
 

                    # =========================
                    # ?? HB-metrics
                    # =========================
                    m_hb = metrics(
                        merged["har_feil"],
                        merged["flag_hb_ny"],
                        beta=2.0
                    )

                    df_metrics_alle_hb.append({
                        "nacegruppe": valgt_nace2,
                        "gruppe": valgt_gid,
                        "key": key_for_df,
                        "U": U,
                        "C": C,
                        "A": A,
                        "hb_precision": m_hb["precision"],
                        "hb_recall": m_hb["recall"],
                        "hb_fbeta": m_hb["f2.0_score"],
                        "hb_f2": m_hb["f2_score"],
                        "hb_f1": m_hb["f1_score"],
                        "errorverdi": np.nan, 
                        "n_sanne_feil": int((merged["har_feil"] == 1).sum()),
                        "n_flagget": int((merged["flag_hb_ny"] == 1).sum()),
                    })

            # FERDIG én (U, C, A)
df_metrics_alle_hb = pd.DataFrame(df_metrics_alle_hb)






# -

# #### Finner beste parametervalg for hver gruppe (jan-feb osv.):

df_metrics_alle_hb["nacegruppe"].unique()

beste_hb_per_gruppe = (
    df_metrics_alle_hb
    .groupby(["gruppe", "nacegruppe", "U", "C", "A"])
    .agg(
        mean_precision=("hb_precision", "mean"),
        mean_recall=("hb_recall", "mean"),
        mean_f_beta=("hb_fbeta", "mean"),
        std_fbeta = ("hb_fbeta", "std"),
        min_fbeta = ("hb_fbeta", "min"),
        mean_f2=("hb_f2", "mean"),
        mean_f1=("hb_f1", "mean"),
        n_keys=("key", "nunique"),
    )
    .reset_index()
    .sort_values("mean_f1", ascending=False)
    .groupby("gruppe",)
    .head(1)
)

# ##### Beregner beste metode totalt for hver gruppe (HB eller akku):

beste_metode_totalt = (
    pd.concat([beste_hb_per_gruppe.assign(type="hb"),
               beste_accu_per_gruppe.assign(type="accu")])
    .sort_values("mean_f1", ascending=False)
    .groupby("gruppe")
    .head(1)
)

# +
først = ["gruppe", "nacegruppe", "type", "errorverdi"]
droppe = ["nacegruppe"]
resten = [c for c in beste_metode_totalt.columns if (c not in først) 
          & (c not in droppe)]

beste_metode_totalt = beste_metode_totalt[først + resten]

# + active=""
# from IPython.display import HTML
#
# HTML(
#     beste_metode_totalt
#         .sort_values(
#             by=["gruppe", "mean_f1"],
#             ascending=[True, False]
#         )
#         .drop(columns=["nacegruppe", "n_keys"])
#         .round(2)
#         .to_html(index=False)
# )
# -

HTML(
    beste_accu_per_gruppe
    .sort_values(by=["gruppe", "mean_f1"], ascending=[True, False])
#    .drop(columns=["nacegruppe", "n_keys"])
    .round(2)
    .to_html(index=False)
)

HTML(
    beste_hb_per_gruppe
    .sort_values(by=["gruppe", "mean_f1"], ascending=[True, False])
#    .drop(columns=["nacegruppe","n_keys", "std_fbeta", "min_fbeta"])
    .round(2)
    .to_html(index=False)
)

# # Kjører med de beste valgene på de nye df'ene:
# ## Dvs. fra januar 2024 til og med april 2025.

aktuelle_keys2 = [
    "202401_202402",
    "202402_202403",
    "202403_202404",
    "202404_202405",
    "202405_202406",
    "202406_202407",
    "202407_202408",
    "202408_202409",
    "202409_202410",
    "202410_202411",
    "202411_202412",
    "202501_202502",
    "202502_202503",
    "202503_202504",
]


filtrerte_dfs = {
    key: splitte_dfs[key]
    for key in aktuelle_keys2
    if key in splitte_dfs
}

total_rows = sum(len(df) for df in filtrerte_dfs.values())
total_rows


# ##### Lager en funksjon som finner gruppenr fra key:

def key_til_gruppe_index(key: str) -> int:
    """
    Eksempel:
    202401_202402 -> 1  (jan-feb)
    202402_202403 -> 2  (feb-mar)
    202403_202404 -> 3  (mar-apr)
    """
    start, slutt = key.split("_")

    slutt_mnd = int(slutt[-2:])   # "202402" -> 2

    # januar-februar skal bli gruppe 1
    gruppe_index = slutt_mnd - 1

    return gruppe_index


beste_lookup = (
    beste_metode_totalt
    .set_index("gruppe")
    .to_dict(orient="index")
)


# +
metrics_per_df = []
y_true_all = []
y_pred_all = []

for key, df_grp in filtrerte_dfs.items():

    if key not in aktuelle_keys2:
        continue

    gruppenr = key_til_gruppe_index(key)
    beste = beste_lookup[gruppenr]
    metode = beste["type"]   # "HB" eller "ACCU"

    print(f"Kjører {metode} for key={key}")

    # ======================
    # Preprocessing
    # ======================
    df_grp = df_grp.copy()
    n_obs = len(df_grp)
    df_grp["periode"] = pd.to_datetime(df_grp["periode_streng"])
    sluttdato = df_grp["periode"].max()

    nace2_p2 = (
        df_grp.loc[
            df_grp["periode"] == sluttdato, ["orgnrb", "nace2"]
        ]
        .rename(columns={"nace2": "nace2_p2"})
    )

    df_grp = df_grp.merge(nace2_p2, on="orgnrb", how="left")
    df_grp = df_grp[df_grp["nace2_p2"] == valgt_nace2]
    
    if df_grp.empty:
        continue

    # ======================
    # Velg og kjør metode
    # ======================

    if metode == "hb":

        df_flagg = kjør_hb_for_df(
            df_grp,
            U=beste["U"],
            C=beste["C"],
            A=beste["A"],
        )

        if df_flagg is None or df_flagg.empty:
            continue

        merged = df_flagg.merge(
            df_grp,
            on=["orgnrb", "periode"],
            how="left"
        )

        merged["flag_pred"] = 0
        merged.loc[
            merged["ratio"] >= 1, "flag_pred"
        ] = merged["flag_hb"]
       
        n_obs1 = len(merged)
    elif metode == "accu":

        merged = run_accu(
            df_grp,
            error=beste["errorverdi"]
        )
        merged = merged[merged["periode"] == sluttdato]
        n_obs1 = len(merged)
        if merged is None or merged.empty:
            continue

        merged["flag_pred"] = merged["flag_accu"]
    else:
        raise ValueError(f"Ukjent metode: {metode}")

    # ======================
    # Metrikk for denne key
    # ======================
    m = metrics(
        merged["har_feil"],
        merged["flag_pred"],
        beta=2.0
    )
    
    y_true_all.extend(merged["har_feil"].astype(int))
    y_pred_all.extend(merged["flag_pred"].astype(int))    

    metrics_per_df.append({
        "key": key,
        "nace": valgt_nace2,
        "metode": metode,
        "precision": m["precision"],
        "recall": m["recall"],
        "fbeta": m["f2.0_score"],
        "f2": m["f2_score"],
        "f1": m["f1_score"],
        "n_sanne_feil": int((merged["har_feil"] == 1).sum()),
        "n_flagget": int((merged["flag_pred"] == 1).sum()),
        "n_obs": n_obs,
        "n_obs1": n_obs1,
    })
    print(f"Antall rader: {n_obs1} ")
metrics_df = pd.DataFrame(metrics_per_df)

# ======================
# ? Lag tmp_df for denne NACE
# ======================
tmp_df = pd.DataFrame({
    "y_true": y_true_all,
    "y_pred": y_pred_all,
    "nace": valgt_nace2,
    "metode": metode,    
})

# ======================
# ? Oppdater global_df
# ======================
global_df_begge = pd.concat([global_df_begge, tmp_df], ignore_index=True)


# -


metrics_df

metrics_df["n_obs"].sum()

metrics_dfs[valgt_nace2] = metrics_df


# + active=""
# global_df_begge
# -

# #### Kjører HB-metoden for alle gruppene:
# #### Finner det beste valget av parametere for HB-metoden for hver gruppe og kjører HB-metoden.


# +
metrics_per_metode = []

y_true_all = []
y_pred_all = []

for key, df_grp in filtrerte_dfs.items():

    if key not in aktuelle_keys2:
        continue

    gruppenr = key_til_gruppe_index(key)

    metode = "hb"  # "HB" eller "ACCU"

    print(f"Kjører {metode} for key={key}")
    
    # ======================
    # ?? Preprocessing
    # ======================
    df_grp = df_grp.copy()
    
    df_grp["periode"] = pd.to_datetime(df_grp["periode_streng"])

    df_grp["row_id"] = (
        df_grp["orgnrb"].astype(str) + "_" + df_grp["periode"].astype(str)
    )

    sluttdato = df_grp["periode"].max()
    startdato = df_grp["periode"].min()

    nace2_p2 = (
        df_grp.loc[
            df_grp["periode"] == sluttdato, ["orgnrb", "nace2"]
        ]
        .rename(columns={"nace2": "nace2_p2"})
    )

    df_grp = df_grp.merge(nace2_p2, on="orgnrb", how="left")
    df_grp = df_grp[df_grp["nace2_p2"] == valgt_nace2]
    if df_grp.empty:
        continue
    n_obs1 = len(df_grp[df_grp["periode"] == startdato])
    n_obs2 = len(df_grp[df_grp["periode"] == sluttdato])
    # ======================
    # ?? Velg og kjør metode
    # ======================
    if metode == "hb":
        rad = beste_hb_per_gruppe.loc[beste_hb_per_gruppe["gruppe"] == gruppenr]

        if rad.empty:
            raise ValueError(f"Fant ingen HB-parametre for gruppe {gruppenr}")

        if len(rad) > 1:
            raise ValueError(f"Flere HB-rader funnet for gruppe {gruppenr}")

        rad = rad.iloc[0]

        df_flagg = kjør_hb_for_df(
            df_grp,
            U=rad["U"],
            C=rad["C"],
            A=rad["A"],
        )
 
       
        if df_flagg is None or df_flagg.empty:
            continue
        n_flagg = len(df_flagg)
        merged = df_flagg.merge(
            df_grp,
            on=["orgnrb", "periode"],
            how="left"
        )

        merged["flag_pred"] = 0
        merged.loc[
            merged["ratio"] >= 1, "flag_pred"
        ] = merged["flag_hb"]

        row_ids_all_global.extend(merged["row_id"])

    elif metode == "accu":
        rad = beste_accu_per_gruppe.loc[beste_accu_per_gruppe["gruppe"] == gruppenr]

        if rad.empty:
            raise ValueError(f"Fant ingen ACCU-parametre for gruppe {gruppenr}")

        if len(rad) > 1:
            raise ValueError(f"Flere ACCU-rader funnet for gruppe {gruppenr}")

        rad = rad.iloc[0]

        merged = run_accu(
            df_grp,
            error=rad["errorverdi"],
            how = "left",
        )

        if merged is None or merged.empty:
            continue

        merged["flag_pred"] = merged["flag_accu"]
        merged = merged[merged["periode"] == sluttdato]
        row_ids_all_global.extend(merged["row_id"])

    else:
        raise ValueError(f"Ukjent metode: {metode}")

    # ======================
    # ?? Metrikk for denne key
    # ======================
    m = metrics(
        merged["har_feil"],
        merged["flag_pred"],
        beta=2.0
    )
    y_true_all.extend(merged["har_feil"].astype(int))
    y_pred_all.extend(merged["flag_pred"].astype(int))    
    n_obs3 = len(merged)

    
    metrics_per_metode.append({
        "key": key,
        "nace": valgt_nace2,
        "metode": metode,
        "precision": m["precision"],
        "recall": m["recall"],
        "f1": m["f1_score"],
        "n_sanne_feil": int((merged["har_feil"] == 1).sum()),
        "n_flagget": int((merged["flag_pred"] == 1).sum()),
        "n_obs2": n_obs2,
        "n_obs3": n_obs3,
        "n_obs1": n_obs1,
        "n_flagg": n_flagg,
    })
# ======================
# ? Lag tmp_df for denne NACE
# ======================
tmp_df = pd.DataFrame({
    "y_true": y_true_all,
    "y_pred": y_pred_all,
    "nace": valgt_nace2,
    "metode": metode,
})

# ======================
# ? Oppdater global_df
# ======================
global_df_hb = pd.concat([global_df_hb, tmp_df], ignore_index=True)
# -

print("Totalt (med duplikater):", len(row_ids_all_global))
print("Unike:", len(set(row_ids_all_global)))

print("Unike rader brukt:", len(set(row_ids_all_global)))
print("Forventet:", 77646)

metrics_hb[valgt_nace2] = metrics_per_metode

type(metrics_hb['47'])

# + active=""
# total = sum(d["n_obs"] for d in metrics_hb["45"])
# total
# -

total = sum(d["n_flagg"] for d in metrics_hb["47"])
total

total3 = sum(d["n_obs1"] for d in metrics_hb["47"])
total3

total3 = sum(d["n_obs2"] for d in metrics_hb["47"])
total3

total3 = sum(d["n_obs3"] for d in metrics_hb["47"])
total3

# + active=""
# total3 = sum(d["n_obs3"] for d in metrics_hb["00"])
# total3
# -

total2 = sum(d["n_obs2"] for d in metrics_hb["47"])
total2

# ### Gjør det samme for akkumuleringsmetoden:

# +
metrics_per_met = []
y_true_all = []
y_pred_all = []

for key, df_grp in filtrerte_dfs.items():

    if key not in aktuelle_keys2:
        continue

    gruppenr = key_til_gruppe_index(key)

    metode = "accu"  # "HB" eller "ACCU"

    print(f"Kjører {metode} for key={key}")

    # ======================
    # ?? Preprocessing
    # ======================
    df_grp = df_grp.copy()

    df_grp["periode"] = pd.to_datetime(df_grp["periode_streng"])
    sluttdato = df_grp["periode"].max()

    nace2_p2 = (
        df_grp.loc[
            df_grp["periode"] == sluttdato, ["orgnrb", "nace2"]
        ]
        .rename(columns={"nace2": "nace2_p2"})
    )

    df_grp = df_grp.merge(nace2_p2, on="orgnrb", how="left")
    df_grp = df_grp[df_grp["nace2_p2"] == valgt_nace2]
    if df_grp.empty:
        continue
    n_obs2 = len(df_grp)
    # ======================
    # ?? Velg og kjør metode
    # ======================
    if metode == "hb":
        rad = beste_hb_per_gruppe.loc[beste_hb_per_gruppe["gruppe"] == gruppenr]

        if rad.empty:
            raise ValueError(f"Fant ingen HB-parametre for gruppe {gruppenr}")

        if len(rad) > 1:
            raise ValueError(f"Flere HB-rader funnet for gruppe {gruppenr}")

        rad = rad.iloc[0]

        df_flagg = kjør_hb_for_df(
            df_grp,
            U=rad["U"],
            C=rad["C"],
            A=rad["A"],
        )
 

        if df_flagg is None or df_flagg.empty:
            continue

        merged = df_flagg.merge(
            df_grp,
            on=["orgnrb", "periode"],
            how="left"
        )

        merged["flag_pred"] = 0
        merged.loc[
            merged["ratio"] >= 1, "flag_pred"
        ] = merged["flag_hb"]
         
    
    elif metode == "accu":
        rad = beste_accu_per_gruppe.loc[beste_accu_per_gruppe["gruppe"] == gruppenr]

        if rad.empty:
            raise ValueError(f"Fant ingen ACCU-parametre for gruppe {gruppenr}")

        if len(rad) > 1:
            raise ValueError(f"Flere ACCU-rader funnet for gruppe {gruppenr}")

        rad = rad.iloc[0]

        merged = run_accu(
            df_grp,
            error=rad["errorverdi"]
        )
        merged = merged[merged["periode"] == sluttdato]
        if merged is None or merged.empty:
            continue

        merged["flag_pred"] = merged["flag_accu"]
        n_obs = len(merged)
    else:
        raise ValueError(f"Ukjent metode: {metode}")

    # ======================
    # ?? Metrikk for denne key
    # ======================
    m = metrics(
        merged["har_feil"],
        merged["flag_pred"],
        beta=2.0
    )
    y_true_all.extend(merged["har_feil"].astype(int))
    y_pred_all.extend(merged["flag_pred"].astype(int))    

   
    metrics_per_met.append({
        "key": key,
        "nace": valgt_nace2,
        "metode": metode,
        "precision": m["precision"],
        "recall": m["recall"],
        "f1": m["f1_score"],
        "n_sanne_feil": int((merged["har_feil"] == 1).sum()),
        "n_flagget": int((merged["flag_pred"] == 1).sum()),
        "n_obs": int(n_obs),
    })
    print(f"Antall rader: {n_obs}")
# ======================
# ? Lag tmp_df for denne NACE
# ======================
tmp_df = pd.DataFrame({
    "y_true": y_true_all,
    "y_pred": y_pred_all,
    "nace": valgt_nace2,
    "metode": metode
})

# ======================
# ? Oppdater global_df
# ======================
global_df_accu = pd.concat([global_df_accu, tmp_df], ignore_index=True)
    
metrics_per_met = pd.DataFrame(metrics_per_met)
# -

metrics_per_met

metrics_accu[valgt_nace2] = metrics_per_met

type(metrics_accu)

total = metrics_accu["47"]["n_obs"].sum()
total

# + active=""
# import pandas as pd
#
# all_data = pd.read_csv("globale_prediksjoner.csv")
#
# globale_metrics = metrics(all_data["y_true"], all_data["y_pred"])
# globale_metrics_df = pd.DataFrame([globale_metrics])
#
# -

metrics_accu[valgt_nace2] = metrics_per_met

len(y_true_all)

len(y_pred_all)

# + active=""
# metrics_accu
# -

global_df_accu.columns

metrics_all = pd.concat(
    metrics_accu.values(),
    ignore_index=True
)

# + active=""
# from IPython.display import HTML
# import pandas as pd
#
# result = globale_maal(metrics_all)
#
# resultater = pd.DataFrame({
#     "Mål": ["Recall", "Precision", "F1"],
#     "Verdi": [
#         result["global_recall"],
#         result["global_precision"],
#         result["global_f1"]
#     ]
# })
#
# HTML(
#     resultater
#         .round(3)
#         .to_html(index=False)
# )
# -

df = metrics_all

# + active=""
# globale_per_key = (
#     df
#     .groupby("key", as_index=False)
#     .apply(globale_maal)
# )

# + active=""
# globale_per_nace = (
#     df
#     .groupby("nace", as_index=False)
#     .apply(globale_maal)
# )

# + active=""
# from IPython.display import HTML
#
# HTML(
#     globale_per_key
#         .round(3)
#         .to_html(index=False)
# )

# + active=""
# from IPython.display import HTML
#
# HTML(
#     globale_per_nace
#         .round(3)
#         .to_html(index=False)
# )
#
# -

# ## Beregner metrikker for beste HB-metode:

# +
globale_metrics_hb_total = metrics(
    global_df_hb["y_true"],
    global_df_hb["y_pred"]
)

globale_metrics_hb_total_df = pd.DataFrame([globale_metrics_hb_total])

# -

globale_metrics_hb_total_df

len(global_df_hb)

len(global_df_accu)

len(global_df_begge)

# + active=""
# df = global_df_hb

# + active=""
# global_recall = (
#     (df["recall"] * df["n_sanne_feil"]).sum()
#     / df["n_sanne_feil"].sum()
# )
# global_precision = (
#     (df["precision"] * df["n_flagget"]).sum()
#     / df["n_flagget"].sum()
# )
# global_f1 = (
#     (df["f1"] * df["n_sanne_feil"]).sum()
#     / df["n_sanne_feil"].sum()
# )

# + active=""
# import pandas as pd
# from IPython.display import HTML
#
# resultater = pd.DataFrame({
#     "Mål": ["Recall", "Precision", "F1"],
#     "Verdi": [global_recall, global_precision, global_f1]
# })

# + active=""
# HTML(
#     resultater
#         .round(3)
#         .to_html(index=False)
# )
# -

# ### Beregener metrikker for beste akku-metode:


# +
globale_metrics_accu_total = metrics(
    global_df_accu["y_true"],
    global_df_accu["y_pred"]
)

globale_metrics_accu_total_df = pd.DataFrame([globale_metrics_accu_total])

# -

globale_metrics_accu_total_df

global_df_accu.columns




