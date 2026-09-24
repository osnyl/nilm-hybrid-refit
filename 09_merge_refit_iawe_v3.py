"""
09_merge_refit_iawe_v3.py

Version corrigée : produit UNE SEULE colonne 'Puissance_Cible',
remplie pour chaque ligne selon l'appareil (Appliance_Target).
"""

import os
import gc
import sys
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.expanduser("~/Desktop/experience-ia-symbolique/nilm-hybrid-refit")
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
from appliance_mapping import REFIT_TO_IAWE_COMMON, get_appliance_name

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUT = os.path.join(OUT_DIR, "donnees_fusionnees_refit_iawe_v3.csv")
os.makedirs(OUT_DIR, exist_ok=True)

FLOAT_DTYPE = "float32"

def engineer_features(df):
    df["Time"] = pd.to_datetime(df["Time"])
    df["Hour"] = df["Time"].dt.hour.astype("int8")
    df["Delta_P"] = (df["Aggregate"] - df["Aggregate"].shift(1)).astype(FLOAT_DTYPE)
    return df.dropna().reset_index(drop=True)

BASE_COLS = ["Time", "Aggregate", "Delta_P", "Hour", "Source", "House_ID"]

# ---- 1. REFIT ----
print("📂 Chargement REFIT...")
refit_cols = ["Time", "Aggregate"] + list(REFIT_TO_IAWE_COMMON.keys())
refit = pd.read_csv(
    os.path.join(RAW_DIR, "CLEAN_House1.csv"),
    usecols=refit_cols,
    dtype={c: FLOAT_DTYPE for c in refit_cols if c != "Time"},
)
refit = engineer_features(refit)
refit["Source"] = "REFIT"
refit["House_ID"] = "REFIT_House1"
print(f"✅ REFIT : {len(refit)} lignes.")

# Empiler les 3 appareils avec UNE SEULE colonne cible
chunks = []
for refit_col, final_name in REFIT_TO_IAWE_COMMON.items():
    real_name = get_appliance_name(refit_col)
    chunk = refit[BASE_COLS].copy()
    chunk["Appliance_Target"] = final_name
    chunk["Puissance_Cible"] = refit[refit_col].values  # <-- UNE SEULE COLONNE
    chunks.append(chunk)
    print(f"   {real_name} → {len(chunk)} lignes")

refit_final = pd.concat(chunks, ignore_index=True)
del refit, chunks
gc.collect()
print(f"✅ REFIT empilé : {len(refit_final)} lignes")

# ---- 2. iAWE ----
print("\n📂 Chargement iAWE...")
iawe_cols = ["Time", "Aggregate"] + list(REFIT_TO_IAWE_COMMON.values())
iawe = pd.read_csv(
    os.path.join(RAW_DIR, "CLEAN_iAWE_house1.csv"),
    usecols=iawe_cols,
    dtype={c: FLOAT_DTYPE for c in iawe_cols if c != "Time"},
)
iawe = engineer_features(iawe)
iawe["Source"] = "iAWE"
iawe["House_ID"] = "iAWE_House1"
print(f"✅ iAWE : {len(iawe)} lignes.")

chunks = []
for final_name in REFIT_TO_IAWE_COMMON.values():
    chunk = iawe[BASE_COLS].copy()
    chunk["Appliance_Target"] = final_name
    chunk["Puissance_Cible"] = iawe[final_name].values  # <-- UNE SEULE COLONNE
    chunks.append(chunk)
    print(f"   {final_name} → {len(chunk)} lignes")

iawe_final = pd.concat(chunks, ignore_index=True)
del iawe, chunks
gc.collect()
print(f"✅ iAWE empilé : {len(iawe_final)} lignes")

# ---- 3. Fusion finale ----
print("\n🔗 Fusion REFIT + iAWE...")
final = pd.concat([refit_final, iawe_final], ignore_index=True)
print(f"✅ Total : {len(final):,} lignes")

# Sauvegarde
final.to_csv(OUTPUT, index=False)
print(f"✅ Sauvegardé : {OUTPUT}")

# Aperçu
print("\nAperçu :")
print(final.head(10).to_string())

del final, refit_final, iawe_final
gc.collect()
