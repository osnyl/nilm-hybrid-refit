# ============================================================
# 09_merge_refit_iawe_v2.py
# Fusionne REFIT et iAWE pour les 3 appareils réellement communs
# (Frigo, Lave-linge, Télévision), avec le mapping OFFICIEL
# vérifié depuis MetaData_Tables.xlsx (pas celui deviné en début
# de session). Filtre aussi les glitches capteur (valeur appareil
# > agrégat total = physiquement impossible).
# ============================================================

import os
import gc
import sys
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
from appliance_mapping import REFIT_TO_IAWE_COMMON, get_appliance_name  # noqa: E402

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

FLOAT_DTYPE = "float32"


def engineer_features(df):
    df["Time"] = pd.to_datetime(df["Time"])
    df["Hour"] = df["Time"].dt.hour.astype("int8")
    df["Delta_P"] = (df["Aggregate"] - df["Aggregate"].shift(1)).astype(FLOAT_DTYPE)
    return df.dropna().reset_index(drop=True)


def filter_impossible_values(df, appliance_col, aggregate_col="Aggregate"):
    """Un appareil ne peut physiquement pas consommer plus que le
    total de la maison — toute valeur > Aggregate est un glitch
    capteur (comme les 1672 lignes de House 1 détectées plus tôt),
    remise à 0 plutôt que gardée telle quelle."""
    n_glitches = (df[appliance_col] > df[aggregate_col]).sum()
    if n_glitches > 0:
        pct = 100 * n_glitches / len(df)
        print(f"      ⚠️  {n_glitches} glitches détectés ({pct:.3f}%) → mis à 0")
        df.loc[df[appliance_col] > df[aggregate_col], appliance_col] = 0.0
    return df


BASE_COLS = ["Time", "Aggregate", "Delta_P", "Hour", "Source", "House_ID"]
common_path = os.path.join(OUT_DIR, "merged_common_appliances_v2.csv")

# ---- 1. REFIT ----
print("📂 Chargement REFIT (mapping vérifié : Appliance1/5/8)...")
refit_cols_needed = ["Time", "Aggregate"] + list(REFIT_TO_IAWE_COMMON.keys())
refit = pd.read_csv(
    os.path.join(RAW_DIR, "CLEAN_House1.csv"),
    usecols=refit_cols_needed,
    dtype={c: FLOAT_DTYPE for c in refit_cols_needed if c != "Time"},
)
refit = engineer_features(refit)
refit["Source"] = "REFIT"
refit["House_ID"] = "REFIT_House1"
print(f"✅ REFIT : {len(refit)} lignes.")

print("\n🔗 Écriture des lignes REFIT (appareils communs, mapping corrigé)...")
first_write = True
for refit_col, final_name in REFIT_TO_IAWE_COMMON.items():
    real_name = get_appliance_name(refit_col)
    print(f"   {refit_col} = {real_name} → colonne finale '{final_name}'")
    chunk = refit[BASE_COLS + [refit_col]].rename(columns={refit_col: final_name})
    chunk = filter_impossible_values(chunk, final_name)
    chunk["Appliance_Target"] = final_name
    chunk.to_csv(common_path, mode="w" if first_write else "a", header=first_write, index=False)
    first_write = False
    print(f"      {len(chunk)} lignes écrites.")

del refit, chunk
gc.collect()
print("🧹 Mémoire REFIT libérée.")

# ---- 2. iAWE ----
print("\n📂 Chargement iAWE...")
iawe_cols_needed = ["Time", "Aggregate"] + list(REFIT_TO_IAWE_COMMON.values())
iawe = pd.read_csv(
    os.path.join(RAW_DIR, "CLEAN_iAWE_house1.csv"),
    usecols=iawe_cols_needed,
    dtype={c: FLOAT_DTYPE for c in iawe_cols_needed if c != "Time"},
)
iawe = engineer_features(iawe)
iawe["Source"] = "iAWE"
iawe["House_ID"] = "iAWE_House1"
print(f"✅ iAWE : {len(iawe)} lignes.")

print("\n🔗 Ajout des lignes iAWE...")
for final_name in REFIT_TO_IAWE_COMMON.values():
    chunk = iawe[BASE_COLS + [final_name]].copy()
    chunk = filter_impossible_values(chunk, final_name)
    chunk["Appliance_Target"] = final_name
    chunk.to_csv(common_path, mode="a", header=False, index=False)
    print(f"   {final_name} (iAWE) : {len(chunk)} lignes ajoutées.")

del iawe, chunk
gc.collect()
print("🧹 Mémoire iAWE libérée.")

print(f"\n✅ Fusion (mapping corrigé) terminée : {common_path}")
print("\n💡 Rappel des corrections apportées par rapport à la première version :")
print("   - Ancien (faux) : Appliance4→TV, Appliance6→Lave-linge")
print("   - Nouveau (officiel) : Appliance5→Lave-linge, Appliance8→TV")
print("   - Glitches capteur (valeur > agrégat) mis à 0 automatiquement")
