"""
02_associer_references_plaid.py

Associe les références PLAID aux échantillons d'entraînement et de test.

Entrées : data/processed/echantillon_entrainement.csv
          data/processed/echantillon_test.csv
          results/references_plaid_par_appareil.csv   <-- nom corrigé
Sorties : data/processed/echantillon_entrainement_plaid.csv
          data/processed/echantillon_test_plaid.csv
"""

import os
import gc
import pandas as pd

BASE = os.path.expanduser("~/Desktop/experience-ia-symbolique/nilm-hybrid-refit")
INPUT_TRAIN = os.path.join(BASE, "data/processed/echantillon_entrainement.csv")
INPUT_TEST = os.path.join(BASE, "data/processed/echantillon_test.csv")
INPUT_PLAID = os.path.join(BASE, "results/references_plaid_par_appareil.csv")
OUTPUT_TRAIN = os.path.join(BASE, "data/processed/echantillon_entrainement_plaid.csv")
OUTPUT_TEST = os.path.join(BASE, "data/processed/echantillon_test_plaid.csv")

APPLIANCE_MAPPING = {
    "Fridge": "Fridge",
    "WashingMachine": "Washing Machine",
    "Television": None,
}

print("Chargement des références PLAID...")
plaid = pd.read_csv(INPUT_PLAID)
plaid_simple = plaid.groupby("type").agg(
    P_steady_mean_W=("P_steady_mean_W", "mean"),
    P_steady_std_W=("P_steady_std_W", "mean"),
    P_peak_mean_W=("P_peak_mean_W", "mean"),
    n_regimes=("status", "count"),
).reset_index()
print(f"  {len(plaid_simple)} types d'appareils")

def enrich(file_in, file_out, label):
    print(f"\nEnrichissement de {label}...")
    df = pd.read_csv(file_in)
    print(f"  {len(df):,} lignes lues")
    df["plaid_type"] = df["Appliance_Target"].map(APPLIANCE_MAPPING)
    merged = df.merge(plaid_simple, left_on="plaid_type", right_on="type", how="left", suffixes=("", "_plaid"))
    merged = merged.drop(columns=["plaid_type", "type"], errors="ignore")
    merged.to_csv(file_out, index=False)
    print(f"  ✅ {len(merged):,} lignes écrites dans {file_out}")
    print(f"  Références PLAID manquantes : {merged['P_steady_mean_W'].isna().sum():,}")
    return merged

train = enrich(INPUT_TRAIN, OUTPUT_TRAIN, "echantillon_entrainement")
test = enrich(INPUT_TEST, OUTPUT_TEST, "echantillon_test")

del plaid, plaid_simple, train, test
gc.collect()
print("\n✅ Terminé.")
