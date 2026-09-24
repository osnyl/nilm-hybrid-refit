"""
03_entrainer_modele_hybride.py (version corrigée)

La colonne 'Fridge' contient la puissance de l'appareil cible,
quel que soit l'appareil (Fridge, WashingMachine, Television).
On filtre par Appliance_Target pour séparer les appareils.

Entrées : data/processed/echantillon_entrainement_plaid.csv
          data/processed/echantillon_test_plaid.csv
          src/regles_symboliques.py
Sortie  : results/resultats_modele_hybride.csv
"""

import os
import gc
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, SRC_DIR)
from regles_symboliques import appliquer_regles, SEUILS_PLAID

BASE = PROJECT_ROOT
INPUT_TRAIN = os.path.join(BASE, "data/processed/echantillon_entrainement_plaid.csv")
INPUT_TEST = os.path.join(BASE, "data/processed/echantillon_test_plaid.csv")
OUTPUT = os.path.join(BASE, "results/resultats_modele_hybride.csv")

print("Chargement des données...")
train = pd.read_csv(INPUT_TRAIN)
test = pd.read_csv(INPUT_TEST)
print(f"  Train : {len(train):,} lignes")
print(f"  Test  : {len(test):,} lignes")

FEATURES = ["Aggregate", "Delta_P", "Hour"]
APPLIANCES = ["Fridge", "WashingMachine", "Television"]
TARGET_COL = "Puissance_Cible"  # colonne de puissance cible (générique)

results = []

for appliance in APPLIANCES:
    print(f"\n=== {appliance} ===")
    train_app = train[train["Appliance_Target"] == appliance]
    test_app = test[test["Appliance_Target"] == appliance]

    if len(train_app) == 0 or len(test_app) == 0:
        print(f"  ⚠️ Pas assez de données, on passe.")
        continue

    X_train = train_app[FEATURES].values
    y_train = train_app[TARGET_COL].values
    X_test = test_app[FEATURES].values
    y_test = test_app[TARGET_COL].values

    rf = RandomForestRegressor(
        n_estimators=50,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    y_pred_sym = np.array([
        appliquer_regles(
            appliance,
            delta=test_app["Delta_P"].values[i],
            hour=test_app["Hour"].values[i],
            pred=y_pred_rf[i],
        )
        for i in range(len(y_pred_rf))
    ])

    def compute_metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        return round(mae, 2), round(rmse, 2), round(r2, 4)

    mae_rf, rmse_rf, r2_rf = compute_metrics(y_test, y_pred_rf)
    mae_sym, rmse_sym, r2_sym = compute_metrics(y_test, y_pred_sym)

    print(f"  RF pur          : MAE={mae_rf} W | RMSE={rmse_rf} W | R²={r2_rf}")
    print(f"  RF + Symbolique : MAE={mae_sym} W | RMSE={rmse_sym} W | R²={r2_sym}")

    results.append({
        "Appareil": appliance,
        "MAE_RF": mae_rf,
        "MAE_Symbolique": mae_sym,
        "RMSE_RF": rmse_rf,
        "RMSE_Symbolique": rmse_sym,
        "R2_RF": r2_rf,
        "R2_Symbolique": r2_sym,
        "Gain_MAE": round(mae_rf - mae_sym, 2),
    })

df_results = pd.DataFrame(results)
print("\n" + "=" * 90)
print("📊 RÉSULTATS FINAUX (avec PLAID + Soft-Boost)")
print("=" * 90)
print(df_results.to_string(index=False))
print("=" * 90)

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
df_results.to_csv(OUTPUT, index=False)
print(f"\n✅ Sauvegardé : {OUTPUT}")

del train, test, df_results
gc.collect()
