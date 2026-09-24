"""
08_plaid_power_reference_v2.py

Calcule des références de puissance PAR RÉGIME pour chaque type d'appareil PLAID.
Au lieu d'une moyenne globale (qui mélange ventilateur et compresseur),
on distingue les régimes stables des régimes transitoires.

Sortie : results/appliance_power_reference_v2.csv
"""

import json
import os
import gc
import pandas as pd
import numpy as np

# ---- Chemins ----
PLAID_DIR = os.path.expanduser(
    "~/Desktop/experience-ia-symbolique/nilm-hybrid-refit/data/raw/PLAID"
)
RESULTS_DIR = os.path.expanduser(
    "~/Desktop/experience-ia-symbolique/nilm-hybrid-refit/results"
)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---- 1. Charger le meta PLAID 2017 ----
print("Chargement de meta_2017.json...")
with open(os.path.join(PLAID_DIR, "meta_2017.json")) as f:
    meta = json.load(f)

meta_rows = []
for entry in meta:
    meta_rows.append({
        "id": str(entry["id"]),
        "status": entry["meta"]["appliance"].get("status", "MANQUANT"),
        "type": entry["meta"]["appliance"].get("type", "MANQUANT"),
    })
meta_df = pd.DataFrame(meta_rows)
print(f"  {len(meta_df)} instances dans meta_2017.json")

# ---- 2. Charger le détail PLAID (issu du script 07/08) ----
detail_path = os.path.join(RESULTS_DIR, "plaid_power_detail.csv")
df = pd.read_csv(detail_path)
df["id"] = df["id"].astype(str)
print(f"  {len(df)} instances dans plaid_power_detail.csv")

# ---- 3. Jointure ----
merged = df.merge(meta_df, on="id", how="left", suffixes=("", "_meta"))
print(f"  {len(merged)} lignes après jointure")
print(f"  Status manquants : {merged['status'].isna().sum()}")

# ---- 4. Calcul des références par régime ----

def compute_regime_stats(group):
    """Calcule les stats de puissance pour un (type, status) donné."""
    p_steady = group["P_steady_W"].dropna()
    p_peak = group["P_peak_W"].dropna()
    if len(p_steady) == 0:
        return None
    return pd.Series({
        "n_instances": len(group),
        "P_steady_mean_W": round(p_steady.mean(), 2),
        "P_steady_std_W": round(p_steady.std(), 2),
        "P_steady_min_W": round(p_steady.min(), 2),
        "P_steady_max_W": round(p_steady.max(), 2),
        "P_peak_mean_W": round(p_peak.mean(), 2) if len(p_peak) > 0 else np.nan,
    })

print("\nCalcul des références par (type, status)...")
regime_stats = merged.groupby(["type", "status"]).apply(compute_regime_stats).reset_index()

# ---- 5. Filtrer les régimes avec trop peu d'instances ----
MIN_INSTANCES = 5
regime_stats = regime_stats[regime_stats["n_instances"] >= MIN_INSTANCES]
print(f"  {len(regime_stats)} régimes retenus (>= {MIN_INSTANCES} instances)")

# ---- 6. Affichage ----
print("\n=== RÉFÉRENCES DE PUISSANCE PAR RÉGIME ===")
print(regime_stats.to_string(index=False))

# ---- 7. Sauvegarde ----
out_path = os.path.join(RESULTS_DIR, "appliance_power_reference_v2.csv")
regime_stats.to_csv(out_path, index=False)
print(f"\n✅ Sauvegardé : {out_path}")

# ---- 8. Nettoyage mémoire ----
del df, meta_df, merged
gc.collect()
