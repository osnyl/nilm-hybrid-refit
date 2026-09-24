# ============================================================
# 08_plaid_power_reference.py
# Calcule la puissance active réelle (P = V x I, moyennée sur un
# cycle) à partir de PLAID, pour construire une table de référence
# par appareil, matchable au Delta_P (en Watts) de REFIT/iAWE.
#
# IMPORTANT : seule la puissance en RÉGIME ÉTABLI (P_steady_W) est
# réellement comparable au signal REFIT/iAWE à 8s. Le pic d'appel
# (P_peak_W) est documenté mais reste invisible à cette résolution.
# ============================================================

import json
import os
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.expanduser("~/Desktop/experience-ia-symbolique/nilm-hybrid-refit")
PLAID_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "PLAID")
CSV_DIR = os.path.join(PLAID_DIR, "2017")
OUT_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(OUT_DIR, exist_ok=True)

SAMPLING_FREQ = 30000
CYCLE_FREQ = 60
WINDOW = int(SAMPLING_FREQ / CYCLE_FREQ)  # 500 échantillons = 1 cycle

TARGET_TYPES = ["Air Conditioner", "Fridge", "Washing Machine", "Vacuum",
                 "Microwave", "Heater"]

print("📂 Chargement de meta_2017.json...")
with open(os.path.join(PLAID_DIR, "meta_2017.json")) as f:
    meta_2017 = json.load(f)

targets = [e for e in meta_2017 if e["meta"]["appliance"]["type"] in TARGET_TYPES]
print(f"✅ {len(targets)} instances ciblées.")


def active_power_envelope(current, voltage, window=WINDOW):
    """Puissance active instantanée p(t) = v(t) x i(t), lissée par
    une MOYENNE glissante d'un cycle (pas un RMS) — c'est la bonne
    façon de calculer une puissance active réelle en régime
    alternatif : la puissance active est la moyenne de v*i sur un
    cycle complet, pas sa racine carrée moyenne."""
    p_instant = voltage.astype(np.float64) * current.astype(np.float64)
    kernel = np.ones(window) / window
    p_envelope = np.convolve(p_instant, kernel, mode="same")
    return p_envelope


results = []
print("\n🚀 Calcul de la puissance active par instance...\n")

for entry in targets:
    file_id = entry["id"]
    csv_path = os.path.join(CSV_DIR, f"{file_id}.csv")
    if not os.path.exists(csv_path):
        continue

    df = pd.read_csv(csv_path, header=None, names=["current", "voltage"])
    p_env = active_power_envelope(df["current"].values, df["voltage"].values)

    peak_power = float(np.max(np.abs(p_env)))
    peak_idx = int(np.argmax(np.abs(p_env)))
    time_to_peak_ms = (peak_idx / SAMPLING_FREQ) * 1000

    # IMPORTANT : le régime établi (appareil ALLUMÉ, stabilisé) n'est
    # pas toujours à la fin du clip. Pour une transition "off-on",
    # l'appareil est ON à la fin -> dernier quart = bon.
    # Pour une transition "on-off", l'appareil est OFF à la fin ->
    # il faut regarder le PREMIER quart (avant la coupure), qui lui
    # est encore ON et stabilisé.
    n = len(p_env)
    transition = entry["meta"]["appliance"].get("status", "off-on")
    if transition == "on-off":
        steady = p_env[: n // 4]       # début du clip = encore allumé
    else:
        steady = p_env[n * 3 // 4:]    # fin du clip = allumé stabilisé (off-on, cas standard)
    steady_power = float(np.mean(np.abs(steady)))

    appliance = entry["meta"]["appliance"]
    results.append({
        "id": file_id,
        "type": appliance["type"],
        "transition": appliance.get("status", "?"),  # "off-on" ou "on-off"
        "wattage_nominal": appliance.get("wattage", ""),
        "P_peak_W": round(peak_power, 1),           # NON comparable à REFIT/iAWE (trop rapide)
        "time_to_peak_ms": round(time_to_peak_ms, 1),
        "P_steady_W": round(steady_power, 1),        # COMPARABLE à Delta_P (REFIT/iAWE)
    })

df_results = pd.DataFrame(results)
detail_path = os.path.join(OUT_DIR, "plaid_power_detail.csv")
df_results.to_csv(detail_path, index=False)
print(f"✅ Détail sauvegardé : {detail_path}")

# ---- Table de référence résumée, une ligne par type ----
ref_table = df_results.groupby("type").agg(
    P_steady_W_mean=("P_steady_W", "mean"),
    P_steady_W_std=("P_steady_W", "std"),
    P_peak_W_mean=("P_peak_W", "mean"),
    n_instances=("id", "count"),
).round(1).reset_index()

ref_path = os.path.join(OUT_DIR, "appliance_power_reference.csv")
ref_table.to_csv(ref_path, index=False)

print(f"\n📊 Table de référence par type d'appareil :")
print(ref_table.to_string(index=False))
print(f"\n✅ Sauvegardée : {ref_path}")

print("\n💡 Utilisation dans 04_symbolic_booster.py :")
print("   Remplace tes seuils devinés à l'œil (ex: 'delta > 500') par")
print("   P_steady_W_mean +/- P_steady_W_std du type d'appareil concerné.")
print("   N'utilise PAS P_peak_W comme seuil de détection sur REFIT/iAWE —")
print("   ce pic dure < 1s et ton signal à 8s ne peut physiquement pas le voir.")
