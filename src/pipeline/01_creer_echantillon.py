"""
01_creer_echantillon.py

Crée un échantillon stratifié (100 000 lignes par appareil ET par source)
en échantillonnant ALÉATOIREMENT dans tout le dataset (pas les premières
lignes), pour capturer les cycles rares (lave-linge).

Entrée  : data/processed/donnees_fusionnees_refit_iawe_v3.csv
Sorties : data/processed/echantillon_entrainement.csv
          data/processed/echantillon_test.csv
"""

import os
import gc
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT = os.path.join(BASE, "data/processed/donnees_fusionnees_refit_iawe_v3.csv")
OUTPUT_TRAIN = os.path.join(BASE, "data/processed/echantillon_entrainement.csv")
OUTPUT_TEST = os.path.join(BASE, "data/processed/echantillon_test.csv")

N_PER_STRATE = 100_000
TRAIN_RATIO = 0.80
RANDOM_STATE = 42
APPLIANCES = ["Fridge", "WashingMachine", "Television"]
SOURCES = ["REFIT", "iAWE"]

# Stockage : on va lire tout le dataset et garder TOUTES les lignes
# de chaque strate, puis échantillonner à la fin.
samples = {(a, s): [] for a in APPLIANCES for s in SOURCES}
counts = {(a, s): 0 for a in APPLIANCES for s in SOURCES}

print("Lecture complète du dataset (toutes les lignes)...\n")
CHUNK_SIZE = 500_000
total_read = 0

for chunk in pd.read_csv(INPUT, chunksize=CHUNK_SIZE):
    total_read += len(chunk)

    for (appliance, source) in samples.keys():
        mask = (chunk["Appliance_Target"] == appliance) & (chunk["Source"] == source)
        sub = chunk.loc[mask]
        if len(sub) > 0:
            samples[(appliance, source)].append(sub.copy())
            counts[(appliance, source)] += len(sub)

    print(f"  {total_read:,} lignes lues...")
    del chunk
    gc.collect()

# Maintenant, on échantillonne aléatoirement dans chaque strate
print("\nÉchantillonnage aléatoire par strate...")
final_samples = []
for key, lst in samples.items():
    if len(lst) == 0:
        print(f"  ⚠️ {key} : aucune donnée")
        continue

    full = pd.concat(lst, ignore_index=True)
    n = min(N_PER_STRATE, len(full))
    sampled = full.sample(n=n, random_state=RANDOM_STATE)

    # Vérifier les valeurs non nulles
    n_nonzero = (sampled["Puissance_Cible"] > 0).sum()
    print(f"  {key} : {len(full):,} disponibles → {n:,} échantillonnées "
          f"({n_nonzero:,} valeurs > 0)")

    final_samples.append(sampled)
    del full
    gc.collect()

# Concaténation
print("\nConcaténation...")
final = pd.concat(final_samples, ignore_index=True)
print(f"  Total : {len(final):,} lignes")

# Split temporel
print("\nSplit temporel...")
final["Time"] = pd.to_datetime(final["Time"])
final = final.sort_values("Time").reset_index(drop=True)
cutoff = int(len(final) * TRAIN_RATIO)
train = final.iloc[:cutoff].copy()
test = final.iloc[cutoff:].copy()

print(f"  Train : {len(train):,} lignes")
print(f"  Test  : {len(test):,} lignes")

# Vérification finale
print("\nVérification des valeurs non nulles dans le train :")
for appliance in APPLIANCES:
    n = (train[train["Appliance_Target"] == appliance]["Puissance_Cible"] > 0).sum()
    print(f"  {appliance} : {n:,}")

train.to_csv(OUTPUT_TRAIN, index=False)
test.to_csv(OUTPUT_TEST, index=False)
print(f"\n✅ Sauvegardés.")

del samples, final_samples, final, train, test
gc.collect()
