import pandas as pd
import numpy as np

# 1. Charger le dataset REFIT (utilise ton chemin)
df = pd.read_csv('data/raw/CLEAN_House1.csv')

# 2. Extraire 1000 lignes de la colonne Aggregate (puissance totale)
power_values = df['Aggregate'].iloc[0:1000].values

# 3. Convertir en courant : I = P / 220V
current_values = power_values / 220.0

# 4. Créer le fichier load_data.txt avec le format temps (en secondes) et courant
time_step = 8  # REFIT échantillonne toutes les 8 secondes
with open('load_data.txt', 'w') as f:
    for i, current in enumerate(current_values):
        t = i * time_step
        f.write(f"{t:.1f} {current:.3f}\n")

print("✅ Fichier load_data.txt généré avec succès !")
print(f"   {len(current_values)} lignes écrites.")
