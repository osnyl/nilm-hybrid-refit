import pandas as pd
import numpy as np


df = pd.read_csv('data/raw/CLEAN_House1.csv')


power_values = df['Aggregate'].iloc[0:1000].values


current_values = power_values / 220.0


time_step = 8
with open('load_data.txt', 'w') as f:
    for i, current in enumerate(current_values):
        t = i * time_step
        f.write(f"{t:.1f} {current:.3f}\n")

print(" Fichier load_data.txt généré avec succès !")
print(f"   {len(current_values)} lignes écrites.")
