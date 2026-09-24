"""
graphique_resultats.py

Génère un graphique comparant RF pur vs RF + Symbolique
pour les 3 appareils (MAE et R²).
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT = os.path.join(BASE, "results/resultats_modele_hybride.csv")
OUTPUT = os.path.join(BASE, "figures/comparaison_rf_vs_symbolique.png")

# Charger les résultats
df = pd.read_csv(INPUT)
print("Données chargées :")
print(df.to_string(index=False))

# Créer le graphique
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

appliances = df["Appareil"].tolist()
x = np.arange(len(appliances))
width = 0.35

# Graphique 1 : MAE
ax1 = axes[0]
ax1.bar(x - width/2, df["MAE_RF"], width, label="RF pur", color="orange")
ax1.bar(x + width/2, df["MAE_Symbolique"], width, label="RF + Symbolique", color="green")
ax1.set_xlabel("Appareil")
ax1.set_ylabel("MAE (W)")
ax1.set_title("Comparaison des MAE : RF pur vs RF + Symbolique")
ax1.set_xticks(x)
ax1.set_xticklabels(appliances)
ax1.legend()
ax1.grid(True, alpha=0.3, axis="y")

# Graphique 2 : R²
ax2 = axes[1]
ax2.bar(x - width/2, df["R2_RF"], width, label="RF pur", color="orange")
ax2.bar(x + width/2, df["R2_Symbolique"], width, label="RF + Symbolique", color="green")
ax2.set_xlabel("Appareil")
ax2.set_ylabel("R²")
ax2.set_title("Comparaison des R² : RF pur vs RF + Symbolique")
ax2.set_xticks(x)
ax2.set_xticklabels(appliances)
ax2.legend()
ax2.grid(True, alpha=0.3, axis="y")
ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.8)

plt.tight_layout()
plt.savefig(OUTPUT, dpi=150)
print(f"\n✅ Graphique sauvegardé : {OUTPUT}")
plt.show()
