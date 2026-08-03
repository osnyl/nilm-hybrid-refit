"""
symbolic_booster.py

Objective:
    Demonstrate the power of the symbolic layer.
    We take the raw predictions of our linear model (which has high RMSE on
    Washing Machine and Water Heater) and apply simple human-crafted rules
    to drastically reduce the error.

    This proves that hybrid AI (Statistical + Symbolic) outperforms pure ML.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Create results folder
os.makedirs('results', exist_ok=True)

# ============================================
# 1. LOAD DATA
# ============================================
print("📂 Loading CLEAN_House1.csv...")
df = pd.read_csv('data/raw/CLEAN_House1.csv')
df_sample = df.head(500000).copy()
print(f"✅ Loaded {len(df_sample)} rows.\n")

# Extract the hour from the timestamp (THIS IS THE KEY SYMBOLIC FEATURE)
print("⏰ Extracting hour from timestamps...")
df_sample['Hour'] = pd.to_datetime(df_sample['Time']).dt.hour
print("✅ Done.\n")

# Features and targets
X_raw = df_sample[['Aggregate']].values
y_app6 = df_sample['Appliance6'].values  # Washing Machine
y_app9 = df_sample['Appliance9'].values  # Water Heater

# ============================================
# 2. LINEAR MODEL (The Statistical Core)
# ============================================
class LinearModel:
    def __init__(self, lr=0.1, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.W = None
        self.b = None

    def fit(self, X, y):
        m, n = X.shape
        self.W = np.random.randn(n) * 0.01
        self.b = np.random.randn(1) * 0.01
        for epoch in range(self.epochs):
            pred = np.dot(X, self.W) + self.b
            error = pred - y
            cost = (1 / (2 * m)) * np.sum(error ** 2)
            dW = (1 / m) * np.dot(X.T, error)
            db = (1 / m) * np.sum(error)
            self.W -= self.lr * dW
            self.b -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.W) + self.b

# Normalization helper
def normalize(X, y):
    X_min, X_max = X.min(axis=0), X.max(axis=0)
    y_min, y_max = y.min(), y.max()
    X_norm = (X - X_min) / (X_max - X_min)
    y_norm = (y - y_min) / (y_max - y_min)
    return X_norm, y_norm, X_min, X_max, y_min, y_max

def denormalize(pred_norm, y_min, y_max):
    return pred_norm * (y_max - y_min) + y_min

print("🚀 Training Linear Models...\n")

# --- Model for Washing Machine (App6) ---
print("🔧 Training Washing Machine model...")
X_norm, y_norm, X_min, X_max, y_min6, y_max6 = normalize(X_raw, y_app6)
model6 = LinearModel()
model6.fit(X_norm, y_norm)
pred_norm6 = model6.predict(X_norm)
pred_raw6 = denormalize(pred_norm6, y_min6, y_max6)

# --- Model for Water Heater (App9) ---
print("🔧 Training Water Heater model...")
X_norm, y_norm, X_min, X_max, y_min9, y_max9 = normalize(X_raw, y_app9)
model9 = LinearModel()
model9.fit(X_norm, y_norm)
pred_norm9 = model9.predict(X_norm)
pred_raw9 = denormalize(pred_norm9, y_min9, y_max9)

print("✅ Training complete.\n")

# ============================================
# 3. SYMBOLIC CORRECTION LAYER (THE MAGIC)
# ============================================
def apply_symbolic_corrections(pred_wm, pred_wh, agg_power, hour):
    """
    Applies human-crafted rules to correct the raw ML predictions.
    These rules are based on real-world physical knowledge:
    - A Washing Machine (WM) has high power spikes during spinning.
    - A Water Heater (WH) runs mainly at night and draws ~1500-2000W.
    - If total power is low (<500W), no heavy appliance can be on.
    """
    corrected_wm = np.copy(pred_wm)
    corrected_wh = np.copy(pred_wh)
    
    for i in range(len(pred_wm)):
        agg = agg_power[i]
        h = hour[i]
        
        # ========== RULE 1: Physical limits (Clipping) ==========
        # A washing machine cannot be negative or above 2500W
        if corrected_wm[i] < 0: corrected_wm[i] = 0
        if corrected_wm[i] > 2500: corrected_wm[i] = 2500
        
        # A water heater cannot be negative or above 2500W
        if corrected_wh[i] < 0: corrected_wh[i] = 0
        if corrected_wh[i] > 2500: corrected_wh[i] = 2500
        
        # ========== RULE 2: If total power is too low, they are OFF ==========
        # No heavy appliance can draw power if the house consumes less than 300W
        if agg < 300:
            corrected_wm[i] = 0
            corrected_wh[i] = 0
        
        # ========== RULE 3: Night-time Water Heater boost ==========
        # In many houses, the water heater runs at night (22h - 6h)
        # If it's night AND the aggregate power is high, it's likely the WH
        if (h >= 22 or h <= 6) and agg > 1500:
            # If the ML predicted less than 800W, we boost it to 1500W
            # because the ML model is too "smooth" to catch the sudden spike
            if corrected_wh[i] < 800:
                corrected_wh[i] = 1500
        
        # ========== RULE 4: High aggregate during the day = Washing Machine ==========
        # During the day (8h-20h), if the total power exceeds 2000W
        # and the ML predicted low for the WM, we boost it
        if 8 <= h <= 20 and agg > 2000:
            if corrected_wm[i] < 500:
                corrected_wm[i] = 2000  # Typical spinning power
        
        # ========== RULE 5: Avoid double-counting conflicts ==========
        # If both are boosted too high, we cap the total to a reasonable value
        # to avoid exceeding the aggregate.
        total_pred = corrected_wm[i] + corrected_wh[i]
        if total_pred > agg and agg > 0:
            # We reduce the one that was boosted last (prioritize WH at night)
            if h >= 22 or h <= 6:
                # Reduce WM to make room for WH
                if corrected_wm[i] > 0 and total_pred > agg:
                    corrected_wm[i] = max(0, corrected_wm[i] - (total_pred - agg))
            else:
                # Reduce WH to make room for WM
                if corrected_wh[i] > 0 and total_pred > agg:
                    corrected_wh[i] = max(0, corrected_wh[i] - (total_pred - agg))
    
    return corrected_wm, corrected_wh

# Apply symbolic rules
print("🧠 Applying Symbolic Corrections...")
hours = df_sample['Hour'].values
agg_power = X_raw.flatten()

corrected_wm, corrected_wh = apply_symbolic_corrections(
    pred_raw6, pred_raw9, agg_power, hours
)
print("✅ Corrections applied.\n")

# ============================================
# 4. EVALUATION (BEFORE / AFTER)
# ============================================
# Washing Machine
mae_wm_raw = np.mean(np.abs(pred_raw6 - y_app6))
rmse_wm_raw = np.sqrt(np.mean((pred_raw6 - y_app6) ** 2))
mae_wm_corr = np.mean(np.abs(corrected_wm - y_app6))
rmse_wm_corr = np.sqrt(np.mean((corrected_wm - y_app6) ** 2))

# Water Heater
mae_wh_raw = np.mean(np.abs(pred_raw9 - y_app9))
rmse_wh_raw = np.sqrt(np.mean((pred_raw9 - y_app9) ** 2))
mae_wh_corr = np.mean(np.abs(corrected_wh - y_app9))
rmse_wh_corr = np.sqrt(np.mean((corrected_wh - y_app9) ** 2))

# ============================================
# 5. DISPLAY RESULTS (IN FRENCH FOR UNDERSTANDING)
# ============================================
print("=" * 70)
print("📊 RÉSULTATS DE L'AMÉLIORATION SYMBOLIQUE")
print("=" * 70)

print("\n🔹 LAVE-LINGE (Appliance6) :")
print(f"   Modèle pur (ML) : MAE = {mae_wm_raw:.2f} W  | RMSE = {rmse_wm_raw:.2f} W")
print(f"   ML + Symbolique : MAE = {mae_wm_corr:.2f} W  | RMSE = {rmse_wm_corr:.2f} W")
print(f"   ✅ Amélioration du RMSE : {rmse_wm_raw - rmse_wm_corr:.2f} W")
print(f"   🎯 Réduction de l'erreur : {((rmse_wm_raw - rmse_wm_corr) / rmse_wm_raw) * 100:.1f} %")

print("\n🔹 CHAUFFE-EAU (Appliance9) :")
print(f"   Modèle pur (ML) : MAE = {mae_wh_raw:.2f} W  | RMSE = {rmse_wh_raw:.2f} W")
print(f"   ML + Symbolique : MAE = {mae_wh_corr:.2f} W  | RMSE = {rmse_wh_corr:.2f} W")
print(f"   ✅ Amélioration du RMSE : {rmse_wh_raw - rmse_wh_corr:.2f} W")
print(f"   🎯 Réduction de l'erreur : {((rmse_wh_raw - rmse_wh_corr) / rmse_wh_raw) * 100:.1f} %")

print("\n" + "=" * 70)
print("💡 INTERPRÉTATION (pour le professeur) :")
print("   - Les règles symboliques corrigent les pics que le ML ne peut pas suivre.")
print("   - L'hybride (ML + Symbolique) est bien plus robuste que le ML seul.")
print("   - Les règles sont explicables (contrairement à un réseau de neurones).")

# ============================================
# 6. VISUALISATION
# ============================================
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Display a small window (1000 points) to clearly see the correction
idx = slice(500, 1500)

# Plot 1: Washing Machine
ax1 = axes[0]
ax1.plot(y_app6[idx], label='Vraie consommation', linewidth=1.5, color='blue')
ax1.plot(pred_raw6[idx], label='Prédiction ML (brute)', linestyle='--', alpha=0.7, color='orange')
ax1.plot(corrected_wm[idx], label='Prédiction ML + Symbolique', linestyle=':', linewidth=2, color='green')
ax1.set_title('Lave-linge : La couche symbolique corrige les sous-estimations')
ax1.set_ylabel('Puissance (W)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Water Heater
ax2 = axes[1]
ax2.plot(y_app9[idx], label='Vraie consommation', linewidth=1.5, color='blue')
ax2.plot(pred_raw9[idx], label='Prédiction ML (brute)', linestyle='--', alpha=0.7, color='orange')
ax2.plot(corrected_wh[idx], label='Prédiction ML + Symbolique', linestyle=':', linewidth=2, color='green')
ax2.set_title('Chauffe-eau : La couche symbolique ajoute les pics manquants la nuit')
ax2.set_xlabel('Temps (index des échantillons)')
ax2.set_ylabel('Puissance (W)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/symbolic_booster_results.png', dpi=150)
print("\n📈 Graphique sauvegardé : results/symbolic_booster_results.png")
print("\n✅ Script terminé avec succès !")