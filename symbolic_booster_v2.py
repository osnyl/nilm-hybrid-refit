"""
symbolic_booster_v2.py

V2: Uses DURATION-BASED symbolic rules.
A water heater runs for a long time (>10 min).
A washing machine has short high-power spikes (<5 min).
This contextual information is key to differentiate them.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('results', exist_ok=True)

# ============================================
# 1. LOAD DATA
# ============================================
print("📂 Loading CLEAN_House1.csv...")
df = pd.read_csv('data/raw/CLEAN_House1.csv')
df_sample = df.head(500000).copy()
print(f"✅ Loaded {len(df_sample)} rows.\n")

# ============================================
# 2. LINEAR MODELS (SAME AS BEFORE)
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

def normalize(X, y):
    X_min, X_max = X.min(axis=0), X.max(axis=0)
    y_min, y_max = y.min(), y.max()
    X_norm = (X - X_min) / (X_max - X_min)
    y_norm = (y - y_min) / (y_max - y_min)
    return X_norm, y_norm, X_min, X_max, y_min, y_max

def denormalize(pred_norm, y_min, y_max):
    return pred_norm * (y_max - y_min) + y_min

X_raw = df_sample[['Aggregate']].values
y_app6 = df_sample['Appliance6'].values  # Washing Machine
y_app9 = df_sample['Appliance9'].values  # Water Heater

print("🚀 Training Linear Models...")
# Washing Machine
X_norm, y_norm, X_min, X_max, y_min6, y_max6 = normalize(X_raw, y_app6)
model6 = LinearModel()
model6.fit(X_norm, y_norm)
pred_raw6 = denormalize(model6.predict(X_norm), y_min6, y_max6)

# Water Heater
X_norm, y_norm, X_min, X_max, y_min9, y_max9 = normalize(X_raw, y_app9)
model9 = LinearModel()
model9.fit(X_norm, y_norm)
pred_raw9 = denormalize(model9.predict(X_norm), y_min9, y_max9)
print("✅ Training complete.\n")

# ============================================
# 3. SYMBOLIC V2 : DURATION-BASED RULES
# ============================================
print("🧠 Applying DURATION-BASED corrections...")

agg = X_raw.flatten()
n = len(agg)

# We will simulate counters for how long the aggregate has been HIGH.
# Since we have 15s intervals: 
# 10 minutes = 40 intervals, 5 minutes = 20 intervals.

counter_high_1500 = 0  # Counts how long we've been above 1500W
counter_high_2000 = 0  # Counts how long we've been above 2000W

corrected_wm = np.copy(pred_raw6)
corrected_wh = np.copy(pred_raw9)

for i in range(n):
    power = agg[i]
    
    # --- Update counters (reset if below threshold) ---
    if power > 1500:
        counter_high_1500 += 1
    else:
        counter_high_1500 = 0
        
    if power > 2000:
        counter_high_2000 += 1
    else:
        counter_high_2000 = 0
    
    # --- RULE 1: WATER HEATER (Long duration) ---
    # If we've been above 1500W for MORE than 10 minutes (40 intervals)
    # it's almost certainly the Water Heater.
    if counter_high_1500 > 40:
        corrected_wh[i] = 1800  # Typical high power of WH
        # If the Water Heater is ON, the Washing Machine can't be the cause of this peak.
        # We lower the WM prediction to avoid double-counting.
        if corrected_wm[i] > 200:
            corrected_wm[i] = 100  # Residual noise
    
    # --- RULE 2: WASHING MACHINE (Short duration) ---
    # If we've been above 2000W for LESS than 5 minutes (20 intervals)
    # but it's a high spike, it's likely the Washing Machine spinning.
    elif power > 2000 and counter_high_2000 < 20:
        corrected_wm[i] = 2000  # Spinning power
        # If WM is ON, the Water Heater is not the culprit.
        if corrected_wh[i] > 300:
            corrected_wh[i] = 0

    # --- RULE 3: Physical Clipping (Safety) ---
    if corrected_wm[i] < 0: corrected_wm[i] = 0
    if corrected_wm[i] > 2500: corrected_wm[i] = 2500
    if corrected_wh[i] < 0: corrected_wh[i] = 0
    if corrected_wh[i] > 2500: corrected_wh[i] = 2500
    
    # --- RULE 4: Low total power -> Everything OFF ---
    if power < 300:
        corrected_wm[i] = 0
        corrected_wh[i] = 0

print("✅ Corrections applied.\n")

# ============================================
# 4. EVALUATION (BEFORE / AFTER)
# ============================================
mae_wm_raw = np.mean(np.abs(pred_raw6 - y_app6))
rmse_wm_raw = np.sqrt(np.mean((pred_raw6 - y_app6) ** 2))
mae_wm_corr = np.mean(np.abs(corrected_wm - y_app6))
rmse_wm_corr = np.sqrt(np.mean((corrected_wm - y_app6) ** 2))

mae_wh_raw = np.mean(np.abs(pred_raw9 - y_app9))
rmse_wh_raw = np.sqrt(np.mean((pred_raw9 - y_app9) ** 2))
mae_wh_corr = np.mean(np.abs(corrected_wh - y_app9))
rmse_wh_corr = np.sqrt(np.mean((corrected_wh - y_app9) ** 2))

print("=" * 70)
print("📊 RÉSULTATS DE L'AMÉLIORATION SYMBOLIQUE V2 (Basée sur la Durée)")
print("=" * 70)

print("\n🔹 LAVE-LINGE (Appliance6) :")
print(f"   Modèle pur (ML) : MAE = {mae_wm_raw:.2f} W  | RMSE = {rmse_wm_raw:.2f} W")
print(f"   ML + Symbolique V2 : MAE = {mae_wm_corr:.2f} W  | RMSE = {rmse_wm_corr:.2f} W")
print(f"   ✅ Amélioration du RMSE : {rmse_wm_raw - rmse_wm_corr:.2f} W")
print(f"   🎯 Réduction de l'erreur : {((rmse_wm_raw - rmse_wm_corr) / rmse_wm_raw) * 100:.1f} %")

print("\n🔹 CHAUFFE-EAU (Appliance9) :")
print(f"   Modèle pur (ML) : MAE = {mae_wh_raw:.2f} W  | RMSE = {rmse_wh_raw:.2f} W")
print(f"   ML + Symbolique V2 : MAE = {mae_wh_corr:.2f} W  | RMSE = {rmse_wh_corr:.2f} W")
print(f"   ✅ Amélioration du RMSE : {rmse_wh_raw - rmse_wh_corr:.2f} W")
print(f"   🎯 Réduction de l'erreur : {((rmse_wh_raw - rmse_wh_corr) / rmse_wh_raw) * 100:.1f} %")

# ============================================
# 5. VISUALIZATION
# ============================================
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
idx = slice(500, 1500)

ax1 = axes[0]
ax1.plot(y_app6[idx], label='Vraie consommation', linewidth=1.5, color='blue')
ax1.plot(pred_raw6[idx], label='Prédiction ML', linestyle='--', alpha=0.7, color='orange')
ax1.plot(corrected_wm[idx], label='ML + Symbolique V2', linestyle=':', linewidth=2, color='green')
ax1.set_title('Lave-linge : Correction par durée (les pics courts sont boostés)')
ax1.legend()
ax1.grid(True)

ax2 = axes[1]
ax2.plot(y_app9[idx], label='Vraie consommation', linewidth=1.5, color='blue')
ax2.plot(pred_raw9[idx], label='Prédiction ML', linestyle='--', alpha=0.7, color='orange')
ax2.plot(corrected_wh[idx], label='ML + Symbolique V2', linestyle=':', linewidth=2, color='green')
ax2.set_title('Chauffe-eau : Correction par durée (les pics longs sont boostés)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('results/symbolic_booster_v2_results.png', dpi=150)
print("\n📈 Graphique sauvegardé : results/symbolic_booster_v2_results.png")
print("\n✅ Script terminé avec succès !")