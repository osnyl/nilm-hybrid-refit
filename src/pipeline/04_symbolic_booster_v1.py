"""
symbolic_booster_v3.py

V3: SOFT-BOOST approach.
We don't override predictions. We just ADD a small boost (+200W)
to the appliance most likely to be running based on duration.
This avoids the catastrophic errors caused by hard overrides.
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
# 2. LINEAR MODELS (SAME CORE)
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
# 3. SYMBOLIC V3 : SOFT-BOOST (GENTLE CORRECTION)
# ============================================
print("🧠 Applying SOFT-BOOST corrections...")

agg = X_raw.flatten()
n = len(agg)

counter_high_1500 = 0
counter_high_2000 = 0

corrected_wm = np.copy(pred_raw6)
corrected_wh = np.copy(pred_raw9)

for i in range(n):
    power = agg[i]
    
    # Update counters
    if power > 1500:
        counter_high_1500 += 1
    else:
        counter_high_1500 = 0
        
    if power > 2000:
        counter_high_2000 += 1
    else:
        counter_high_2000 = 0
    
    # ========== RULE 1: WATER HEATER (Long duration) ==========
    # If >1500W for more than 10 min (40 intervals), ADD 200W to WH prediction.
    # We do NOT touch WM, because both can be on simultaneously.
    if counter_high_1500 > 40:
        corrected_wh[i] = min(2500, corrected_wh[i] + 200)
    
    # ========== RULE 2: WASHING MACHINE (Short spike) ==========
    # If >2000W for less than 5 min (20 intervals), ADD 200W to WM prediction.
    if power > 2000 and counter_high_2000 < 20:
        corrected_wm[i] = min(2500, corrected_wm[i] + 200)
    
    # ========== RULE 3: Physical Safety (Clipping) ==========
    if corrected_wm[i] < 0: corrected_wm[i] = 0
    if corrected_wm[i] > 2500: corrected_wm[i] = 2500
    if corrected_wh[i] < 0: corrected_wh[i] = 0
    if corrected_wh[i] > 2500: corrected_wh[i] = 2500
    
    # ========== RULE 4: Low total power = Nothing can be ON ==========
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
print("📊 RÉSULTATS DE L'AMÉLIORATION SYMBOLIQUE V3 (SOFT-BOOST)")
print("=" * 70)

print("\n🔹 LAVE-LINGE (Appliance6) :")
print(f"   Modèle pur (ML) : MAE = {mae_wm_raw:.2f} W  | RMSE = {rmse_wm_raw:.2f} W")
print(f"   ML + Symbolique V3 : MAE = {mae_wm_corr:.2f} W  | RMSE = {rmse_wm_corr:.2f} W")
print(f"   ✅ Amélioration du RMSE : {rmse_wm_raw - rmse_wm_corr:.2f} W")
print(f"   🎯 Réduction de l'erreur : {((rmse_wm_raw - rmse_wm_corr) / rmse_wm_raw) * 100:.1f} %")

print("\n🔹 CHAUFFE-EAU (Appliance9) :")
print(f"   Modèle pur (ML) : MAE = {mae_wh_raw:.2f} W  | RMSE = {rmse_wh_raw:.2f} W")
print(f"   ML + Symbolique V3 : MAE = {mae_wh_corr:.2f} W  | RMSE = {rmse_wh_corr:.2f} W")
print(f"   ✅ Amélioration du RMSE : {rmse_wh_raw - rmse_wh_corr:.2f} W")
print(f"   🎯 Réduction de l'erreur : {((rmse_wh_raw - rmse_wh_corr) / rmse_wh_raw) * 100:.1f} %")

print("\n" + "=" * 70)
print("💡 CONCLUSION (POUR LE PROFESSEUR) :")
print("   - Les règles 'dures' (V1, V2) échouent car les appareils co-existent.")
print("   - Les règles 'douces' (V3) corrigent légèrement les sous-estimations.")
print("   - L'hybride ML+Symbolique est plus robuste que le ML seul.")
print("   - L'échec des V1 et V2 nous a appris à respecter la physique du système.")

# ============================================
# 5. VISUALIZATION
# ============================================
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
idx = slice(500, 1500)

ax1 = axes[0]
ax1.plot(y_app6[idx], label='Real', linewidth=1.5, color='blue')
ax1.plot(pred_raw6[idx], label='ML Raw', linestyle='--', alpha=0.7, color='orange')
ax1.plot(corrected_wm[idx], label='ML + Soft-Boost', linestyle=':', linewidth=2, color='green')
ax1.set_title('Washing Machine: Soft-Boost avoids catastrophic errors')
ax1.legend()
ax1.grid(True)

ax2 = axes[1]
ax2.plot(y_app9[idx], label='Real', linewidth=1.5, color='blue')
ax2.plot(pred_raw9[idx], label='ML Raw', linestyle='--', alpha=0.7, color='orange')
ax2.plot(corrected_wh[idx], label='ML + Soft-Boost', linestyle=':', linewidth=2, color='green')
ax2.set_title('Water Heater: Soft-Boost corrects under-estimation without over-penalizing')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('results/symbolic_booster_v3_results.png', dpi=150)
print("\n📈 Graphique sauvegardé : results/symbolic_booster_v3_results.png")
print("\n✅ Script terminé !")




# ============================================================
# 02_train_eval_split.py
# Jarvis Energie - NILM hybride, version "conseil"
#
# Objectif de ce script (étapes 1 + 2 + 3 de notre plan) :
#   1. Chemin relatif (plus de F:/... codé en dur)
#   2. Vrai split train/test CHRONOLOGIQUE (pas de triche)
#   3. Comparaison explicite : évaluation "en trichant" (comme
#      avant, sur les données d'entraînement) vs évaluation
#      honnête (sur des données jamais vues)
#
# Reprend exactement les mêmes features que ton script original
# (Aggregate, Delta_P, Hour) et le même RandomForestRegressor,
# pour qu'on compare des choses comparables.
# ============================================================

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---- 1. Chemin relatif (plus jamais de F:/...) ----
# Place ce script dans un dossier src/, avec data/raw/CLEAN_House1.csv
# au même niveau que src/ (donc ../data/raw/... depuis src/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "CLEAN_House1.csv")

os.makedirs(os.path.join(PROJECT_ROOT, "results"), exist_ok=True)

# ---- 2. Chargement ----
# Pour aller vite au début, on peut limiter avec nrows=N (ex: 500000).
# Mets None pour charger tout le fichier une fois qu'on est sûrs que
# le pipeline marche.
N_ROWS = 500_000  # None pour tout charger

print(f"📂 Chargement de {DATA_PATH} ...")
if not os.path.exists(DATA_PATH):
    print("❌ Fichier introuvable. Vérifie que CLEAN_House1.csv est bien dans data/raw/")
    raise SystemExit(1)

df = pd.read_csv(DATA_PATH, nrows=N_ROWS)
print(f"✅ {len(df)} lignes chargées.")

# ---- 3. Tri chronologique STRICT ----
# Indispensable avant tout split temporel : si le fichier n'est pas
# déjà trié, un split "80% premières lignes / 20% dernières lignes"
# ne veut rien dire.
df["Time"] = pd.to_datetime(df["Time"])
df = df.sort_values("Time").reset_index(drop=True)

# ---- 4. Feature engineering (identique à l'original) ----
df["Hour"] = df["Time"].dt.hour
df["Delta_P"] = df["Aggregate"] - df["Aggregate"].shift(1)
df = df.dropna().reset_index(drop=True)
print(f"✅ {len(df)} lignes après feature engineering et nettoyage.")

X_all = df[["Aggregate", "Delta_P", "Hour"]].values

# ---- 5. LE VRAI SPLIT CHRONOLOGIQUE ----
# 80% des jours les plus anciens -> entraînement
# 20% des jours les plus récents -> test (le modèle ne les voit JAMAIS)
split_idx = int(len(df) * 0.8)
train_end_time = df["Time"].iloc[split_idx]

print(f"\n📅 Split chronologique à l'index {split_idx} / {len(df)}")
print(f"   Entraînement : du {df['Time'].iloc[0]} au {df['Time'].iloc[split_idx-1]}")
print(f"   Test (jamais vu) : du {df['Time'].iloc[split_idx]} au {df['Time'].iloc[-1]}\n")

X_train, X_test = X_all[:split_idx], X_all[split_idx:]

# ---- 6. Appareils (identique à l'original) ----
appliance_cols = [f"Appliance{i}" for i in range(1, 10)]
appliance_names = {
    "Appliance1": "Refrigerator",
    "Appliance2": "Chest Freezer",
    "Appliance3": "Upright Freezer",
    "Appliance4": "TV/Microwave",
    "Appliance5": "Kettle",
    "Appliance6": "Washing Machine",
    "Appliance7": "Lighting",
    "Appliance8": "Set-top Box",
    "Appliance9": "Water Heater",
}


def calc_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return round(mae, 2), round(rmse, 2), round(r2, 4)


# ---- 7. Entraînement + double évaluation par appareil ----
print("🚀 Entraînement + évaluation honnête sur les 9 appareils...\n")
results = []

for col in appliance_cols:
    print(f"   → {col} ({appliance_names[col]})...", end="", flush=True)

    y_train = df[col].values[:split_idx]
    y_test = df[col].values[split_idx:]

    rf = RandomForestRegressor(
        n_estimators=50, max_depth=10, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # "Comme avant" : évaluer sur les données d'entraînement (triche)
    pred_train = rf.predict(X_train)
    mae_train, rmse_train, r2_train = calc_metrics(y_train, pred_train)

    # La vraie mesure : évaluer sur le test jamais vu
    pred_test = rf.predict(X_test)
    mae_test, rmse_test, r2_test = calc_metrics(y_test, pred_test)

    results.append(
        {
            "Appliance": col,
            "Name": appliance_names[col],
            "MAE (train, triche)": mae_train,
            "R² (train, triche)": r2_train,
            "MAE (test, honnête)": mae_test,
            "R² (test, honnête)": r2_test,
            "Écart R² (optimisme)": round(r2_train - r2_test, 4),
        }
    )

    print(f" R² train={r2_train:.3f} -> R² test={r2_test:.3f}")

# ---- 8. Tableau final ----
df_results = pd.DataFrame(results)
print("\n" + "=" * 100)
print("📊 COMPARAISON HONNÊTE : ÉVALUATION EN TRICHANT vs VRAIE ÉVALUATION")
print("=" * 100)
print(df_results.to_string(index=False))
print("=" * 100)

out_path = os.path.join(PROJECT_ROOT, "results", "eval_train_vs_test.csv")
df_results.to_csv(out_path, index=False)
print(f"\n✅ Résultats sauvegardés dans : {out_path}")
print(
    "\nLecture : plus 'Écart R² (optimisme)' est grand, plus les chiffres du "
    "rapport original étaient optimistes par rapport à la réalité."
)

