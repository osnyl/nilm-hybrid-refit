# ============================================================
# train_temporal_model.py
# Modèle linéaire avec DELTA de puissance + HEURE
# ============================================================

import os
import pandas as pd
import numpy as np

# ---- 1. Dossier de travail ----
project_path = 'F:/experience-ia-symbolique/nilm-hybrid-refit'
if os.path.exists(project_path):
    os.chdir(project_path)
    print(f"✅ Dossier : {os.getcwd()}")
else:
    print("⚠️ Chemin introuvable.")
    exit()

# ---- 2. Chargement du dataset ----
print("📂 Chargement du dataset REFIT...")
df = pd.read_csv('data/raw/CLEAN_House1.csv')
print(f"✅ Chargé : {len(df)} lignes.")

# ---- 3. Échantillonnage (500 000 lignes) ----
df_sample = df.head(500000).copy()
print(f"📊 Utilisation de {len(df_sample)} lignes.")

# ---- 4. Feature Engineering ----
print("🔄 Création des nouvelles features...")

# 4.1. Extraire l'heure (de 0 à 23)
df_sample['Time'] = pd.to_datetime(df_sample['Time'])
df_sample['Hour'] = df_sample['Time'].dt.hour

# 4.2. Créer le delta de puissance (variation)
df_sample['Delta_P'] = df_sample['Aggregate'] - df_sample['Aggregate'].shift(1)

# 4.3. Supprimer les lignes avec des NaN (les premières lignes)
df_sample = df_sample.dropna()

# ---- 5. Préparer X et y ----
# X : puissance actuelle + delta + heure
X_raw = df_sample[['Aggregate', 'Delta_P', 'Hour']].values
y_raw = df_sample['Appliance9'].values  # cela correspond au chauffe-eau

print(f"📊 X (avec delta + heure) : {X_raw.shape}")
print(f"📊 y : {y_raw.shape}")

# ---- 6. Normalisation ----
def normalize(X, y):
    X_min, X_max = X.min(axis=0), X.max(axis=0)
    y_min, y_max = y.min(), y.max()
    X_norm = (X - X_min) / (X_max - X_min + 1e-10)
    y_norm = (y - y_min) / (y_max - y_min + 1e-10)
    return X_norm, y_norm, X_min, X_max, y_min, y_max

def denormalize(pred_norm, y_min, y_max):
    return pred_norm * (y_max - y_min) + y_min

X_norm, y_norm, X_min, X_max, y_min, y_max = normalize(X_raw, y_raw)

# ---- 7. Modèle linéaire ----
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
            if epoch % 500 == 0 and epoch > 0:
                print(f"   Epoch {epoch:4d} | Loss: {cost:.6f}")

    def predict(self, X):
        return np.dot(X, self.W) + self.b

print("🚀 Entraînement du modèle avec Delta + Heure...")
model = LinearModel(lr=0.1, epochs=1000)
model.fit(X_norm, y_norm)

# ---- 8. Évaluation ----
pred_norm = model.predict(X_norm)
pred_raw = denormalize(pred_norm, y_min, y_max)

mae = np.mean(np.abs(pred_raw - y_raw))
rmse = np.sqrt(np.mean((pred_raw - y_raw) ** 2))
ss_res = np.sum((y_raw - pred_raw) ** 2)
ss_tot = np.sum((y_raw - np.mean(y_raw)) ** 2)
r2 = 1 - (ss_res / ss_tot)

print("\n📊 RÉSULTATS AVEC DELTA + HEURE")
print("=" * 50)
print(f"MAE  : {mae:.2f} W")
print(f"RMSE : {rmse:.2f} W")
print(f"R²   : {r2:.4f}")
print("=" * 50)

# ---- 9. Comparaison ----
print("\n📌 Comparaison avec le modèle simple (sans delta, sans heure) :")
print(f"   Ancien MAE  : 24.14 W")
print(f"   Ancien RMSE : 31.89 W")






# ---- 5. Préparer X ----
X_raw = df_sample[['Aggregate', 'Delta_P', 'Hour']].values

# ---- 6. Définition des 9 appareils ----
appliance_cols = [f'Appliance{i}' for i in range(1, 10)]

appliance_names = {
    'Appliance1': 'Refrigerator',
    'Appliance2': 'Chest Freezer',
    'Appliance3': 'Upright Freezer',
    'Appliance4': 'TV/Microwave',
    'Appliance5': 'Kettle',
    'Appliance6': 'Washing Machine',
    'Appliance7': 'Lighting',
    'Appliance8': 'Set-top Box',
    'Appliance9': 'Water Heater'
}

# ---- 7. Fonctions ----
def normalize(X, y):
    X_min, X_max = X.min(axis=0), X.max(axis=0)
    y_min, y_max = y.min(), y.max()
    X_norm = (X - X_min) / (X_max - X_min + 1e-10)
    y_norm = (y - y_min) / (y_max - y_min + 1e-10)
    return X_norm, y_norm, X_min, X_max, y_min, y_max

def denormalize(pred_norm, y_min, y_max):
    return pred_norm * (y_max - y_min) + y_min

class LinearModel:
    def __init__(self, lr=0.1, epochs=500):
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

# ---- 8. Boucle sur tous les appareils ----
print("\n🚀 Évaluation sur les 9 appareils...\n")
results = []

for col in appliance_cols:
    print(f"   → {col} ({appliance_names[col]})...", end='', flush=True)
    
    y_raw = df_sample[col].values
    X_norm, y_norm, X_min, X_max, y_min, y_max = normalize(X_raw, y_raw)
    
    model = LinearModel(lr=0.1, epochs=500)
    model.fit(X_norm, y_norm)
    
    pred_norm = model.predict(X_norm)
    pred_raw = denormalize(pred_norm, y_min, y_max)
    
    mae = np.mean(np.abs(pred_raw - y_raw))
    rmse = np.sqrt(np.mean((pred_raw - y_raw) ** 2))
    ss_res = np.sum((y_raw - pred_raw) ** 2)
    ss_tot = np.sum((y_raw - np.mean(y_raw)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    results.append({
        'Appliance': col,
        'Name': appliance_names[col],
        'MAE (W)': round(mae, 2),
        'RMSE (W)': round(rmse, 2),
        'R²': round(r2, 4),
        'Max Power (W)': round(y_max, 2)
    })
    
    print(f" MAE={mae:.2f}W, R²={r2:.4f}")

# ---- 9. Tableau final ----
import pandas as pd
df_results = pd.DataFrame(results)

print("\n" + "=" * 80)
print("📊 RÉSULTATS COMPLETS : MODÈLE DELTA + HEURE")
print("=" * 80)
print(df_results.to_string(index=False))
print("=" * 80)

# ---- 10. Sauvegarde ----
os.makedirs('results', exist_ok=True)
df_results.to_csv('results/temporal_model_results.csv', index=False)
print("\n✅ Résultats sauvegardés dans : results/temporal_model_results.csv")










# ============================================================
# train_temporal_model.py
# Modèle Random Forest avec Delta + Heure
# ============================================================

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---- 1. Dossier de travail ----
project_path = 'F:/experience-ia-symbolique/nilm-hybrid-refit'
if os.path.exists(project_path):
    os.chdir(project_path)
else:
    print("⚠️ Chemin introuvable.")
    exit()

# ---- 2. Chargement du dataset ----
print("📂 Chargement du dataset REFIT...")
df = pd.read_csv('data/raw/CLEAN_House1.csv')
print(f"✅ Chargé : {len(df)} lignes.")

# ---- 3. Échantillonnage (500 000 lignes) ----
df_sample = df.head(500000).copy()
print(f"📊 Utilisation de {len(df_sample)} lignes.")

# ---- 4. Feature Engineering ----
print("🔄 Construction des features...")
df_sample['Time'] = pd.to_datetime(df_sample['Time'])
df_sample['Hour'] = df_sample['Time'].dt.hour
df_sample['Delta_P'] = df_sample['Aggregate'] - df_sample['Aggregate'].shift(1)
df_sample = df_sample.dropna()
print(f"✅ {len(df_sample)} lignes après nettoyage.\n")

# ---- 5. Préparer X ----
X_raw = df_sample[['Aggregate', 'Delta_P', 'Hour']].values

# ---- 6. Définition des 9 appareils ----
appliance_cols = [f'Appliance{i}' for i in range(1, 10)]
appliance_names = {
    'Appliance1': 'Refrigerator',
    'Appliance2': 'Chest Freezer',
    'Appliance3': 'Upright Freezer',
    'Appliance4': 'TV/Microwave',
    'Appliance5': 'Kettle',
    'Appliance6': 'Washing Machine',
    'Appliance7': 'Lighting',
    'Appliance8': 'Set-top Box',
    'Appliance9': 'Water Heater'
}

print("🚀 Entraînement du modèle Random Forest sur les 9 appareils...\n")
results = []

# ---- 7. Boucle sur les appareils ----
for col in appliance_cols:
    print(f"   → {col} ({appliance_names[col]})...", end='', flush=True)
    
    y_raw = df_sample[col].values
    
    # Modèle Random Forest (100 arbres, profondeur max 10 pour éviter le sur-apprentissage)
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_raw, y_raw)
    
    # Prédiction et évaluation
    pred_raw = model.predict(X_raw)
    
    mae = mean_absolute_error(y_raw, pred_raw)
    rmse = np.sqrt(mean_squared_error(y_raw, pred_raw))
    ss_res = np.sum((y_raw - pred_raw) ** 2)
    ss_tot = np.sum((y_raw - np.mean(y_raw)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    results.append({
        'Appliance': col,
        'Name': appliance_names[col],
        'MAE (W)': round(mae, 2),
        'RMSE (W)': round(rmse, 2),
        'R²': round(r2, 4),
        'Max Power (W)': round(y_raw.max(), 2)
    })
    
    print(f" MAE={mae:.2f}W, R²={r2:.4f}")

# ---- 8. Tableau final ----
df_results = pd.DataFrame(results)
print("\n" + "=" * 80)
print("📊 RÉSULTATS RANDOM FOREST : DELTA + HEURE")
print("=" * 80)
print(df_results.to_string(index=False))
print("=" * 80)

# ---- 9. Sauvegarde ----
os.makedirs('results', exist_ok=True)
df_results.to_csv('results/random_forest_results.csv', index=False)
print("\n✅ Résultats sauvegardés dans : results/random_forest_results.csv")




# ============================================================
# EXPERIENCE 2 : SYSTÈME HYBRIDE (ML + SYMBOLIQUE + ENSEMBLE)
# Comparaison des performances sur les 9 appareils REFIT
# ============================================================

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---- 1. Dossier de travail ----
project_path = 'F:/experience-ia-symbolique/nilm-hybrid-refit'
if os.path.exists(project_path):
    os.chdir(project_path)
else:
    print("⚠️ Chemin introuvable.")
    exit()

# ---- 2. Chargement du dataset ----
print("📂 Chargement du dataset REFIT...")
df = pd.read_csv('data/raw/CLEAN_House1.csv')
print(f"✅ Chargé : {len(df)} lignes.")

# ---- 3. Échantillonnage (500 000 lignes) ----
df_sample = df.head(500000).copy()
print(f"📊 Utilisation de {len(df_sample)} lignes.")

# ---- 4. Feature Engineering ----
print("🔄 Construction des features...")
df_sample['Time'] = pd.to_datetime(df_sample['Time'])
df_sample['Hour'] = df_sample['Time'].dt.hour
df_sample['Delta_P'] = df_sample['Aggregate'] - df_sample['Aggregate'].shift(1)
df_sample = df_sample.dropna()
print(f"✅ {len(df_sample)} lignes après nettoyage.\n")

# ---- 5. Préparation des données ----
X = df_sample[['Aggregate', 'Delta_P', 'Hour']].values

appliance_cols = [f'Appliance{i}' for i in range(1, 10)]
appliance_names = {
    'Appliance1': 'Refrigerator',
    'Appliance2': 'Chest Freezer',
    'Appliance3': 'Upright Freezer',
    'Appliance4': 'TV/Microwave',
    'Appliance5': 'Kettle',
    'Appliance6': 'Washing Machine',
    'Appliance7': 'Lighting',
    'Appliance8': 'Set-top Box',
    'Appliance9': 'Water Heater'
}

results = []
print("🚀 Entraînement et évaluation du système hybride...\n")

# ---- 6. Boucle sur les 9 appareils ----
for col in appliance_cols:
    print(f"   → {col} ({appliance_names[col]})...", end='', flush=True)
    
    y = df_sample[col].values
    
    # ---- 6.1 Modèle ML pur (Random Forest) ----
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    y_pred_rf = rf.predict(X)
    
    # ---- 6.2 Régression linéaire (pour l'ensemble) ----
    lr = LinearRegression()
    lr.fit(X, y)
    y_pred_lr = lr.predict(X)
    
    # ---- 6.3 Correction symbolique (Soft-Boost) ----
    y_pred_symbolic = y_pred_rf.copy()
    delta = df_sample['Delta_P'].values
    hour = df_sample['Hour'].values
    agg = df_sample['Aggregate'].values
    
    for i in range(len(y_pred_symbolic)):
        # Règle 1 : Clipping physique (toujours active)
        if y_pred_symbolic[i] < 0:
            y_pred_symbolic[i] = 0
        if y_pred_symbolic[i] > y.max() * 1.2:
            y_pred_symbolic[i] = y.max() * 1.2
        
        # Règle 2 : Pic court (bouilloire, micro-ondes)
        if abs(delta[i]) > 500 and y_pred_rf[i] < 50:
            y_pred_symbolic[i] = abs(delta[i]) * 0.8
        
        # Règle 3 : Pic long (chauffe-eau, lave-linge)
        if 300 < abs(delta[i]) < 500 and y_pred_rf[i] < 100:
            y_pred_symbolic[i] = abs(delta[i]) * 0.7
        
        # Règle 4 : Contexte horaire (chauffe-eau la nuit)
        if (hour[i] < 6 or hour[i] > 22) and agg[i] > 1500 and y_pred_rf[i] < 100:
            y_pred_symbolic[i] = 1500
    
    # ---- 6.4 Ensemble : moyenne pondérée (Linéaire + RF) ----
    # On donne plus de poids à la RF si elle est meilleure (basé sur le MAE)
    # Ici, on simplifie : moyenne simple des deux
    y_pred_ensemble = (y_pred_rf + y_pred_lr) / 2
    
    # ---- 6.5 Métriques ----
    def compute_metrics(y_true, y_pred, name, col):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        return {
            'Appliance': col,
            'Name': name,
            'MAE (W)': round(mae, 2),
            'RMSE (W)': round(rmse, 2),
            'R²': round(r2, 4)
        }
    
    metrics_rf = compute_metrics(y, y_pred_rf, appliance_names[col] + ' (RF)', col)
    metrics_symbolic = compute_metrics(y, y_pred_symbolic, appliance_names[col] + ' (Symbolique)', col)
    metrics_ensemble = compute_metrics(y, y_pred_ensemble, appliance_names[col] + ' (Ensemble)', col)
    metrics_lr = compute_metrics(y, y_pred_lr, appliance_names[col] + ' (Linéaire)', col)
    
    results.append(metrics_rf)
    results.append(metrics_symbolic)
    results.append(metrics_ensemble)
    results.append(metrics_lr)
    
    print(f" RF={metrics_rf['MAE (W)']}W | Sym={metrics_symbolic['MAE (W)']}W | Ens={metrics_ensemble['MAE (W)']}W")

# ---- 7. Affichage du tableau comparatif ----
df_results = pd.DataFrame(results)
print("\n" + "=" * 100)
print("📊 RÉSULTATS COMPARATIFS : MODÈLES PURS vs HYBRIDE (Symbolique + Ensemble)")
print("=" * 100)
print(df_results.to_string(index=False))
print("=" * 100)

# ---- 8. Sauvegarde ----
os.makedirs('results', exist_ok=True)
df_results.to_csv('results/experience2_hybride_results.csv', index=False)
print("\n✅ Résultats sauvegardés dans : results/experience2_hybride_results.csv")











# EXPERIENCE 3 : SYSTÈME HYBRIDE INTELLIGENT (Soft-Boost)


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---- Créer le dossier des figures ----
os.makedirs('figures', exist_ok=True)

# ---- 1. Dossier de travail ----
project_path = 'F:/experience-ia-symbolique/nilm-hybrid-refit'
if os.path.exists(project_path):
    os.chdir(project_path)
else:
    print("⚠️ Chemin introuvable.")
    exit()

# ---- 2. Chargement du dataset ----
print("📂 Chargement du dataset REFIT...")
df = pd.read_csv('data/raw/CLEAN_House1.csv')
df_sample = df.head(100000).copy()
print(f"✅ {len(df_sample)} lignes.")

# ---- 3. Feature Engineering ----
print("🔄 Construction des features...")
df_sample['Time'] = pd.to_datetime(df_sample['Time'])
df_sample['Hour'] = df_sample['Time'].dt.hour
df_sample['Delta_P'] = df_sample['Aggregate'] - df_sample['Aggregate'].shift(1)
df_sample = df_sample.dropna()
print(f"✅ {len(df_sample)} lignes après nettoyage.\n")

X = df_sample[['Aggregate', 'Delta_P', 'Hour']].values

appliance_cols = [f'Appliance{i}' for i in range(1, 10)]
appliance_names = {
    'Appliance1': 'Refrigerator',
    'Appliance2': 'Chest Freezer',
    'Appliance3': 'Upright Freezer',
    'Appliance4': 'TV_Microwave',  # <-- modifié pour éviter le slash
    'Appliance5': 'Kettle',
    'Appliance6': 'Washing_Machine',  # modifié pour éviter l'espace
    'Appliance7': 'Lighting',
    'Appliance8': 'Set-top_Box',
    'Appliance9': 'Water_Heater'
}

results = []
print("🚀 Entraînement du système hybride intelligent...\n")

for col in appliance_cols:
    print(f"   → {col} ({appliance_names[col]})...", end='', flush=True)
    
    y = df_sample[col].values

    # ---- Modèle RF ----
    rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    y_pred_rf = rf.predict(X)

    # ---- Correction symbolique (Soft-Boost intelligent) ----
    y_pred_symbolic = y_pred_rf.copy()
    delta = df_sample['Delta_P'].values
    hour = df_sample['Hour'].values
    agg = df_sample['Aggregate'].values

    for i in range(len(y_pred_symbolic)):
        if col == 'Appliance6':  # Lave-linge
            if 300 < abs(delta[i]) < 800 and (hour[i] > 20 or hour[i] < 6):
                y_pred_symbolic[i] += 0.3 * abs(delta[i])
        elif col == 'Appliance9':  # Chauffe-eau
            if abs(delta[i]) > 500 and (hour[i] < 6 or hour[i] > 22):
                y_pred_symbolic[i] += 0.4 * abs(delta[i])
        elif col == 'Appliance5':  # Bouilloire
            if abs(delta[i]) > 800 and y_pred_rf[i] < 100:
                y_pred_symbolic[i] += 0.5 * abs(delta[i])
        elif col == 'Appliance4':  # TV/Micro-ondes
            if 600 < abs(delta[i]) < 1000 and y_pred_rf[i] < 50:
                y_pred_symbolic[i] += 0.3 * abs(delta[i])
        
        # Clipping
        if y_pred_symbolic[i] < 0:
            y_pred_symbolic[i] = 0
        if y_pred_symbolic[i] > y.max() * 1.2:
            y_pred_symbolic[i] = y.max() * 1.2

    # ---- Métriques ----
    def calc_metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        return mae, rmse, r2

    mae_rf, rmse_rf, r2_rf = calc_metrics(y, y_pred_rf)
    mae_sym, rmse_sym, r2_sym = calc_metrics(y, y_pred_symbolic)

    results.append({
        'Appliance': col,
        'Name': appliance_names[col],
        'RF MAE': round(mae_rf, 2),
        'Sym MAE': round(mae_sym, 2),
        'RF RMSE': round(rmse_rf, 2),
        'Sym RMSE': round(rmse_sym, 2),
        'RF R2': round(r2_rf, 4),
        'Sym R2': round(r2_sym, 4),
        'Gain MAE': round(mae_rf - mae_sym, 2)
    })

    print(f" RF={mae_rf:.2f}W | Sym={mae_sym:.2f}W | Gain={mae_rf - mae_sym:.2f}W")

    # ---- Graphique pour cet appareil ----
    plt.figure(figsize=(12, 5))
    plt.plot(y[:500], label='Réel', linewidth=1.5, color='blue', alpha=0.7)
    plt.plot(y_pred_rf[:500], label='Random Forest', linestyle='--', linewidth=1.5, color='orange', alpha=0.7)
    plt.plot(y_pred_symbolic[:500], label='Soft-Boost Symbolique', linestyle=':', linewidth=2, color='green')
    plt.title(f"{appliance_names[col]} ({col}) - Prédictions (500 premiers échantillons)")
    plt.xlabel("Temps (échantillons)")
    plt.ylabel("Puissance (W)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # ---- Sauvegarde sécurisée (remplacement des caractères problématiques) ----
    safe_name = f"{col}_{appliance_names[col]}_predictions.png".replace('/', '_')
    filename = f"figures/{safe_name}"
    plt.savefig(filename, dpi=150)
    plt.close()

# ---- 5. Tableau récapitulatif ----
df_results = pd.DataFrame(results)
print("\n" + "=" * 100)
print("📊 RÉSULTATS DE L'EXPÉRIENCE 3 : SOFT-BOOST INTELLIGENT")
print("=" * 100)
print(df_results.to_string(index=False))
print("=" * 100)

# ---- 6. Sauvegarde du tableau ----
os.makedirs('results', exist_ok=True)
df_results.to_csv('results/experience3_softboost_intelligent.csv', index=False)

# ---- 7. Graphique comparatif global (tous les appareils) ----
plt.figure(figsize=(12, 6))
x = np.arange(len(df_results))
width = 0.35

plt.bar(x - width/2, df_results['RF MAE'], width, label='Random Forest', color='orange')
plt.bar(x + width/2, df_results['Sym MAE'], width, label='Soft-Boost Symbolique', color='green')

plt.xticks(x, df_results['Name'], rotation=45, ha='right')
plt.ylabel('MAE (W)')
plt.title('Comparaison des performances : Random Forest vs Soft-Boost Symbolique')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('figures/comparaison_globale_MAE.png', dpi=150)
plt.close()

print("\n✅ Graphiques sauvegardés dans le dossier 'figures/'")
print("✅ Résultats sauvegardés dans : results/experience3_softboost_intelligent.csv")







# ============================================================
# EXPERIENCE 4 : 2 000 000 LIGNES + GRAPHIQUES + DELESTAGE
# Random Forest + Soft-Boost intelligent + Fonction S
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time

# ---- Dossiers ----
os.makedirs('figures', exist_ok=True)
os.makedirs('results', exist_ok=True)

# ---- 1. Dossier de travail ----
project_path = 'F:/experience-ia-symbolique/nilm-hybrid-refit'
if os.path.exists(project_path):
    os.chdir(project_path)
else:
    print("⚠️ Chemin introuvable.")
    exit()

# ---- 2. Chargement du dataset (2 millions de lignes) ----
print("📂 Chargement du dataset REFIT (2 000 000 lignes)...")
df = pd.read_csv('data/raw/CLEAN_House1.csv', nrows=2000000)
print(f"✅ {len(df)} lignes chargées.")

# ---- 3. Feature Engineering ----
print("🔄 Construction des features...")
df['Time'] = pd.to_datetime(df['Time'])
df['Hour'] = df['Time'].dt.hour
df['Delta_P'] = df['Aggregate'] - df['Aggregate'].shift(1)
df = df.dropna()
print(f"✅ {len(df)} lignes après nettoyage.\n")

X = df[['Aggregate', 'Delta_P', 'Hour']].values

# ---- 4. Appareils ----
appliance_cols = [f'Appliance{i}' for i in range(1, 10)]
appliance_names = {
    'Appliance1': 'Refrigerator',
    'Appliance2': 'Chest Freezer',
    'Appliance3': 'Upright Freezer',
    'Appliance4': 'TV_Microwave',
    'Appliance5': 'Kettle',
    'Appliance6': 'Washing_Machine',
    'Appliance7': 'Lighting',
    'Appliance8': 'Set-top_Box',
    'Appliance9': 'Water_Heater'
}

results = []
print("🚀 Entraînement sur 2 000 000 lignes...\n")
start_time = time.time()

for col in appliance_cols:
    print(f"   → {col} ({appliance_names[col]})...", end='', flush=True)
   
    y = df[col].values

    # ---- Modèle RF ----
    rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    y_pred_rf = rf.predict(X)

    # ---- Soft-Boost intelligent (identique à Exp3) ----
    y_pred_symbolic = y_pred_rf.copy()
    delta = df['Delta_P'].values
    hour = df['Hour'].values
    agg = df['Aggregate'].values

    for i in range(len(y_pred_symbolic)):
        if col == 'Appliance6':  # Lave-linge
            if 300 < abs(delta[i]) < 800 and (hour[i] > 20 or hour[i] < 6):
                y_pred_symbolic[i] += 0.3 * abs(delta[i])
        elif col == 'Appliance9':  # Chauffe-eau
            if abs(delta[i]) > 500 and (hour[i] < 6 or hour[i] > 22):
                y_pred_symbolic[i] += 0.4 * abs(delta[i])
        elif col == 'Appliance5':  # Bouilloire
            if abs(delta[i]) > 800 and y_pred_rf[i] < 100:
                y_pred_symbolic[i] += 0.5 * abs(delta[i])
        elif col == 'Appliance4':  # TV/Micro-ondes
            if 600 < abs(delta[i]) < 1000 and y_pred_rf[i] < 50:
                y_pred_symbolic[i] += 0.3 * abs(delta[i])

        if y_pred_symbolic[i] < 0:
            y_pred_symbolic[i] = 0

    # ---- Métriques ----
    def calc_metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        return mae, rmse, r2

    mae_rf, rmse_rf, r2_rf = calc_metrics(y, y_pred_rf)
    mae_sym, rmse_sym, r2_sym = calc_metrics(y, y_pred_symbolic)

    results.append({
        'Appliance': col,
        'Name': appliance_names[col],
        'RF MAE': round(mae_rf, 2),
        'Sym MAE': round(mae_sym, 2),
        'RF RMSE': round(rmse_rf, 2),
        'Sym RMSE': round(rmse_sym, 2),
        'RF R2': round(r2_rf, 4),
        'Sym R2': round(r2_sym, 4),
        'Gain MAE': round(mae_rf - mae_sym, 2)
    })

    print(f" MAE={mae_rf:.2f}W | R²={r2_rf:.4f}")

# ---- Affichage du tableau ----
df_results = pd.DataFrame(results)
print("\n" + "=" * 100)
print("📊 RÉSULTATS SUR 2 000 000 LIGNES")
print("=" * 100)
print(df_results.to_string(index=False))
print("=" * 100)

# ---- Sauvegarde ----
df_results.to_csv('results/experience4_2M_results.csv', index=False)

# ---- Graphiques (500 premiers échantillons) ----
print("\n📈 Génération des graphiques...")
for col in appliance_cols:
    y = df[col].values[:500]
    y_pred_rf_sample = rf.predict(X[:500])
    y_pred_sym_sample = y_pred_symbolic[:500]

    plt.figure(figsize=(12, 5))
    plt.plot(y, label='Réel', linewidth=1.5, color='blue', alpha=0.7)
    plt.plot(y_pred_rf_sample, label='Random Forest', linestyle='--', linewidth=1.5, color='orange', alpha=0.7)
    plt.plot(y_pred_sym_sample, label='Soft-Boost', linestyle=':', linewidth=2, color='green')
    plt.title(f"{appliance_names[col]} - Prédictions (2M lignes)")
    plt.xlabel("Temps (échantillons)")
    plt.ylabel("Puissance (W)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"figures/2M_{col}_{appliance_names[col]}_predictions.png", dpi=150)
    plt.close()

print("✅ Graphiques sauvegardés.")

# ============================================================
# PHASE 2 : SIMULATION DE LA RÈGLE DE DÉLESTAGE (FONCTION S)
# ============================================================
print("\n⚡ SIMULATION DE LA FONCTION DE DÉLESTAGE S")

# On prend un échantillon réduit pour la simulation
df_sample = df.head(5000).copy()
X_sample = X[:5000]
y_pred_rf_sample = rf.predict(X_sample)

# ---- Paramètres de la fonction S ----
Pmax = 2200  # Puissance max souscrite (10A)
lambda_penalty = 100  # Pénalité de dépassement
price = 1.0  # Prix du kWh (1 = heure pleine)

# ---- Simulation du délestage ----
print("   → Calcul de la fonction S sur 5000 échantillons...")

# On simule un appareil à couper (ex: Chauffe-eau = Appliance9)
y_pred_wh = rf.predict(df[['Aggregate', 'Delta_P', 'Hour']].values[:5000])

# Fonction S simplifiée
def compute_S(power, price, Pmax, lambda_penalty):
    S = price * power
    exceed = max(0, power - Pmax)
    S += lambda_penalty * exceed ** 2
    return S

scores = np.array([compute_S(p, price, Pmax, lambda_penalty) for p in y_pred_wh])
scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)

# ---- Visualisation du délestage ----
plt.figure(figsize=(12, 6))
plt.plot(y_pred_wh[:200], label='Puissance prédite (Chauffe-eau)', linewidth=1.5, color='blue')
plt.plot(scores_norm[:200] * 500, label='Score S (normalisé)', linestyle='--', linewidth=1.5, color='red')
plt.axhline(y=Pmax, color='black', linestyle=':', label='Pmax (2200W)')
plt.title("Simulation de la fonction de délestage S sur le chauffe-eau")
plt.xlabel("Temps (échantillons)")
plt.ylabel("Puissance (W)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/delestage_simulation.png", dpi=150)
plt.close()

print("✅ Graphique de simulation du délestage sauvegardé.")

elapsed_time = time.time() - start_time
print(f"\n⏱️ Temps total d'exécution : {elapsed_time:.1f} secondes")
print("\n✅ Expérience 4 terminée avec succès !")