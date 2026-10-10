















import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error









try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()




if os.path.basename(SCRIPT_DIR) == "src":
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
else:
    PROJECT_ROOT = SCRIPT_DIR

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "CLEAN_House1.csv")



if not os.path.exists(DATA_PATH):
    candidats = [
        os.path.join(os.getcwd(), "data", "raw", "CLEAN_House1.csv"),
        os.path.join(os.getcwd(), "CLEAN_House1.csv"),
        os.path.join(os.getcwd(), "..", "data", "raw", "CLEAN_House1.csv"),
    ]
    for c in candidats:
        if os.path.exists(c):
            DATA_PATH = os.path.abspath(c)
            break

os.makedirs(os.path.join(PROJECT_ROOT, "results"), exist_ok=True)





N_ROWS = 500_000

print(f" Dossier courant (cwd) : {os.getcwd()}")
print(f" Chemin résolu pour le fichier : {DATA_PATH}")
if not os.path.exists(DATA_PATH):
    print(" Fichier introuvable après toutes les pistes de secours.")
    print("   Vérifie le chemin exact de CLEAN_House1.csv et adapte DATA_PATH")
    print("   directement en dur juste ici si besoin, ex :")
    print('   DATA_PATH = r"C:/chemin/complet/vers/CLEAN_House1.csv"')
    raise SystemExit(1)

df = pd.read_csv(DATA_PATH, nrows=N_ROWS)
print(f" {len(df)} lignes chargées.")





df["Time"] = pd.to_datetime(df["Time"])
df = df.sort_values("Time").reset_index(drop=True)


df["Hour"] = df["Time"].dt.hour
df["Delta_P"] = df["Aggregate"] - df["Aggregate"].shift(1)
df = df.dropna().reset_index(drop=True)
print(f" {len(df)} lignes après feature engineering et nettoyage.")

X_all = df[["Aggregate", "Delta_P", "Hour"]].values




split_idx = int(len(df) * 0.8)
train_end_time = df["Time"].iloc[split_idx]

print(f"\n Split chronologique à l'index {split_idx} / {len(df)}")
print(f"   Entraînement : du {df['Time'].iloc[0]} au {df['Time'].iloc[split_idx-1]}")
print(f"   Test (jamais vu) : du {df['Time'].iloc[split_idx]} au {df['Time'].iloc[-1]}\n")

X_train, X_test = X_all[:split_idx], X_all[split_idx:]


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



print(" Entraînement + évaluation honnête sur les 9 appareils...\n")
results = []

for col in appliance_cols:
    print(f"   → {col} ({appliance_names[col]})...", end="", flush=True)

    y_train = df[col].values[:split_idx]
    y_test = df[col].values[split_idx:]

    rf = RandomForestRegressor(
        n_estimators=50, max_depth=10, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)


    pred_train = rf.predict(X_train)
    mae_train, rmse_train, r2_train = calc_metrics(y_train, pred_train)


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


df_results = pd.DataFrame(results)
print("\n" + "=" * 100)
print(" COMPARAISON HONNÊTE : ÉVALUATION EN TRICHANT vs VRAIE ÉVALUATION")
print("=" * 100)
print(df_results.to_string(index=False))
print("=" * 100)

out_path = os.path.join(PROJECT_ROOT, "results", "eval_train_vs_test.csv")
df_results.to_csv(out_path, index=False)
print(f"\n Résultats sauvegardés dans : {out_path}")
print(
    "\nLecture : plus 'Écart R² (optimisme)' est grand, plus les chiffres du "
    "rapport original étaient optimistes par rapport à la réalité."
)
