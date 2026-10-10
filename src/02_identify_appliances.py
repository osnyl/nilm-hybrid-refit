import pandas as pd
import numpy as np


print(" Chargement du fichier...")
df = pd.read_csv('data/raw/CLEAN_House1.csv')
print(f" {len(df)} lignes chargées.\n")


appliance_cols = [f'Appliance{i}' for i in range(1, 10)]

print(" ANALYSE DES SIGNATURES DES APPAREILS")
print("=" * 60)


results = []

for col in appliance_cols:
    data = df[col]


    on_data = data[data > 10]

    if len(on_data) == 0:
        print(f" {col} : Jamais allumé (ou bruit négligeable)")
        continue


    mean_power = on_data.mean()
    max_power = on_data.max()



    is_on = data > 10

    changes = is_on.astype(int).diff()

    nb_starts = (changes == 1).sum()

    total_seconds_on = is_on.sum() * 15

    if nb_starts > 0:
        avg_cycle_min = (total_seconds_on / nb_starts) / 60
    else:
        avg_cycle_min = 0


    total_days = len(df) * 15 / (24 * 3600)
    cycles_per_day = nb_starts / total_days if total_days > 0 else 0



    hours = pd.to_datetime(df['Time']).dt.hour
    peak_hour = hours[on_data.index].mode().iloc[0] if len(on_data) > 0 else 0


    if mean_power < 30:
        type_guess = " Lampe / Électronique"
    elif 30 <= mean_power < 200:
        type_guess = " Réfrigérateur / Petit électro"
    elif 200 <= mean_power < 1000:
        type_guess = " TV / Micro-ondes / Ordinateur"
    elif 1000 <= mean_power < 3000:
        type_guess = " Chauffe-eau / Four / Lave-linge"
    else:
        type_guess = " Gros moteur (Climatisation / Pompe)"


    results.append({
        'Appareil': col,
        'Puissance_moyenne(W)': round(mean_power, 1),
        'Puissance_max(W)': round(max_power, 1),
        'Nb_allumages_total': nb_starts,
        'Cycles_par_jour': round(cycles_per_day, 2),
        'Durée_moyenne_cycle(min)': round(avg_cycle_min, 1),
        'Heure_de_pointe': f"{int(peak_hour)}h",
        'Classification': type_guess
    })


print("\n RÉSULTATS DE L'ANALYSE")
print("=" * 80)
for r in results:
    print(f"{r['Appareil']} :")
    print(f"   Puissance : {r['Puissance_moyenne(W)']}W (max {r['Puissance_max(W)']}W)")
    print(f"   Cycles par jour : {r['Cycles_par_jour']} (Durée moyenne : {r['Durée_moyenne_cycle(min)']} min)")
    print(f"   Heure de pointe : {r['Heure_de_pointe']}")
    print(f"    Classification : {r['Classification']}")
    print("-" * 40)

print("\n Analyse terminée !")
