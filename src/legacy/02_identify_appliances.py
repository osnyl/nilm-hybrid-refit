import pandas as pd
import numpy as np

# Charger les données
print("📂 Chargement du fichier...")
df = pd.read_csv('data/raw/CLEAN_House1.csv')
print(f"✅ {len(df)} lignes chargées.\n")

# On va analyser les colonnes Appliance1 à Appliance9
appliance_cols = [f'Appliance{i}' for i in range(1, 10)]

print("🔍 ANALYSE DES SIGNATURES DES APPAREILS")
print("=" * 60)

# Dictionnaire pour stocker les infos
results = []

for col in appliance_cols:
    data = df[col]
    
    # 1. On filtre les valeurs > 10W pour ignorer le bruit (quand l'appareil est éteint)
    on_data = data[data > 10]
    
    if len(on_data) == 0:
        print(f"⚠️ {col} : Jamais allumé (ou bruit négligeable)")
        continue
    
    # 2. Puissance moyenne quand il est allumé
    mean_power = on_data.mean()
    max_power = on_data.max()
    
    # 3. Durée de fonctionnement (approximative)
    # On regarde les blocs où l'appareil est allumé
    is_on = data > 10
    # Détection des changements (début/fin de cycle)
    changes = is_on.astype(int).diff()
    # Nombre de fois où il s'allume (débuts de cycle)
    nb_starts = (changes == 1).sum()
    # Durée totale allumé (en secondes) * intervalle de 15s
    total_seconds_on = is_on.sum() * 15
    # Durée moyenne d'un cycle (en minutes)
    if nb_starts > 0:
        avg_cycle_min = (total_seconds_on / nb_starts) / 60
    else:
        avg_cycle_min = 0
    
    # Calcul du nombre de cycles par jour (sur la durée totale du dataset)
    total_days = len(df) * 15 / (24 * 3600)  # Durée totale en jours
    cycles_per_day = nb_starts / total_days if total_days > 0 else 0

    # 4. Heure de pointe (quand il consomme le plus)
    # On récupère les heures des moments où l'appareil est allumé
    hours = pd.to_datetime(df['Time']).dt.hour
    peak_hour = hours[on_data.index].mode().iloc[0] if len(on_data) > 0 else 0
    
    # 5. Classification automatique basée sur les règles apprises
    if mean_power < 30:
        type_guess = "💡 Lampe / Électronique"
    elif 30 <= mean_power < 200:
        type_guess = "🧊 Réfrigérateur / Petit électro"
    elif 200 <= mean_power < 1000:
        type_guess = "📺 TV / Micro-ondes / Ordinateur"
    elif 1000 <= mean_power < 3000:
        type_guess = "🔥 Chauffe-eau / Four / Lave-linge"
    else:
        type_guess = "⚡ Gros moteur (Climatisation / Pompe)"
    
    # Stockage
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

# Affichage du tableau
print("\n📊 RÉSULTATS DE L'ANALYSE")
print("=" * 80)
for r in results:
    print(f"{r['Appareil']} :")
    print(f"  ⚡ Puissance : {r['Puissance_moyenne(W)']}W (max {r['Puissance_max(W)']}W)")
    print(f"  🔄 Cycles par jour : {r['Cycles_par_jour']} (Durée moyenne : {r['Durée_moyenne_cycle(min)']} min)")
    print(f"  🕒 Heure de pointe : {r['Heure_de_pointe']}")
    print(f"  🏷️  Classification : {r['Classification']}")
    print("-" * 40)

print("\n🎉 Analyse terminée !")