import pandas as pd

# 1. Charger le fichier (le chemin est relatif au dossier où on est)
print("📂 Chargement du fichier CLEAN_House1.csv...")
df = pd.read_csv('data/raw/CLEAN_House1.csv')

# 2. Afficher les 5 premières lignes
print("\n🔍 Les 5 premières lignes :")
print(df.head())

# 3. Afficher les noms des colonnes
print("\n📋 Colonnes disponibles :")
print(df.columns.tolist())

# 4. Afficher le nombre de lignes et de colonnes
print(f"\n📊 Taille du dataset : {df.shape} (lignes, colonnes)")

# 5. Statistiques de base sur la consommation totale
print("\n📈 Statistiques de la consommation totale (Aggregate) :")
print(df['Aggregate'].describe())

# 6. Vérifier s'il y a des valeurs manquantes
print("\n⚠️ Valeurs manquantes par colonne :")
print(df.isnull().sum())

print("\n✅ Exploration terminée !")