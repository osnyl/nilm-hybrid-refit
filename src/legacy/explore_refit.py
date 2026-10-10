import pandas as pd


print(" Chargement du fichier CLEAN_House1.csv...")
df = pd.read_csv('data/raw/CLEAN_House1.csv')


print("\n Les 5 premières lignes :")
print(df.head())


print("\n Colonnes disponibles :")
print(df.columns.tolist())


print(f"\n Taille du dataset : {df.shape} (lignes, colonnes)")


print("\n Statistiques de la consommation totale (Aggregate) :")
print(df['Aggregate'].describe())


print("\n Valeurs manquantes par colonne :")
print(df.isnull().sum())

print("\n Exploration terminée !")
