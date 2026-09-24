# Rapport de sessions — construction et validation du pipeline hybride NILM

**Projet :** `nilm-hybrid-refit`  
**Sessions :** 23 et 24 septembre 2026  
**Étudiant :** SOSSOU BIADJA Charbel Osnyl  
**Formation :** ENSET Lokossa — génie électrotechnique

## Objet du rapport

Ces deux sessions ont porté sur la construction, la correction et la première validation d’un pipeline de recherche hybride combinant REFIT, iAWE et PLAID. Le travail concerne exclusivement l’expérimentation scientifique NILM et les règles symboliques associées.

## Session du 23 septembre — données et règles symboliques

### Point de départ

Le projet disposait d’un dataset REFIT–iAWE fusionné d’environ 24,6 millions de lignes pour trois appareils communs : réfrigérateur, machine à laver et télévision. Un modèle Random Forest existait déjà, mais les seuils symboliques étaient estimés approximativement. PLAID avait été téléchargé pour fournir des références de puissance et de courant d’appel.

Trois difficultés principales ont été identifiées : les seuils n’étaient pas calibrés sur une référence physique, le dataset était trop lourd pour une machine disposant de 8 Go de RAM, et le nombre d’appareils avait été volontairement limité pour garantir la cohérence entre les sources.

### Mémoire et traitement par blocs

Le chargement complet du dataset provoquait des plantages lorsque Spyder ou Jupyter fonctionnaient simultanément. Le traitement a donc été déplacé vers le terminal, avec lecture par morceaux (`chunksize = 500000`) et libération explicite de la mémoire avec `gc.collect()`.

### Correction du mapping REFIT

Le mapping supposé initialement attribuait incorrectement certains canaux à des appareils. La correspondance a été vérifiée à partir du fichier officiel `MetaData_Tables.xlsx` fourni par les créateurs de REFIT. Le mapping corrigé est centralisé dans `src/appliance_mapping.py` afin d’éviter des définitions contradictoires entre scripts.

### Réduction du dataset

Le dataset fusionné contenait beaucoup de motifs redondants. Un échantillonnage par appareil et par source a été retenu afin de réduire la consommation mémoire. La méthode doit toutefois être distinguée de l’évaluation : le nombre d’échantillons, la graine et la séparation temporelle doivent être documentés pour chaque benchmark.

### Références PLAID par régime

Les régimes de fonctionnement peuvent être hétérogènes, notamment pour les appareils à compresseur. Les références PLAID sont donc calculées par type d’appareil et par régime lorsque les métadonnées le permettent. PLAID sert de référence physique externe ; il n’est pas aligné ligne par ligne sur REFIT ou iAWE.

### Résultat de la session

Le module `src/regles_symboliques.py` centralise les références et la fonction `appliquer_regles()`. Les règles actives ont ensuite été neutralisées lorsque les expériences ont montré qu’elles dégradaient les performances du Random Forest.

## Session du 24 septembre — correction et validation initiale

### Dataset mal structuré

Un résultat nul ou quasi nul pour la machine à laver a révélé une erreur de construction du dataset. L’ancienne fusion créait plusieurs colonnes cibles et remplissait implicitement les colonnes non concernées par zéro. Le script `src/pipeline/09_merge_refit_iawe_v3.py` produit désormais une colonne unique `Puissance_Cible`, associée à `Appliance_Target`.

Le dataset corrigé a atteint environ 24 675 114 lignes dans l’expérience décrite par le rapport de session. Ce chiffre reste un résultat de session et doit être recalculé avec la configuration versionnée avant d’être présenté comme une sortie officielle reproductible.

### Échantillonnage et cycles rares

La prise des premières lignes du dataset manquait des cycles rares de la machine à laver. L’échantillonnage a été modifié pour couvrir l’ensemble du dataset avec `sample(..., random_state=42)`. Le rapport de session signale alors des valeurs non nulles pour les trois appareils étudiés.

Cette correction élimine un biais évident, mais elle ne remplace pas une évaluation temporelle propre. Un échantillon aléatoire ne doit pas mélanger les périodes qui doivent rester séparées dans le test final.

### Règles symboliques dégradantes

Les règles de type `abs(delta) > 150 and pred < 50` avaient été conçues pour un ancien modèle linéaire. Avec le Random Forest, le boost pouvait dégrader la MAE, notamment pour le réfrigérateur et la machine à laver. La décision actuelle est donc de laisser `REGLES_PAR_APPAREIL = {}` tant qu’une nouvelle calibration n’a pas été définie et évaluée séparément.

Cette neutralité est un résultat méthodologique : elle signifie que le dépôt ne revendique pas encore un gain du modèle hybride.

### Résultats enregistrés dans le rapport

| Appareil | MAE (W) | RMSE (W) | R² |
|---|---:|---:|---:|
| Fridge | 25,48 | 45,34 | −0,018 |
| WashingMachine | 14,39 | 124,49 | 0,384 |
| Television | 7,55 | 13,07 | 0,038 |

Ces valeurs sont conservées comme résultats exploratoires d’une configuration donnée. Elles ne doivent pas être combinées avec les résultats d’autres expériences sans vérifier les données, les variables et le protocole de séparation.

### Enseignements

Un MAE nul est un signal d’alarme lorsqu’il n’est pas physiquement plausible. Le mapping doit provenir de la métadonnée primaire. Les cycles rares exigent une stratégie d’échantillonnage documentée. Enfin, une règle symbolique doit être calibrée pour le modèle cible et validée sur une période indépendante.

## Bilan et prochaines étapes scientifiques

Les sessions ont produit une structure de données corrigée, un mapping centralisé, des références PLAID par régime et une première comparaison entre Random Forest et règles symboliques. Elles n’ont pas encore produit un benchmark définitif.

Les prochaines étapes sont :

1. fixer un découpage chronologique entraînement/validation/test ;
2. enregistrer la version des datasets et les paramètres de prétraitement ;
3. comparer le modèle linéaire, le Random Forest et une future version symbolique sur exactement le même test ;
4. ajouter des tests automatisés sur le module de règles ;
5. étudier les variables temporelles et les retards sans utiliser d’information future ;
6. caractériser séparément le cas difficile du réfrigérateur ;
7. rapporter les résultats avec leurs limites et sans extrapolation automatique vers le Bénin.

## Références

[1]: https://doi.org/10.5281/zenodo.5063428 "REFIT: Electrical Load Measurements (Cleaned)"

[2]: https://doi.org/10.1038/sdata.2016.122 "REFIT electrical load measurements dataset descriptor"

[1] [2]
