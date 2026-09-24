# NILM hybride et règles symboliques

[English version](../README.md) · **Version française**

> **Statut : recherche scientifique exploratoire en cours.** Ce dépôt documente des expériences de désagrégation non intrusive des charges électriques (NILM), des références physiques issues de PLAID et l’évaluation de règles symboliques. Il ne s’agit pas d’un produit, d’un contrôleur validé sur le terrain ou d’un projet de didactisation.

## Objet scientifique

Le projet étudie si la puissance active agrégée d’un logement peut servir à estimer la consommation de certains appareils. Un modèle statistique constitue la référence. Des règles symboliques interprétables sont ensuite étudiées comme couche de correction éventuelle.

Le délestage intelligent est un **contexte d’application futur**. Le dépôt actuel ne démontre pas la commande sûre et autonome d’appareils reliés au secteur. Les données principales proviennent de foyers britanniques du dataset REFIT ; elles ne prouvent donc pas la transférabilité automatique vers des foyers béninois.

## Avancée actuelle

| Élément | État |
|---|---|
| Exploration de REFIT House 1 | Base exploratoire disponible |
| Modèles linéaires et Random Forest | Résultats exploratoires stockés |
| Comparaison temporelle | Diagnostic initial disponible |
| Références PLAID par régime | Pipeline implémenté pour certains appareils |
| Règles symboliques | Module centralisé ; règles actives neutralisées après dégradation observée |
| Fusion REFIT–iAWE | Pipeline corrigé pour trois appareils communs |
| Validation locale au Bénin | Non réalisée |
| Benchmark final reproductible | À finaliser |

Les résultats conservés montrent une conclusion importante : les règles Soft-Boost n’améliorent pas nécessairement tous les appareils. Elles doivent donc être évaluées avec le même découpage temporel et le même jeu de test que le modèle de référence.

## Organisation

```text
.
├── docs/                         Rapports et méthodologie
│   ├── research_methodology.md  Protocole et limites scientifiques
│   └── session-reports/          Rapports de sessions
├── figures/                      Figures sélectionnées
├── results/                      Résumés CSV et résultats
├── src/pipeline/                 Scripts actuels de traitement
├── src/regles_symboliques.py     Module des règles interprétables
├── src/appliance_mapping.py      Mapping centralisé des appareils
├── src/legacy/                   Anciens scripts exploratoires
└── tests/                        Tests du module symbolique
```

Les datasets bruts REFIT, iAWE et PLAID ne sont pas stockés dans Git. Ils doivent être obtenus depuis leurs sources officielles et leurs licences doivent être respectées.

## Documentation scientifique

- [Méthodologie et périmètre scientifique](research_methodology.md)
- [Rapport Markdown des sessions des 23 et 24 septembre 2026](session-reports/rapport_sessions_23_24_septembre_2026.md)
- [Rapport de recherche en français au format PDF](Rapport%20de%20Recherche%20Delesteur%20Intelligent%202026.pdf)
- [Research report in English au format PDF](Intelligent%20Load%20Shedder_Research%20Report%202026.pdf)

## Résultats et limites

Les fichiers de `results/` proviennent d’expériences différentes. Ils ne doivent pas être combinés en un benchmark unique. Les valeurs présentes sont des sorties exploratoires et ne démontrent ni une précision sur le terrain, ni une transférabilité vers le Bénin, ni une sécurité de commande.

La prochaine étape principale est de fixer un protocole chronologique entraînement/validation/test, d’enregistrer les versions de données et de comparer le modèle linéaire, le Random Forest et une version symbolique sur exactement le même test.

## Sécurité

> Ce dépôt n’est pas une notice de câblage. Aucune connexion expérimentale au secteur ne doit être réalisée sans isolation adaptée, protections correctement dimensionnées, supervision qualifiée et respect des règles électriques locales.

## Références

[1]: https://doi.org/10.5281/zenodo.5063428 "REFIT: Electrical Load Measurements (Cleaned)"

[2]: https://doi.org/10.1038/sdata.2016.122 "REFIT electrical load measurements dataset descriptor"

[1] [2]
