# NILM hybride et gestion priorisée des charges

[English version](../README.md) · **Version française**

> **Statut : recherche exploratoire en cours.** Ce dépôt documente un projet sur la désagrégation non intrusive des charges électriques (NILM), les règles symboliques et les concepts de gestion priorisée de la puissance. Il ne s’agit pas d’un contrôleur prêt à l’emploi, d’un dispositif certifié ou d’une validation de terrain au Bénin.

## Avancée actuelle

Le projet dispose maintenant d’un pipeline organisé autour de l’exploration de REFIT, de références physiques issues de PLAID, d’une fusion expérimentale REFIT–iAWE et d’un module centralisé de règles symboliques. Les résultats enregistrés montrent une conclusion importante : les règles Soft-Boost n’améliorent pas nécessairement tous les appareils. Elles doivent donc être évaluées sur un découpage temporel propre avant toute conclusion définitive.

Le dépôt sert aussi de base de recherche pour un futur **banc didactique basse tension**. Ce banc sera traité séparément et représentera les appareils par de petites charges équivalentes afin de rendre les essais transportables et sûrs.

## Organisation

```text
.
├── docs/                         Rapports et documentation
│   └── session-reports/          Rapports de sessions de travail
├── figures/                      Figures sélectionnées
├── results/                      Résumés CSV et résultats
├── src/pipeline/                 Scripts actuels de traitement
├── src/regles_symboliques.py     Module des règles interprétables
├── src/appliance_mapping.py      Mapping centralisé des appareils
└── src/legacy/                   Anciens scripts exploratoires
```

Les jeux de données bruts REFIT, iAWE et PLAID ne sont pas stockés dans Git. Ils doivent être obtenus depuis leurs sources officielles, puis placés dans `data/raw/` conformément aux conditions de leurs licences.

## Résultats et prudence d’interprétation

Les fichiers de `results/` proviennent d’expériences différentes. Ils ne doivent pas être combinés en un classement unique. Les valeurs présentes sont des sorties exploratoires. Elles ne démontrent pas une précision sur le terrain, une transférabilité automatique vers les foyers béninois ou une sécurité de commande sur le secteur.

## Rapports

- [Rapport de recherche en français](Rapport%20de%20Recherche%20Delesteur%20Intelligent%202026.pdf)
- [Research report in English](Intelligent%20Load%20Shedder_Research%20Report%202026.pdf)
- [Rapports de sessions](session-reports/)

## Sécurité

> Ne reliez jamais un montage expérimental au secteur sans isolation adaptée, protections correctement dimensionnées, supervision qualifiée et respect des règles électriques locales. Ce dépôt n’est pas une notice de câblage.

## Prochaines étapes

La prochaine étape scientifique est de terminer le benchmark avec une séparation chronologique entraînement/validation/test, une configuration enregistrée et des métriques comparables. La prochaine étape pédagogique est de construire un banc basse tension compact pour illustrer la mesure de puissance, la hiérarchisation des charges et la décision de gestion.

Pour les données et les références scientifiques, voir la [documentation anglaise principale](../README.md).
