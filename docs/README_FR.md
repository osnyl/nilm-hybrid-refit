# NILM hybride pour un délestage intelligent

[English version](../README.md) · **Version française**

> **Statut de recherche — exploratoire.** Ce dépôt documente un projet de recherche et développement sur la désagrégation non intrusive des charges électriques (NILM) et sur des concepts de délestage par priorité destinés aux foyers à faibles ressources. Il ne s’agit **ni** d’un dispositif électrique prêt à la vente, **ni** d’un équipement certifié, **ni** d’un système déjà validé sur le terrain.

## Présentation du projet

Ce projet étudie la possibilité d’utiliser la **puissance active totale** d’un logement afin d’estimer la consommation de certains appareils et d’éclairer, à terme, un système de délestage intelligent. Les expériences d’apprentissage automatique utilisent le jeu de données public REFIT. Les rapports associés décrivent également des scénarios simulés de délestage sous Proteus.

L’idée de recherche est hybride. Une couche statistique estime la consommation d’appareils à partir du signal agrégé, tandis que des règles symboliques simples et interprétables sont testées comme couche de correction. L’objectif à long terme est d’aider un foyer à préserver les charges les plus importantes lorsque la puissance disponible est limitée. Le contenu public actuel constitue une **base de recherche**, non un contrôleur opérationnel.

| Élément | Portée actuelle |
|---|---|
| Donnée d’entrée | Puissance active agrégée du logement dans REFIT House 1 |
| Appareils étudiés | Neuf canaux d’appareils étiquetés de REFIT House 1 |
| Méthodes | Régression linéaire, expériences Random Forest et règles exploratoires Soft-Boost |
| Lien avec le délestage | Concept de recherche et discussion de simulations ; contrôleur déployable absent de ce dépôt |
| Contexte visé | Foyers à faibles ressources ; transfert vers des foyers béninois non encore validé |

## Contenu du dépôt

Le dépôt contient des scripts Python d’exploration de données, d’analyse des canaux d’appareils, d’expérimentation de modèles, de règles symboliques et de préparation d’un signal `load_data.txt` utilisable dans une simulation. Il contient aussi des figures et des résultats au format CSV. Le [rapport de recherche en français](Rapport%20de%20Recherche%20Delesteur%20Intelligent%202026.pdf) ainsi que le [research report in English](Intelligent%20Load%20Shedder_Research%20Report%202026.pdf) sont accessibles directement dans ce dossier `docs/`.

Le dépôt apporte des éléments sur le travail de modélisation exploratoire réalisé avec REFIT. Il ne fournit pas encore un paquet matériel entièrement reproductible : le snapshot public ne contient pas de projet source Proteus, de programme Arduino/ESP32, de schéma électronique complet, de circuit imprimé, de nomenclature de composants, de protocole de collecte de données de terrain ni de certification de sécurité électrique.

## Limites et position scientifique

Trois distinctions sont essentielles pour comprendre correctement ce travail.

D’abord, REFIT a été mesuré dans des foyers du Royaume-Uni. Ce jeu de données est utile pour développer et étudier des méthodes NILM, mais il ne prouve pas que le même modèle aura le même comportement dans des foyers béninois. Les appareils disponibles, les habitudes d’usage, les installations électriques, la qualité de tension et la composition des ménages peuvent différer.

Ensuite, les fichiers CSV présents dans `results/` conservent les sorties de plusieurs expériences. Les scripts actuels nécessitent encore un protocole clairement documenté de séparation chronologique entre entraînement, validation et test. Les valeurs présentes doivent donc être lues comme des **résultats exploratoires**, et non comme une preuve définitive de performance hors échantillon, de précision commerciale ou de performance sur le terrain.

Enfin, la fonction avancée à quatre termes parfois discutée dans les rapports — coût financier, confort de l’utilisateur, sécurité et usure des appareils — est une **perspective de recherche**. Les fichiers Python publics n’implémentent pas un optimiseur temps réel validé à partir de cette fonction. Le mécanisme de délestage qui devra être évalué dans un futur prototype est un contrôle par priorités des charges non essentielles.

> **Avertissement de sécurité :** ne reliez jamais directement au secteur un microcontrôleur, un relais, un capteur ou un programme expérimental sans l’intervention d’un électricien qualifié, une isolation adaptée, des protections électriques appropriées et le respect des règles locales. Ce dépôt ne constitue pas une notice de câblage.

## Jeu de données REFIT

Les expériences utilisent le jeu de données nettoyé **REFIT Electrical Load Measurements**, en particulier le fichier `CLEAN_House1.csv`. REFIT met à disposition des mesures de puissance active agrégée et de puissance par appareil provenant de 20 foyers britanniques, échantillonnées toutes les 8 secondes. Le jeu de données est diffusé sous licence CC BY 4.0 et doit être attribué correctement. [1] [2]

Les données brutes REFIT ne sont volontairement **pas stockées dans ce dépôt**. Il faut les télécharger depuis la source officielle puis placer le fichier requis ici :

```text
data/raw/CLEAN_House1.csv
```

Le fichier `.gitignore` exclut le CSV brut. Ce choix évite un dépôt trop lourd et limite la redistribution inutile de données tierces.

### Citation obligatoire de REFIT

Toute personne qui utilise ou prolonge ce travail doit citer le jeu de données et son article descriptif :

> Murray, D., Stankovic, L. et Stankovic, V. *An electrical load measurements dataset of United Kingdom households from a two-year longitudinal study.* Scientific Data 4, 160122 (2017). https://doi.org/10.1038/sdata.2016.122

## Organisation du dépôt

```text
.
├── docs/       Rapports de recherche en français et en anglais
├── figures/    Figures de prédiction et de simulation
├── results/    Résumés CSV et figures de résultats
├── src/        Scripts Python exploratoires
├── load_data.txt
├── requirements.txt
└── README.md
```

| Chemin | Fonction |
|---|---|
| `src/01_explore_refit.py` | Chargement et exploration descriptive de REFIT House 1 |
| `src/02_identify_appliances.py` | Analyse descriptive simple des canaux d’appareils |
| `src/03_train_models.py` | Expériences de modèles linéaires et Random Forest ; le fichier doit encore être rendu portable et mieux structuré pour l’évaluation |
| `src/04_symbolic_booster.py` | Expérience exploratoire de règles Soft-Boost sur certains canaux |
| `src/05_generate_load_file.py` | Conversion d’échantillons de puissance agrégée en un signal temps/courant pour simulation |
| `results/` | Résumés des expériences ; lire les limites ci-dessous avant de les utiliser comme preuve définitive |

## Résultats expérimentaux actuellement disponibles

Le fichier [`results/experience4_2M_results.csv`](../results/experience4_2M_results.csv) contient un résumé d’expérience sur deux millions de lignes. Les exemples ci-dessous montrent une leçon importante : dans cette expérience enregistrée, les règles Soft-Boost n’améliorent pas chaque appareil et dégradent légèrement certains résultats. C’est une information scientifique utile. Elle indique que la couche symbolique doit être validée avec méthode, et non présentée de façon optimiste sans vérification.

| Canal d’appareil | MAE Random Forest (W) | MAE Soft-Boost (W) | Lecture prudente |
|---|---:|---:|---|
| Réfrigérateur | 22,48 | 22,48 | Pas de changement mesuré dans cette expérience |
| Lave-linge | 7,78 | 7,91 | Légère dégradation dans cette expérience |
| Chauffe-eau | 28,78 | 31,83 | Dégradation dans cette expérience |

Le dépôt contient aussi des fichiers de résultats antérieurs avec d’autres tailles d’échantillon et d’autres méthodes. Il ne faut pas les rassembler comme s’ils formaient un seul classement. La suite du travail devra fixer la graine aléatoire, enregistrer le prétraitement, utiliser des découpages chronologiques entraînement/validation/test et rapporter les métriques sur une période de test non utilisée pendant l’entraînement.

## État de reproductibilité

Le dépôt est public, mais sa reproductibilité est actuellement **partielle**. Les améliorations suivantes sont nécessaires avant de présenter l’exécution complète comme reproductible par un tiers.

| Élément | État actuel | Amélioration nécessaire |
|---|---|---|
| Données brutes | Non suivies par Git, volontairement | Télécharger `CLEAN_House1.csv` depuis REFIT |
| Dépendances | `requirements.txt` est incomplet pour les scripts Random Forest | Ajouter `scikit-learn` et fixer les versions après un essai propre |
| Chemins de fichiers | Certains scripts utilisent un chemin personnel Windows | Le remplacer par un chemin relatif au projet |
| Évaluation | Les scripts nécessitent un protocole de test séparé | Mettre en place une séparation chronologique entraînement/validation/test |
| Simulation matérielle | Figures et `load_data.txt` publics, mais sources Proteus/Arduino absentes | Ajouter les fichiers légalement partageables ou documenter clairement leur absence |
| Sécurité | Pas de validation terrain ni de certification | Tester sous supervision avant toute connexion au secteur |

Pour créer un environnement Python local, le point de départ est :

```bash
python -m venv .venv
# Sous Windows : .venv\Scripts\activate
# Sous Linux/macOS : source .venv/bin/activate
pip install -r requirements.txt
pip install scikit-learn
```

Ensuite, téléchargez manuellement REFIT, placez `CLEAN_House1.csv` dans `data/raw/` et vérifiez les chemins de fichiers avant toute exécution. Le flux complet ne doit pas être présenté comme reproduit tant que le découpage de données et la configuration n’ont pas été corrigés.

## Lien avec le délestage intelligent

Les rapports associés discutent trois scénarios de foyers simulés ainsi qu’une architecture future de délestage intelligent. Le contenu public actuel doit être compris comme la **composante de recherche NILM** et une composante de préparation de données pour simulation. Il ne démontre pas encore un système sûr fonctionnant en temps réel, capable d’identifier n’importe quel appareil dans un foyer béninois et de commander de manière autonome des charges reliées au secteur.

Une feuille de route responsable est la suivante :

1. Valider le modèle de base avec un protocole temporel reproductible et une partie de test réellement séparée.
2. Mesurer, avec consentement, la puissance globale et quelques charges identifiées dans un petit pilote local.
3. Comparer le modèle entraîné sur REFIT avec les données locales sans supposer à l’avance qu’il se transférera bien.
4. Tester d’abord un contrôleur simple à priorités sur des conditions basse tension et contrôlées.
5. Associer un électricien qualifié avant toute installation reliée au secteur.
6. Mesurer les bénéfices pour l’utilisateur, la sécurité et les effets sur la consommation avant toute promesse commerciale.

## Comment citer ce dépôt

Aucun DOI de projet n’a encore été attribué. Après la création d’une archive Zenodo relue et publiée, le DOI pourra être ajouté ici ainsi que dans un fichier `CITATION.cff`.

En attendant, une citation provisoire est :

> SOSSOU BIADJA, Charbel (2026). *NILM hybride pour un délestage intelligent : expériences exploratoires avec REFIT et concepts de délestage.* Dépôt GitHub : https://github.com/Osnyl/nilm-hybrid-refit

## Licence

Aucune licence n’a encore été sélectionnée pour ce dépôt. Ce choix doit être fait consciemment avant le dépôt Zenodo.

| Choix possible | Conséquence principale |
|---|---|
| **MIT** | Licence de code courte et permissive ; autorise des réutilisations, y compris commerciales |
| **GPL-3.0** | Exige que les dérivés logiciels distribués restent ouverts sous GPL ; peut limiter certains partenariats commerciaux |
| **Tous droits réservés** | Garde davantage de contrôle, mais ne constitue pas une licence open source et rend la réutilisation académique moins claire |
| **CC BY 4.0** pour les rapports | Permet la réutilisation des documents avec attribution, si ce niveau d’ouverture est souhaité |

Ne copiez pas une licence depuis un autre projet sans décider au préalable du niveau d’ouverture souhaité pour le code, les documents, les plans électroniques et un futur produit commercial.

## Auteur

**Charbel SOSSOU BIADJA**
Étudiant en électrotechnique et auteur d’un projet autonome de recherche et développement
ENSET / UNSTIM, Bénin

---

## Références

[1] [Murray, D. et Stankovic, L. — REFIT: Electrical Load Measurements (Cleaned), Zenodo](https://doi.org/10.5281/zenodo.5063428)

[2] [Murray, D., Stankovic, L. et Stankovic, V. — An electrical load measurements dataset of United Kingdom households from a two-year longitudinal study, Scientific Data](https://doi.org/10.1038/sdata.2016.122)
