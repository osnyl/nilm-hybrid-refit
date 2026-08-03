\# Jarvis Energy : Hybrid NILM System



\## English Version



\### Overview

This project implements a \*\*Non-Intrusive Load Monitoring (NILM)\*\* system using a hybrid approach combining \*\*Linear Regression\*\* and \*\*Symbolic Rules\*\*.



The system disaggregates household electricity consumption (from a single smart meter) into individual appliances, using the public \*\*REFIT\*\* dataset. It applies a \*\*Principle of Least Action\*\* to decide which appliances to shed, prioritizing user comfort and energy savings.



\### Key Features

\- \*\*Statistical Core\*\* : Linear Regression trained via Gradient Descent (implemented from scratch).

\- \*\*Symbolic Layer\*\* : Contextual rules based on duration and power thresholds to correct ML errors (Soft-Boost).

\- \*\*9 Appliances Recognized\*\* : From refrigerators to washing machines.

\- \*\*Decision Logic\*\* : Applies a cost function \\( S = \\int (Cost + Comfort + Safety + Wear + Health) dt \\) to determine load shedding actions.



\### Dataset

\- \*\*Name\*\* : REFIT Power Dataset (House 1)

\- \*\*Source\*\* : Zenodo

\- \*\*Size\*\* : 383 MB, 6,960,008 rows (15-second intervals)

\- \*\*Structure\*\* : `Aggregate` (total power) and `Appliance1` to `Appliance9` (individual loads)



\### Methodology

1\. \*\*Exploration\*\* : Analysis of consumption patterns and appliance signatures (cycles, power, peak hours).

2\. \*\*Normalization\*\* : Min-Max scaling to handle the wide range of values (0W to 29kW).

3\. \*\*Linear Model\*\* : Predicts each appliance's power using only the aggregate power.

4\. \*\*Symbolic Correction\*\* : Applies physical rules (e.g., "If power > 1500W for > 10 min, boost the Water Heater").

5\. \*\*Decision Action\*\* : The final system uses a cost function to decide which appliance to turn off.



\### Project Structure

\- `src/01\_explore\_refit.py` : Loads and visualizes the dataset.

\- `src/02\_identify\_appliances.py` : Extracts appliance signatures (cycles, power).

\- `src/03\_train\_models.py` : Trains linear models for all appliances and evaluates performance.

\- `src/04\_symbolic\_booster.py` : Applies soft-boost rules to correct predictions (V3).



\### Quick Start

```bash

git clone https://github.com/YOUR\_USERNAME/nilm-hybrid-refit.git

cd nilm-hybrid-refit

pip install -r requirements.txt

\# Place CLEAN\_House1.csv in data/raw/

python src/01\_explore\_refit.py

python src/02\_identify\_appliances.py

python src/03\_train\_models.py

python src/04\_symbolic\_booster.py





\#Ce projet implémente un système de suivi de charge non intrusif (NILM) utilisant une approche hybride combinant régression linéaire et règles symboliques. Il utilise le dataset REFIT pour désagréger 9 appareils à partir de la puissance totale.



\*\*Auteur\*\*: SOSSOU BIADJA Osnyl

