# Hybrid NILM for Intelligent Load Shedding

> **Research status — exploratory.** This repository documents an academic research and development project on non-intrusive load monitoring (NILM) and priority-based load-shedding concepts for households in low-resource settings. It is **not** a production-ready, safety-certified, or field-validated electrical protection device.

## Overview

This project investigates whether **aggregate active-power measurements** can support appliance-level consumption estimation and inform a future intelligent load-shedding system. The machine-learning experiments use the public REFIT electrical-load dataset, while the accompanying research reports discuss a Proteus-based simulation of household load-shedding scenarios.

The central research idea is hybrid: a statistical model estimates appliance consumption from the aggregate signal, and simple, interpretable symbolic rules are explored as a possible corrective layer. The longer-term aim is to help a household preserve its most important loads when its available power is constrained. The present public snapshot is a **research baseline**, not an operational controller.

| Dimension | Current scope |
|---|---|
| Input data | Aggregate active power from REFIT House 1 |
| Appliance channels studied | Nine labelled appliance channels in REFIT House 1 |
| Modelling approach | Linear baselines, Random Forest experiments, and exploratory Soft-Boost rules |
| Load-shedding link | Research concept and simulation discussion; no deployable controller in this repository |
| Target context | Low-resource households; field transfer to Benin has **not** yet been validated |

## What this repository contains

The repository includes Python scripts for data exploration, appliance-channel exploration, model experiments, symbolic-rule experiments, and preparation of a `load_data.txt` signal for simulation use. It also contains stored figures and CSV result summaries. The supporting [research report in English](docs/Intelligent%20Load%20Shedder_Research%20Report%202026.pdf) and [rapport de recherche en français](docs/Rapport%20de%20Recherche%20Delesteur%20Intelligent%202026.pdf) are available in the `docs/` directory.

The project currently provides evidence of **exploratory modelling work** on REFIT data. It does not yet provide a complete reproducible hardware package: the public snapshot does not contain a Proteus project file, an Arduino/ESP32 sketch, PCB files, a bill of materials, a field-data collection protocol, or electrical-safety certification.

## Research boundary and honesty statement

Several important distinctions are deliberately made in this README.

First, REFIT measurements were collected in United Kingdom homes. They are useful for developing and studying NILM methods, but they are not proof that the model will behave the same way in Beninese households, where appliances, occupancy patterns, wiring, voltage conditions, and usage habits can differ.

Second, the CSV files in `results/` preserve experimental outputs. The current scripts require a documented chronological train/validation/test protocol before the reported figures can be presented as out-of-sample performance. Therefore, the values below should be read as **exploratory experiment outputs**, not as a claim of field accuracy or deployment readiness.

Third, the four-term optimization function sometimes discussed in the research reports—financial cost, user comfort, safety, and appliance wear—is a **future research direction**. The public Python files do not implement a validated real-time optimizer based on that function. The practical load-shedding mechanism to be validated in a future prototype is priority-based control of non-essential loads.

> **Safety notice:** Do not connect experimental microcontrollers, relays, sensors, or software directly to mains electricity without a qualified electrician, suitable isolation and protection devices, and compliance with local electrical rules. This repository must not be used as wiring instructions.

## Dataset

The experiments use the cleaned REFIT Electrical Load Measurements dataset, particularly `CLEAN_House1.csv`. REFIT provides aggregate and appliance-level active-power measurements from 20 UK households, sampled at 8-second intervals. The dataset is available under CC BY 4.0 and must be properly attributed. [1] [2]

The raw REFIT data are deliberately **not stored in this repository**. Download them from the official record and place the required file at:

```text
data/raw/CLEAN_House1.csv
```

The current `.gitignore` excludes this raw CSV. This keeps the repository small and avoids redistributing a third-party dataset unnecessarily.

### Required dataset citation

Please cite the REFIT dataset and its data descriptor when using or extending this work:

> Murray, D., Stankovic, L. and Stankovic, V. *An electrical load measurements dataset of United Kingdom households from a two-year longitudinal study.* Scientific Data 4, 160122 (2017). https://doi.org/10.1038/sdata.2016.122

## Repository layout

```text
.
├── docs/       Research reports in French and English
├── figures/    Prediction and simulation figures
├── results/    CSV summaries and generated result figures
├── src/        Exploratory Python scripts
├── load_data.txt
├── requirements.txt
└── README.md
```

| Path | Purpose |
|---|---|
| `src/01_explore_refit.py` | Basic loading and descriptive exploration of REFIT House 1 |
| `src/02_identify_appliances.py` | Simple descriptive appliance-channel analysis |
| `src/03_train_models.py` | Linear and Random Forest experiment code; currently needs refactoring for portability and evaluation discipline |
| `src/04_symbolic_booster.py` | Exploratory Soft-Boost rule experiment on selected appliance channels |
| `src/05_generate_load_file.py` | Converts aggregate power samples into a simple current/time signal for simulation use |
| `results/` | Stored experimental summaries; see the limitations below before using them as evidence |

## Experimental results currently stored

The file [`results/experience4_2M_results.csv`](results/experience4_2M_results.csv) contains a two-million-row experiment summary. The selected examples below illustrate an important research lesson: in this run, the stored Soft-Boost rules did **not** improve every appliance channel and slightly worsened several of them. This is useful evidence that the symbolic layer needs disciplined validation rather than optimistic claims.

| Appliance channel | Random Forest MAE (W) | Soft-Boost MAE (W) | Interpretation |
|---|---:|---:|---|
| Refrigerator | 22.48 | 22.48 | No measured change in this stored experiment |
| Washing machine | 7.78 | 7.91 | Small deterioration in the stored experiment |
| Water heater | 28.78 | 31.83 | Deterioration in the stored experiment |

The repository also contains earlier result files with different sample sizes and methods. They should not be combined as if they were one benchmark. Future work should fix the random seed, record the preprocessing settings, use chronological train/validation/test splits, and report metrics on a held-out test segment.

## Reproducibility status

The repository is intentionally public, but reproducibility is currently **partial**. Before running the full workflow, the following improvements are needed:

| Item | Current status | Required improvement |
|---|---|---|
| Raw data | Not tracked, by design | Download `CLEAN_House1.csv` from the official REFIT source |
| Dependencies | `requirements.txt` is incomplete for Random Forest code | Add `scikit-learn` and pin versions after a clean run |
| File paths | Some scripts include a local Windows path | Replace it with a relative project path |
| Evaluation | Current scripts need a clean hold-out protocol | Use chronological train/validation/test splits |
| Hardware simulation | Only figures and `load_data.txt` are public | Add the legal-to-share Proteus source and Arduino/ESP32 code, or document their absence |
| Safety | No field verification or certification | Test only under supervision before any mains-connected prototype |

For a local Python environment, begin with:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install scikit-learn
```

Then download the REFIT file manually, place it in `data/raw/`, and review file paths before executing any script. Do not run the workflow until the evaluation split and configuration have been checked.

## Connection to intelligent load shedding

The associated reports discuss three simulated household scenarios and a future intelligent load-shedding architecture. The current public repository should be interpreted as the **NILM research component** and a simulation-data preparation component. It does not yet demonstrate a safe, real-time system that identifies arbitrary appliances in a Beninese home or autonomously controls mains loads.

A responsible prototype roadmap is:

1. Validate the baseline model with a reproducible chronological hold-out protocol.
2. Measure aggregate consumption and selected appliance loads in a small, consent-based local pilot.
3. Compare the REFIT-trained baseline against local data without claiming transferability in advance.
4. Implement a simple priority-based controller for low-voltage test conditions first.
5. Involve a qualified electrician before any mains-connected installation.
6. Evaluate user benefit, electrical safety, and actual consumption effects before making commercial claims.

## How to cite this repository

A project DOI has not yet been assigned. After a Zenodo archival release is created, add the DOI here and create a `CITATION.cff` file with the release details.

Until then, a temporary citation format is:

> SOSSOU BIADJA, Charbel (2026). *Hybrid NILM for Intelligent Load Shedding: Exploratory REFIT Experiments and Load-Shedding Research Concepts.* GitHub repository: https://github.com/Osnyl/nilm-hybrid-refit

## Licence

No licence file has yet been selected for this repository. This decision should be made deliberately before the Zenodo release.

- **MIT** is simple and permissive for code, but it allows broad reuse, including commercial reuse.
- **GPL-3.0** preserves openness of derivative software, but may discourage some commercial partners.
- **All rights reserved** retains stricter control but is not an open-source licence and makes reuse by researchers less clear.
- The research reports and figures can use a distinct document licence, such as **CC BY 4.0**, if you choose to permit attribution-based reuse.

Do not copy a licence from another project without first deciding how open you want the code, the reports, the electronics design, and any future commercial product to be.

## Author

**Charbel SOSSOU BIADJA**<br>
Electrical Engineering student and independent research-and-development project author<br>
ENSET / UNSTIM, Benin

---

## References

[1] Murray, D. and Stankovic, L. (2016). *REFIT: Electrical Load Measurements (Cleaned).* Zenodo. https://doi.org/10.5281/zenodo.5063428

[2] Murray, D., Stankovic, L. and Stankovic, V. (2017). *An electrical load measurements dataset of United Kingdom households from a two-year longitudinal study.* Scientific Data, 4, 160122. https://doi.org/10.1038/sdata.2016.122
