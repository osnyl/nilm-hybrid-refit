# Hybrid NILM and Priority-Based Load Shedding

[Version française](docs/README_FR.md)

> **Research status: exploratory and in progress.** This repository documents an academic research-and-development project on non-intrusive load monitoring (NILM) and priority-based management of household electrical loads. It is not a production controller, a safety-certified device, or evidence of field performance in Beninese households.

## Abstract

The project investigates whether aggregate active-power measurements can support appliance-level estimation and inform a future intelligent load-management system. The current experiments use public REFIT data from United Kingdom homes. A statistical model provides the baseline, while interpretable symbolic rules are evaluated as a possible corrective layer. The load-shedding idea is treated as a future application context, not as a validated controller in this repository.

The repository is deliberately split between **research code**, **stored experimental evidence**, and **scientific documentation**. The current public snapshot should be read as a documented research baseline, not as a validated deployable product.

## Project progress

| Milestone | Status | Evidence in this repository |
|---|---|---|
| REFIT House 1 exploration | Completed baseline | `src/legacy/` and stored figures |
| Linear and Random Forest baselines | Exploratory results available | `results/` |
| Chronological train/test comparison | Initial diagnostic available | `results/eval_train_vs_test.csv` |
| PLAID physical-reference analysis | Implemented for selected regimes | `results/references_plaid_par_appareil.csv` and pipeline scripts |
| Symbolic-rule module | Centralized and unit-checkable | `src/regles_symboliques.py` |
| REFIT + iAWE common-appliance pipeline | Implemented, final clean benchmark pending | `src/pipeline/` |
| Local Benin validation | Not started | Future work |

The most important current finding is methodological: stored Soft-Boost experiments do **not** improve every appliance. For example, in one 2-million-row exploratory run, the Soft-Boost MAE was unchanged for the refrigerator and higher for the washing machine and water heater. This result is retained because negative or mixed results are useful for deciding what must be validated next.

## Research questions

1. How well can aggregate active power estimate selected appliance channels under a documented temporal split?
2. Can physical references from PLAID help define interpretable symbolic rules without being mistaken for a direct time-aligned measurement source?
3. Does the symbolic layer improve a held-out test segment compared with the statistical baseline?
4. Which additional temporal and physical features improve estimation without causing leakage or overfitting?

## Method

The experimental pipeline uses aggregate power, power variation and time context as candidate features. Random Forest and linear baselines are compared using MAE, RMSE and R². PLAID is used to estimate power references by appliance and operating regime; it is not joined sample-by-sample to REFIT or iAWE because the datasets have incompatible sampling contexts. A symbolic layer applies interpretable rules to selected predictions. The intended next benchmark is a fixed chronological train/validation/test protocol with recorded preprocessing and random seeds.

## Repository layout

```text
.
├── docs/
│   ├── README_FR.md
│   ├── *.pdf                         Research reports
│   └── session-reports/              Session documentation
├── figures/                          Selected figures
├── results/                          CSV summaries and result figures
├── src/
│   ├── pipeline/                     Current REFIT/iAWE/PLAID pipeline
│   ├── regles_symboliques.py         Interpretable rule module
│   ├── appliance_mapping.py          Central appliance mapping
│   └── legacy/                       Earlier exploratory scripts
├── load_data.txt                     Example signal for simulation preparation
├── requirements.txt
├── CITATION.cff
└── README.md
```

The raw datasets are intentionally excluded from Git. Generated processed datasets should also remain outside version control unless a small, legally shareable sample is explicitly documented.

## Reproducing the experiments

Create an environment and install the dependencies:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

Place the legally obtained REFIT file at:

```text
data/raw/CLEAN_House1.csv
```

The PLAID and iAWE scripts require their source files in the corresponding `data/raw/` subdirectories. Review each script's input requirements before running a large experiment. The pipeline scripts are research utilities and may require substantial memory when processing full datasets.

Example entry points from the repository root are:

```bash
python src/legacy/01_explore_refit.py
python src/pipeline/01_creer_echantillon.py
python src/pipeline/02_associer_references_plaid.py
python src/pipeline/03_entrainer_modele_hybride.py
```

The final comparative benchmark is not yet claimed to be fully reproducible. Before publication of a definitive performance table, the project still needs a clean temporal split, fixed configuration, documented dataset versions and a fresh end-to-end run.

## Current evidence

The stored CSV files contain results from different experiments and sample sizes. They must not be merged into one benchmark. A few representative observations from `results/experience4_2M_results.csv` are:

| Appliance | Random Forest MAE (W) | Soft-Boost MAE (W) | Interpretation |
|---|---:|---:|---|
| Refrigerator | 22.48 | 22.48 | No measured change in this run |
| Washing machine | 7.78 | 7.91 | Small deterioration in this run |
| Water heater | 28.78 | 31.83 | Deterioration in this run |

The values are exploratory outputs, not a claim of field accuracy, transferability to Benin, or operational safety.

## Scientific scope and documentation

This repository is strictly scientific. It documents NILM experiments, data construction, physical references and symbolic-rule evaluation. It does not claim to contain a didacticization project, a complete Proteus source file, a deployable Arduino/ESP32 controller, a PCB design, a field-data protocol or electrical-safety certification. The detailed protocol and scope are documented in [`docs/research_methodology.md`](docs/research_methodology.md), and the latest work is available in [`docs/session-reports/rapport_sessions_23_24_septembre_2026.md`](docs/session-reports/rapport_sessions_23_24_septembre_2026.md).

## Limitations and safety

REFIT was collected in UK homes. Its appliance mix, voltage context, wiring, occupancy patterns and usage habits are not proof of performance in Beninese homes. Local validation requires consent, appropriate measurement equipment and a separately documented protocol.

> **Safety notice:** Never connect experimental microcontrollers, sensors, relays or software directly to mains electricity without qualified supervision, suitable isolation and protection, correctly rated components, and compliance with local electrical rules. This repository is not a wiring guide.

## Roadmap

1. Complete and review the chronological benchmark on a held-out test segment.
2. Add tests for the mapping and symbolic-rule module.
3. Record exact dataset versions, preprocessing settings and random seeds.
4. Publish only small, legally shareable examples rather than raw third-party datasets.
5. Compare REFIT-trained estimates with a small, consent-based local pilot before making transfer claims.
6. Archive a reviewed release and add a DOI only after the results and licensing are settled.

## Citation

If you use or discuss this repository, cite the project metadata in [`CITATION.cff`](CITATION.cff) and cite the REFIT dataset and descriptor:

> Murray, D., Stankovic, L. and Stankovic, V. *An electrical load measurements dataset of United Kingdom households from a two-year longitudinal study.* Scientific Data 4, 160122 (2017). [doi:10.1038/sdata.2016.122](https://doi.org/10.1038/sdata.2016.122)

## Author

**Charbel SOSSOU BIADJA (Osnyl)**
Electrical Engineering student and independent research-and-development project author
ENSET / UNSTIM, Benin

## References

[1]: https://doi.org/10.5281/zenodo.5063428 "REFIT: Electrical Load Measurements (Cleaned)"

[2]: https://doi.org/10.1038/sdata.2016.122 "REFIT electrical load measurements dataset descriptor"

[3]: https://github.com/Osnyl/nilm-hybrid-refit "Project repository"

[1] [2] [3]
