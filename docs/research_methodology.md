# Research methodology and scope

## Scientific scope

This repository is a **scientific NILM research project**. Its purpose is to study appliance-level estimation from aggregate active-power measurements and to evaluate whether interpretable symbolic rules can complement a statistical model. The repository is not a didacticization project, a finished household controller, or a field-validated product.

The load-shedding concept is treated as a future application context. The results currently stored in the repository do not demonstrate safe autonomous control of mains-connected appliances.

## Data sources

The principal source is the cleaned REFIT Electrical Load Measurements dataset, especially House 1. REFIT contains aggregate and appliance-level active-power measurements from UK homes. REFIT is useful for method development, but it cannot by itself establish transferability to Beninese homes.

The project also uses iAWE for a limited common-appliance experiment and PLAID for external physical power references. PLAID is not joined to REFIT or iAWE sample by sample. Its sampling rate and measurement context are different. In this project, PLAID provides descriptive references by appliance type and operating regime.

Raw third-party datasets are excluded from Git. A reproducible run must record the dataset version, source URL, license, expected file name and preprocessing configuration.

## Data construction

The REFIT–iAWE experiment is restricted to three common semantic targets: refrigerator, washing machine and television. The mapping is centralized in `src/appliance_mapping.py` and is based on the dataset metadata rather than on a guess from channel statistics.

The corrected merged representation uses one target column, `Puissance_Cible`, together with `Appliance_Target`. This avoids the earlier failure mode in which separate target columns were filled with zeros for non-target appliances and could produce meaningless near-zero errors.

The large merged data are sampled to reduce memory pressure. Sampling must not be confused with validation: an experiment is only interpretable when the sampling rule, random seed and temporal split are recorded. Rare appliance cycles require sampling across the available time range rather than taking only the first rows.

## Candidate features and models

The current candidate features are aggregate active power, its first difference (`Delta_P`) and hour of day. Future comparisons should add lagged or windowed features only as separate ablation conditions, for example `P(t-1)`, `P(t-2)`, rolling mean and rolling variance.

The benchmark should compare at least:

1. a linear baseline;
2. a Random Forest baseline;
3. Random Forest plus the symbolic layer.

All three conditions must use the same target definition, preprocessing, temporal split and evaluation period.

## Evaluation protocol

The definitive benchmark is not yet complete. The required protocol is:

- chronological train, validation and test periods;
- no random shuffling across the final temporal boundary;
- preprocessing fitted only on the training period when a fitted transformation is used;
- fixed random seed for stochastic models;
- identical target and feature definitions across model conditions;
- MAE, RMSE and R² reported per appliance;
- sample counts and the proportion of non-zero target values reported;
- configuration and dataset identifiers stored with the result table.

The CSV files in `results/` were produced by different exploratory runs. They must not be merged into one official benchmark. In particular, a result measured on training data must not be presented as a held-out test result.

## Symbolic layer

`src/regles_symboliques.py` contains the PLAID reference table and the function `appliquer_regles()`. The current `REGLES_PAR_APPAREIL` mapping is intentionally empty because earlier rules degraded Random Forest results. An empty mapping is therefore a neutral baseline, not evidence that symbolic rules are beneficial.

A future rule must be specified before evaluation. The specification should include the appliance, condition, physical rationale, calibration source, calibration period, expected direction of correction and an ablation comparison against Random Forest alone. Rule thresholds must not be tuned on the final test segment.

## Current recorded evidence

One stored exploratory run reports the following values for the Random Forest and Soft-Boost conditions:

| Appliance | Random Forest MAE (W) | Soft-Boost MAE (W) | Interpretation |
|---|---:|---:|---|
| Refrigerator | 22.48 | 22.48 | No measured change in this run |
| Washing machine | 7.78 | 7.91 | Small deterioration in this run |
| Water heater | 28.78 | 31.83 | Deterioration in this run |

Another recorded hybrid experiment reports MAE 25.48 W, 14.39 W and 7.55 W for refrigerator, washing machine and television, respectively, with R² values of −0.018, 0.384 and 0.038. These values are retained as exploratory evidence and are not interchangeable with the table above because the runs use different data construction or evaluation settings.

## Reproducibility checklist

Before claiming a final result, the project should provide:

- the exact dataset identifiers and metadata files;
- the exact file layout expected under `data/raw/`;
- dependency versions;
- the preprocessing configuration;
- the random seed;
- the chronological split boundaries;
- the model hyperparameters;
- a test command that does not require private paths;
- a result table with one clearly identified held-out benchmark.

## Limitations

The current work has no local Benin measurement campaign, no field validation, no uncertainty analysis and no safety certification. REFIT and PLAID describe different populations, instruments and sampling contexts. Their combination can support hypothesis generation and method development, but it does not prove generalization.

## References

[1]: https://doi.org/10.5281/zenodo.5063428 "REFIT: Electrical Load Measurements (Cleaned)"

[2]: https://doi.org/10.1038/sdata.2016.122 "REFIT electrical load measurements dataset descriptor"

[1] [2]
