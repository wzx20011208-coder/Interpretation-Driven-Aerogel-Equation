# Supplementary Code and Data

**Manuscript:** *Interpretation-driven construction of an explicit equation for the thermal conductivity of aerogel composite insulation under coupled degradation states*

This package contains all source code and experimental data required to reproduce the results reported in the manuscript. All scripts resolve the data through relative paths (`../data`) and were verified to reproduce the published numbers.

## 1. Package structure

```
Supplementary_Code_and_Data/
├── data/                        # experimental datasets (3 files)
├── code_main/                   # main analysis pipeline (5 scripts)
├── pysr_baseline/               # PySR symbolic-regression baseline comparison (8 scripts)
├── verification/                # verification and recalculation scripts (8 + B_1000boot_ridge/)
└── README.md
```

## 2. Requirements

- Python 3.10 or later (tested on 3.13)
- Packages: `numpy`, `pandas`, `scipy`, `statsmodels`, `scikit-learn`, `catboost`, `shap`, `matplotlib`, `sympy`
- `pysr` (requires a Julia runtime) — only for `pysr_baseline/01–04`
- Keep the folder structure intact: every script locates the data via `../data` relative to its own location. Generated outputs (figures and JSON files) are written to `code_main/outputs/` (created automatically) or next to the script.

## 3. Data description

| File | Content |
|---|---|
| `data/degradation_dataset.xlsx` | Sheet `degradation_data`, 324 rows × 7 columns: degraded-condition measurements (243 single-factor + 81 coupled runs from an L9(3⁴) orthogonal design crossed with 9 temperature–humidity grid cells) |
| `data/intact_conditions_summary.xlsx` | Sheet `summary`: 27 intact-condition grid-cell statistics (mean, standard deviation, number of readings); sheet `raw_readings`: 1542 individual thermal-conductivity readings of the intact state |
| `data/FPI_CellMatched_detail.csv` | Measured coupled-degradation interaction index (FPI, %) and its additive-superposition components at the 27 temperature–humidity grid cells |

**Column dictionary**

| Column | Unit / levels | Description |
|---|---|---|
| `Specimen_ID` | — | Specimen label |
| `Temperature_C` | °C, 5 / 10 / 15 / 20 / 25 / 30 / 35 / 40 / 45 | Ambient temperature |
| `RH_pct` | %, 30 / 60 / 90 | Relative humidity |
| `FT_Cycles` | 0 / 18 / 21 / 24 | Freeze–thaw cycles |
| `Crack_Length_cm` | cm, 0 / 2.5 / 5 / 10 | Parallel crack length (air-filled, fixed 0.5 mm width) |
| `Water_Content_pct` | %, 0 / 23 / 38 / 56 | Water content by mass |
| `Thermal_Conductivity` | W/(m·K) | Measured thermal conductivity |
| `Thermal_Conductivity_Mean/_STD`, `N_Readings` | W/(m·K), — | Intact-state summary statistics |
| `fpi_pct`, `L`, `C` | % | Coupled interaction index and superposition components |

Full sample: 351 points = 324 degraded + 27 intact grid-cell means.

## 4. The final explicit equation (11 terms + intercept)

```
λ = 2.338e-2 + 2.217e-4·T − 8.980e-7·T² + 2.983e-7·RH²
  + 5.084e-4·√FT + 5.687e-4·√CR + 1.724e-4·W − 1.423e-6·W²
  − 2.123e-5·(T·√CR) − 2.909e-7·(RH·W) + 1.604e-4·(√FT·√CR) − 4.117e-6·(RH·√CR)
```

Three-layer structure: intact baseline (T, T², RH²) + degradation main effects (√FT, √CR, W, W²) + coupling interactions (T×√CR, RH×W, √FT×√CR, RH×√CR). Fitted by unweighted OLS; cross-validation with `KFold(n_splits=5, shuffle=True, random_state=42)`.

## 5. Scripts and reproduction guide

### `code_main/` — main pipeline

| Script | Content | Key reproduced numbers |
|---|---|---|
| `main_analysis.py` | Main pipeline: data loading, CatBoost black-box training, SHAP attribution, backward elimination, OLS fitting of the final 11-term equation, FPI magnitude check (Sections 2.2, 3.1–3.3) | Full-fit R² = 0.9644; CV R² = 0.9594, MAPE = 1.76%; CatBoost CV R² = 0.9821, MAPE = 1.10% |
| `shap_three_layer.py` | Three-layer SHAP attribution and third-order term nested test (Section 3.2) | SHAP values 0.002056 / 0.001726 / 0.001082 / 0.000553 / 0.000319 |
| `supplementary_validation.py` | LOLO / LOCO / bootstrap / Sobol supplementary validation (Section 3.4) | LOCO R² = 0.9622 |
| `scenario_extrapolation.py` | Leave-one-scenario-out extrapolation over the five standardized degradation scenarios | Scenario-wise RMSE improvements |
| `classical_model_comparison.py` | Comparison with classical analytical models: Wiener/Maxwell–Eucken bounds with Knudsen correction, moisture-response models, series air-gap crack model, damage-accumulation models, temperature anchor | Model-comparison table |

`supplementary_validation.py` creates `code_main/outputs/v4_supplement.json`; `shap_three_layer.py` and `scenario_extrapolation.py` merge their sections into it (they start a fresh file if it is absent). Suggested run order: `main_analysis.py` → `shap_three_layer.py` → `supplementary_validation.py` → `scenario_extrapolation.py` → `classical_model_comparison.py`.

### `pysr_baseline/` — symbolic-regression blind-search baseline

| Script | Content | Key reproduced numbers |
|---|---|---|
| `00_data_check.py` | Dataset consistency check | 351 samples |
| `01_pysr_cv.py` | PySR 5-fold CV, operator set A (general) / B (physics-informed), maxsize = 25 | A: R² = 0.9116; B: R² = 0.8710 |
| `02_pysr_stability.py` | 5-seed stability | Crack length retained in 0/10 folds |
| `03_pysr_budget.py` | Computational-budget experiment (300 iterations) | R² = 0.927 |
| `04_pysr_maxsize.py` | Complexity budget maxsize = 40 / 60 | A40: 0.9457; B40: 0.9401; B60: 0.9459 |
| `05_catboost_unweighted.py` | Unweighted CatBoost reference (main protocol, Section 3.1) | R² = 0.9821, MAPE = 1.10% |
| `06_sign_check.py`, `06b_sign_agreement.py` | Sign-agreement checks (06b reads the JSON outputs of 01/02; run them first) | — |

Note: 01–04 require PySR/Julia and are computationally heavy.

### `verification/` — key-number recalculation

| Script | Key reproduced numbers |
|---|---|
| `verify_eniso_numbers.py` | FULL 351: R² = 0.9646; DEGRADED 324: R² = 0.9693; 5-fold OOF CV: R² = 0.9605 |
| `27_fpi_consistent.py` | FPI unified protocol (measured and equation both at grid-cell level means): equation mean 9.40%, capture ratio 85.7%, Pearson r = 0.935, Spearman ρ = 0.896 |
| `compute_table6_matrix.py` | Table 6: equation vs. CatBoost vs. EN ISO 10456 by condition group (EN ISO: R² = 0.912 intact, 0.423 single-factor, −17.605 coupled, MAPE 18.04%) |
| `21_aic_bic_final.py` | AIC: final 11-term (−4101.3) < full 13-term (−4100.6) |
| `03_recompute_key_metrics.py`, `10_M1_refit_full.py`, `12_final11_correction_test.py`, `31_reproduce_blackbox.py` | Supporting recalculations (nested CV 0.9133 → 0.9417 → 0.9598 → 0.9594; low-temperature crack-sign analysis; black-box replication) |
| `B_1000boot_ridge/B1_1000_bootstrap.py` | 1000× bootstrap: sign-stability rate 100% for all 12 coefficients under both stratifications |
| `B_1000boot_ridge/b1b_1000_bootstrap_retention.py` | Structure-retention frequencies of the 11 final terms (28.7%–100%) |
| `B_1000boot_ridge/B2_ridge_sensitivity.py`, `B2_fig.py` | Ridge-regression sensitivity and figure (B2_fig reads the JSON written by B2) |

## 6. Notes

- Data-file headers are in English; the scripts internally use Chinese column aliases assigned positionally right after loading. Glossary: 温度 = temperature; 湿度 = relative humidity; 冻融次数 = freeze–thaw cycles; 裂缝长度 = crack length; 含水率 = water content; 导热系数 = thermal conductivity; 样本 = specimen.
- Console output of the scripts is partly in Chinese.
- The degradation scenarios follow the standardized terminology: Intact, Freeze-thaw only, Crack only, Moisture only, and Coupled.
