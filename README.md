<h1>Telco Customer Churn Prediction</h1>

Author: Reyner Elazaro Tampubolon and Demas Fadel Anggara

## 1. Project Overview

This project builds and evaluates a supervised classification model that estimates each telecom customer's probability of churning (cancelling the subscription), and turns those probabilities into a retention-contact decision for the **Customer Success (CS)** team.

The workflow covers:
- data understanding, data quality and exploratory analysis;
- preprocessing and feature engineering;
- a fair comparison of six algorithms with **nested cross-validation**;
- a class-imbalance study (class weighting, random oversampling, SMOTENC);
- cost-based threshold selection;
- a single final test evaluation;
- an illustrative business cost analysis;
- a saved, reusable model artifact.

## 2. Business Problem

About one in four customers in the data churned. Retention actions (offers, plan changes, technical support) can only be targeted if at-risk customers are identified *before* they leave, but identification is currently reactive and manual.

**Research question:** *How can we identify customers at risk of churning, from their profile, services, contract and charges, so that the CS team can target retention actions more precisely and earlier?*

## 3. Objectives

1. Identify customer characteristics associated with churn.
2. Build and compare classification models, and select one that is accurate, reliable and explainable.
3. Produce calibrated churn probabilities for ranking customers by risk.
4. Choose a decision threshold based on the trade-off between missed churners and unnecessary offers.
5. Estimate, under clearly stated assumptions, whether model-based targeting is better than contacting no one or everyone.

## 4. Dataset

| Item | Value |
|---|---|
| File | `data/raw/data_telco_customer_churn.csv` (course-provided; read-only) |
| Size | 4,930 rows × 11 columns |
| Target | `Churn`: No = 3,614 (73.31%), Yes = 1,316 (26.69%); **Yes is the positive class** |
| Predictors | `Dependents`, `tenure`, `OnlineSecurity`, `OnlineBackup`, `InternetService`, `DeviceProtection`, `TechSupport`, `Contract`, `PaperlessBilling`, `MonthlyCharges` |
| Data quality | No missing values; 77 exact duplicate rows (146 rows involved), retained because there is no customer ID; 8 customers with `tenure = 0`, retained; `No internet service` in the four add-on columns occurs exactly when `InternetService = No` |

Feature definitions come from `Telco Customer Churn.docx`. The collection period and source company are not documented.

## 5. Project Structure

```
├── README.md                     <- This file
├── requirements.txt              <- Pinned runtime dependencies
├── PROJECT_PROGRESS.md           <- Progress log, decisions and open issues
├── CLAUDE.md                     <- Project working instructions
├── Capstone Guideline.md         <- Course assessment guideline
├── data/
│   ├── raw/
│   │   └── data_telco_customer_churn.csv   <- Original dataset (read-only, never modified)
│   └── processed/                          <- Written by notebook 00, Section 6.1 (see Section 7.1)
│       ├── data_cleaned.csv                <- All 4,930 rows after the documented cleaning decisions
│       ├── data_train.csv                  <- The 3,944 training customers of the project split
│       ├── data_test.csv                   <- The 986 test customers of the project split
│       └── split_metadata.json             <- Split configuration, row counts, SHA-256 hashes
├── notebooks/                    <- The maintained analysis (run in order 00 -> 09)
├── src/                          <- Shared code imported by every notebook
│   ├── data_preprocessing.py     <- Constants, loading/validation, split, encoders, pipeline factory,
│   │                                dataset export/verification, fully nested candidate configurations
│   ├── feature_engineering.py    <- Candidate features, VIF, correlation ratio
│   ├── evaluation.py             <- Nested CV, metrics, cost model, selection rule, result storage
│   └── visualization.py          <- Plot helpers and figure export
├── models/
│   ├── churn_pipeline.pkl        <- Final deployable model (preprocessing + model + threshold)
│   └── model_metadata.json       <- Configuration, metrics, limitations, package versions
├── reports/
│   ├── experiment_comparison.csv <- Consolidated model comparison (nested CV)
│   ├── model_selection.json      <- Training-stage decisions passed to notebook 09
│   ├── cv_results/               <- Saved nested-CV results of notebooks 01-07
│   └── figures/                  <- Figures exported by the notebooks (used in the presentation)
├── presentation/telco_churn_presentation.pptx   <- Project presentation
├── docs/
│   ├── PROJECT_METHODOLOGY.md     <- Methodology reference
│   └── ORAL_DEFENSE_STUDY_GUIDE.md <- Personal oral-examination preparation (not part of the submission notebooks)
├── archive/                      <- Superseded material, kept for traceability (not part of the analysis)
│   ├── trial_claude_monolithic_v2.ipynb    <- Earlier single-notebook version
│   └── legacy/1.0-bdp-notebook.ipynb      <- Original course-template draft (Indonesian), superseded by notebooks 00-09
├── backups/                      <- Unmodified copies of files taken before each round of changes
└── trials.ipynb                  <- Early experiments (not maintained; reads a CSV path that no longer exists)
```

`src/tools.py` is an empty course-template file and is not used.

## 6. Installation and Environment Setup

Tested with **Python 3.14.5** on Windows.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

A Jupyter front end (JupyterLab, Jupyter Notebook, or VS Code with the Python/Jupyter extensions) is needed to open the notebooks; `ipykernel` from `requirements.txt` provides the kernel.

## 7. How to Run the Notebooks

Open the project folder and run the notebooks **in order**. Each one locates the project root automatically, so it can be started from the project root or from `notebooks/`.

1. `00` (independent).
2. `01`–`07` (independent of each other; each saves its results to `reports/cv_results/`).
3. `08` (reads the results of 01–07 and writes `reports/model_selection.json` and `reports/experiment_comparison.csv`).
4. `09` (reads the selection, evaluates the test set once, and writes `models/`).

Total runtime is about 7 minutes on a 16-core machine (last full run: 412 seconds).

### 7.1 Exported datasets (`data/processed/`)

Notebook 00 (Section 6.1) writes three CSV files for transparency and reuse, using `src.data_preprocessing.export_processed_datasets`. That function calls the **same** `split_train_test` function as every notebook, so the split is not drawn a second time. `verify_processed_datasets` then checks the following and raises an error if any check fails:
- files exist and load;
- row counts, column order and data types;
- no missing values;
- no train/test overlap and full coverage;
- the cleaned file is identical to the raw data;
- train/test membership **and row order** equal `split_train_test`.

| File | Rows | Churn = Yes | Purpose |
|---|---|---|---|
| `data_cleaned.csv` | 4,930 | 1,316 (26.69%) | Cleaned, analysis-ready data before model-specific preprocessing |
| `data_train.csv` | 3,944 (80%) | 1,053 (26.70%) | The training set used by notebooks 00–08 |
| `data_test.csv` | 986 (20%) | 263 (26.67%) | The holdout evaluated once in notebook 09 |

- **Columns:** `row_id` (the row's position in the raw CSV), the 10 predictors, then `Churn` (`Yes`/`No`). `row_id` is needed because 77 exact duplicate rows are kept, so identical rows can legitimately appear in both files; only an identifier can prove there is no overlap.
- **Cleaning** removed no rows and changed no values (Section 4 of this README; notebook 00, Section 5.7).
- **The files are not model-ready.** Categories are still text, `tenure` is not log-transformed and nothing is scaled. Fit `build_pipeline(...)` on the training data so that encoding and scaling stay inside the training folds.
- **Do not use `data_test.csv` for any modelling decision.** It is provided only to reproduce the single final evaluation.
- The split configuration (`test_size = 0.20`, stratified on `Churn`, `random_state = 34`) and file hashes are recorded in `split_metadata.json`.

```python
import sys
sys.path.insert(0, ".")  # project root
from src.data_preprocessing import load_processed, split_features_target

train = load_processed("train")            # validated, with the documented dtypes
X_train, y_train = split_features_target(train)   # 10 predictors and Churn as 0/1 (Yes = 1)
```

The training file keeps the split's row order, so `get_cv()` produces exactly the folds used in the notebooks.

## 8. Notebook Guide

| Notebook | Content |
|---|---|
| `00_data_understanding_and_preprocessing` | Business understanding, data understanding, data quality, stratified 80/20 split, export of `data/processed/` CSVs (Section 6.1), EDA (training set), encoding, feature engineering (`log(1+tenure)`), feature selection |
| `01_logistic_regression` | How it works, baseline, nested-CV tuning, calibration, coefficients/odds ratios, partial dependence, permutation importance |
| `02_decision_tree` | Same structure; tree visualisation |
| `03_random_forest` | Same structure; permutation vs impurity importance |
| `04_hist_gradient_boosting` | Same structure (scikit-learn gradient boosting) |
| `05_xgboost` | Same structure (XGBoost) |
| `06_lightgbm` | Same structure (LightGBM) |
| `07_smote_comparison` | No correction vs class weighting vs random oversampling vs SMOTENC, for Logistic Regression and HistGradientBoosting |
| `08_model_comparison_and_selection` | Consolidated comparison, pre-specified selection rule, cost-based threshold, sensitivity analysis, success criteria (including a check of all six models against the original targets and of shared false positives), fully nested robustness check of the preprocessing choices |
| `09_final_evaluation_and_business_analysis` | Single test-set evaluation with bootstrap CIs, business cost analysis, external-validation status, limitations, recommendations, model artifact |

## 9. Methodology

- **Leakage prevention:** every learned step (encoding, scaling, resampling) sits inside a scikit-learn / imbalanced-learn `Pipeline` and is fitted only on training folds. EDA, feature engineering, feature selection, tuning, model selection and threshold selection use only the training set.
- **Split and validation:**
  - one stratified 80/20 train/test split (3,944 / 986 customers, `random_state = 34`);
  - **nested 5-fold stratified cross-validation**: an inner grid search for hyperparameter tuning, outer folds for out-of-fold predictions that are free of hyperparameter-selection optimism.
- **What is *not* fully nested:** the preprocessing choices of notebook 00 (encoding, `log(1 + tenure)`, keeping all 10 features) were made with ordinary 5-fold CV on the training set. Those are the same folds later used as outer folds, so the nested estimates in Section 10 carry some preprocessing-selection optimism.
  - Notebook 08, Section 6 measures it. For Logistic Regression it repeats all 19 preprocessing configurations from notebook 00 inside the inner loop.
  - Mean PR-AUC changes from 0.666 to 0.665 (−0.001). The selection rule still picks Logistic Regression, and the cost-optimal threshold stays at 0.27.
  - The tree and boosting models were not re-run fully nested.
- **Preprocessing:**
  - binary add-on indicators (Yes = 1, otherwise 0);
  - one-hot `InternetService` and `Contract`;
  - binary `Dependents` / `PaperlessBilling`;
  - standardised `log(1 + tenure)` and `MonthlyCharges`.
- **Metrics:**
  - PR-AUC (primary, threshold-independent, focused on churners);
  - ROC-AUC and Brier score (calibration);
  - threshold-dependent accuracy, precision, recall, F1 and the confusion matrix.
- See `docs/PROJECT_METHODOLOGY.md` for details.

## 10. Models Evaluated (nested CV on the training set)

| Model | PR-AUC (mean ± SD) | ROC-AUC | Brier score |
|---|---|---|---|
| **Logistic Regression (selected)** | **0.666 ± 0.027** | 0.846 | 0.136 |
| XGBoost | 0.666 ± 0.021 | 0.847 | 0.136 |
| LightGBM | 0.665 ± 0.018 | 0.846 | 0.136 |
| Random Forest | 0.663 ± 0.024 | 0.845 | 0.137 |
| HistGradientBoosting | 0.657 ± 0.015 | 0.844 | 0.137 |
| Decision Tree | 0.623 ± 0.011 | 0.830 | 0.142 |
| Dummy (most frequent) | 0.267 | 0.500 | 0.267 |
| *Logistic Regression, fully nested (preprocessing also selected in the inner loop)* | *0.665 ± 0.028* | *0.845* | *0.137* |

**XGBoost and LightGBM** were tuned with exactly the same nested protocol (notebooks 05–06). They tie with Logistic Regression (differences ≤ 0.002 PR-AUC), so the added complexity buys nothing on these ten features.

**Class imbalance (notebook 07):** class weighting, random oversampling and SMOTENC changed PR-AUC by at most ±0.002 for either model, gave the same precision at matched recall, and worsened calibration (Brier 0.136 → about 0.164). They were not adopted; the imbalance is handled by the decision threshold instead.

## 11. Model-Selection Rationale

**Pre-specified rule:** choose the most interpretable model unless the best model beats it by more than 0.02 PR-AUC in at least 4 of 5 outer folds.

**Logistic Regression** (`C = 1`, L2, unweighted) has the highest mean nested PR-AUC. No other model is clearly better. It has the smallest training–validation gap, well-calibrated probabilities, readable coefficients and no extra dependencies.

**Threshold 0.27:**
- It minimises expected cost on nested out-of-fold training predictions.
- It matches the theoretical break-even probability c/(sH) = 0.278 for calibrated probabilities.
- It also corresponds to recall ≥ 0.80 on the training data.

## 12. Final Results (internal holdout test set, evaluated once)

| Metric | Test | 95% bootstrap CI |
|---|---|---|
| Recall | 0.810 | 0.764–0.859 |
| Precision | 0.516 | 0.469–0.563 |
| F1 | 0.630 | 0.589–0.673 |
| Accuracy | 0.747 | 0.721–0.774 |
| ROC-AUC | 0.854 | 0.830–0.878 |
| PR-AUC | 0.685 | 0.626–0.741 |
| Brier score | 0.132 | 0.121–0.143 |

**Confusion matrix (986 customers):** TP 213, FN 50, FP 200, TN 523. The model contacts 413 customers (42%) and reaches 81% of churners.

**Original success criteria (`notes.md`, Section 1.6) vs results**

| Criterion | Target | Test result (95% CI) | Met? |
|---|---|---|---|
| Recall | ≥ 80% | 0.810 (0.764–0.859) | **Yes** (point estimate; the interval includes values below 0.80) |
| Precision | ≥ 60% | 0.516 (0.469–0.563) | **No** |
| F1 | ≥ 65% | 0.630 (0.589–0.673) | **No** |
| ROC-AUC | ≥ 80% | 0.854 (0.830–0.878) | **Yes** |

**Why the precision and F1 targets were missed.** Precision and F1 depend on the threshold, so they cannot be "fixed" by retuning it. The recall and precision targets lie on the same precision–recall curve. On the training data (nested out-of-fold predictions), the best precision at recall ≥ 0.80 is 0.522, and keeping precision ≥ 0.60 limits recall to 0.649. The maximum F1 at any threshold is 0.635. The same holds for every algorithm (notebook 08, Section 5.4). No threshold of any of the six models meets precision ≥ 0.60 and F1 ≥ 0.65 together, and their maximum F1 values are only 0.625–0.645. At the same flag rate, Random Forest, HistGradientBoosting, XGBoost and LightGBM also flag 91–94% of Logistic Regression's false positives (rank correlation 0.97–0.98). The shortfall comes from the features, not from the choice of algorithm.

With these ten features, the targets are therefore not jointly achievable. The project uses an expected-cost criterion to choose the operating point instead (notebook 08, Section 5.3).

**Illustrative business comparison (same test customers)**
- *Assumptions:* 12-month revenue horizon, 30% retention success rate, offer cost = one month of charges. All hypothetical.
- *Expected cost (revenue at risk + offer spend):* **model 209,177** vs do nothing 236,717 vs contact everyone 230,637. The model strategy is 11.6% below doing nothing.
- These are **revenue-at-risk estimates, not profit or realised savings**. The conclusion reverses when offers are expensive or ineffective (see notebooks 08–09).

**External validation:** not performed — no independent dataset is available.

## 13. Model Inference Example

```python
import pickle
import sys

import pandas as pd

sys.path.insert(0, ".")  # run from the project root: the pipeline references functions in src/

with open("models/churn_pipeline.pkl", "rb") as fh:  # only load pickles from trusted sources
    model = pickle.load(fh)

customers = pd.DataFrame([
    {"Dependents": "No", "tenure": 2, "OnlineSecurity": "No", "OnlineBackup": "No",
     "InternetService": "Fiber optic", "DeviceProtection": "No", "TechSupport": "No",
     "Contract": "Month-to-month", "PaperlessBilling": "Yes", "MonthlyCharges": 85.5},
    {"Dependents": "Yes", "tenure": 60, "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
     "InternetService": "DSL", "DeviceProtection": "Yes", "TechSupport": "Yes",
     "Contract": "Two year", "PaperlessBilling": "No", "MonthlyCharges": 70.0},
])
print(model.predict_proba(customers)[:, 1].round(3))  # churn probabilities -> [0.824 0.01 ]
print(model.predict(customers))                       # 1 = contact (probability >= 0.27) -> [1 0]
```

**How the artifact was built:**
- It is a scikit-learn `FixedThresholdClassifier` (threshold 0.27) wrapping the full preprocessing + Logistic Regression pipeline.
- It was refitted on all 4,930 labelled customers after every decision had been fixed on the training data.
- It accepts the 10 original columns as a DataFrame.
- Unseen category values raise an error by design.
- Metadata are in `models/model_metadata.json`.

**Security:** pickle files can execute code when loaded. Load this file only from this trusted project.

## 14. Limitations

- **Validation:** internal only (one split of one dataset); no external validation.
- **Earlier test-set exposure:** the test set was scored by earlier pipeline versions during the project. No decision used those scores, but the holdout is no longer perfectly fresh.
- **Preprocessing selection is not fully nested:** the encoding, `log(1 + tenure)` and feature-set choices used the same training folds that later serve as outer folds. A fully nested re-check for Logistic Regression changed mean PR-AUC by only −0.001 (notebook 08, Section 6), but the tree and boosting models were not re-checked.
- **Success criteria:** the original precision (≥ 60%) and F1 (≥ 65%) targets were not met.
- **Features:** only ten, with no usage, complaint, payment or contract-end-date data. Six algorithms reach the same performance ceiling (PR-AUC ≈ 0.665).
- **Committed customers:** the model rarely detects churn among customers on one- or two-year contracts.
- **Survivorship and data origin:** survivorship effects in `tenure`; undocumented data origin.
- **Business figures:** based on hypothetical assumptions; they measure revenue at risk, not profit.

## 15. Future Improvements

- Collect behavioural and service-experience features.
- Validate on a later period or another customer base.
- Run a randomised retention pilot to measure the real offer success rate.
- Monitor calibration and drift after deployment.
- Re-evaluate the boosting models when richer features exist.

## 16. Reproducibility

- `RANDOM_STATE = 34` is used for the split, every CV splitter, every model, permutation importance and the bootstrap.
- Shared constants and functions live in `src/`.
- Every notebook has been executed top-to-bottom in a fresh kernel, and the saved outputs reflect that run.
- Re-running produces identical results; only run times and object memory addresses differ.

## 17. Contact

Reyner Elazaro Tampubolon and Demas Fadel Anggara — Capstone Project, Module 3.
