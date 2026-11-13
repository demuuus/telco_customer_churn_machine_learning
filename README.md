# Telco Customer Churn Prediction

Capstone Project, Module 3 (Machine Learning). Authors: Reyner Elazaro Tampubolon and Demas Fadel Anggara.

## 1. Project Overview

This project builds and evaluates a supervised binary classifier that estimates the probability that a telecom customer has churned (cancelled the subscription), using ten customer attributes. Six algorithms are compared with nested cross-validation, class-imbalance handling is examined, and an operating threshold is chosen from an illustrative cost model. The selected model is Logistic Regression with a probability threshold of 0.27.

The model is a churn **classification** model fitted on historical customer attributes. The dataset has no dates or observation window, so the project does not establish that the model predicts churn *before* it happens (see [Section 14](#14-project-limitations)).

## 2. Problem Statement

About one customer in four in the data has churned. A customer-success team that wants to direct retention offers needs a way to rank customers by churn risk and to decide how many of them to contact. The research question of the project is: *which customers have a high probability of churn, given their profile, services, contract and charges, and how should that probability be turned into a contact decision?*

## 3. Project Objectives

1. Describe the customer characteristics associated with churn.
2. Compare several classification algorithms under one validation protocol and select one with a documented rule.
3. Produce probability estimates that rank customers by risk and are reasonably calibrated.
4. Choose a decision threshold by explicit reasoning about missed churners versus unnecessary offers.
5. Report, under clearly stated assumptions, how the resulting contact strategy compares with contacting no one or everyone.

The original success targets (`notes.md`, Section 1.6) were recall ≥ 80%, precision ≥ 60%, F1 ≥ 65% and ROC-AUC ≥ 80%.

## 4. Dataset Description

| Item | Value |
|---|---|
| File | `data/raw/data_telco_customer_churn.csv` (provided with the course; not modified) |
| Size | 4,930 rows, 11 columns |
| Target | `Churn`: No = 3,614 (73.31%), Yes = 1,316 (26.69%). *Yes* is the positive class |
| Predictors (10) | `Dependents`, `tenure`, `OnlineSecurity`, `OnlineBackup`, `InternetService`, `DeviceProtection`, `TechSupport`, `Contract`, `PaperlessBilling`, `MonthlyCharges` |
| Missing values | None |
| Duplicates | 77 exact duplicate rows (146 rows involved), retained because no customer identifier exists |
| Other | 8 customers with `tenure = 0` (retained); `No internet service` in the four add-on columns occurs exactly when `InternetService = No` |

Feature definitions supplied with the dataset:

| Feature | Meaning |
|---|---|
| `Dependents` | Whether the customer has dependents |
| `tenure` | Number of months the customer has stayed with the company |
| `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport` | Whether the customer has the add-on service (Yes / No / No internet service) |
| `InternetService` | Type of internet service (DSL / Fiber optic / No) |
| `Contract` | Contract type by duration (Month-to-month / One year / Two year) |
| `PaperlessBilling` | Whether bills are issued in paperless form |
| `MonthlyCharges` | Monthly charge for the services |
| `Churn` | Whether the customer churned |

The source company, the collection period and the licence of the dataset are not documented. See [Section 14](#14-project-limitations).

## 5. Exploratory Data Analysis

EDA was carried out on the training set only (notebook `00`). Selected observations from the 3,944 training customers (descriptive, not causal):

- Churn rate by contract: month-to-month 43.3%, one year 9.9%, two year 2.7%.
- Churn rate by tenure band: 0–12 months 47.9%, 13–24 months 27.7%, 25–48 months 21.1%, 49–72 months 9.1%.
- Churn rate by internet service: fiber optic 42.2%, DSL 18.8%, no internet 7.6%.
- Customers without tech support (but with internet) churn at 41.4%, with tech support at 15.4%.
- Paperless billing: 33.4% versus 16.5%. Customers with dependents: 14.8% versus 31.8%.

Tenure partly reflects survival: long-tenure customers are partly those who have not churned yet, which exaggerates its apparent protective effect.

## 6. Data Preprocessing

All learned transformations live inside a scikit-learn `Pipeline`, so they are fitted on training folds only.

| Step | Treatment |
|---|---|
| Duplicates, `tenure = 0`, outliers | All kept; nothing removed or imputed |
| `Dependents`, `PaperlessBilling` | Binary indicator (Yes = 1) |
| Add-on services (4 columns) | Binary indicator (Yes = 1; No and *No internet service* = 0) |
| `InternetService`, `Contract` | One-hot (reference: DSL and Month-to-month) |
| `tenure` | `log(1 + tenure)`, standardised |
| `MonthlyCharges` | Standardised |
| Feature set | All ten original features (no removal improved cross-validated performance meaningfully) |

The cleaned and split datasets are exported to `data/processed/` by notebook `00`; `data/processed/split_metadata.json` records the split settings and file hashes. Those files are not model-ready (categories are still text); the pipeline performs the encoding.

## 7. Methodology

- **Split.** One stratified 80/20 split (3,944 training and 986 test customers, `random_state = 34`).
- **Validation.** Stratified 5-fold nested cross-validation on the training set: an inner grid search tunes hyper-parameters (scored by PR-AUC), and the outer folds produce out-of-fold (OOF) predictions for every training customer.
- **Metrics.** PR-AUC (primary, threshold-independent), ROC-AUC and Brier score; accuracy, precision, recall and F1 depend on the threshold.
- **Leakage control.** EDA, feature decisions, tuning, model selection and threshold selection used the training set only. The test set was scored once for the final evaluation in notebook `09`.
- **Known deviations.** The preprocessing choices of notebook `00` were made with ordinary cross-validation on the same training folds later used as outer folds. A fully nested re-check for Logistic Regression (notebook `08`, Section 6) changed mean PR-AUC from 0.666 to 0.665; the tree and boosting models were not re-checked. The test set was also scored by two earlier pipeline versions before the final evaluation; no decision used those scores, but the holdout is not perfectly fresh.

## 8. Models Evaluated

Logistic Regression, Decision Tree, Random Forest, HistGradientBoosting, XGBoost and LightGBM (notebooks `01`–`06`), with a most-frequent-class dummy as reference. Class weighting, random oversampling and SMOTENC were compared in notebook `07` for Logistic Regression and HistGradientBoosting.

**Nested cross-validation results (training set, mean ± SD over 5 outer folds):**

| Model | PR-AUC | ROC-AUC | Brier score |
|---|---|---|---|
| **Logistic Regression** | **0.666 ± 0.027** | 0.846 | 0.136 |
| XGBoost | 0.666 ± 0.021 | 0.846 | 0.136 |
| LightGBM | 0.665 ± 0.018 | 0.846 | 0.136 |
| Random Forest | 0.663 ± 0.024 | 0.845 | 0.137 |
| HistGradientBoosting | 0.657 ± 0.015 | 0.844 | 0.137 |
| Decision Tree | 0.623 ± 0.011 | 0.830 | 0.142 |
| Dummy (most frequent) | 0.267 | 0.500 | 0.267 |

Class weighting, random oversampling and SMOTENC changed PR-AUC by at most ±0.002, gave the same precision at matched recall and worsened calibration (Brier 0.136 to about 0.164). They were not adopted.

## 9. Model Selection

The selection rule was fixed in advance: choose the most interpretable model unless the model with the highest mean nested PR-AUC beats it by more than 0.02 in at least four of the five outer folds. Logistic Regression has the highest mean PR-AUC, so it is selected (`C = 1`, L2 penalty, unweighted). Its differences from XGBoost, LightGBM and Random Forest are at most 0.003 PR-AUC, far below the fold-to-fold standard deviation, so the data do not distinguish these models. Logistic Regression is preferred for its readable coefficients and the absence of extra dependencies.

## 10. Threshold Selection

The model outputs a probability, and the threshold turns it into a contact decision. A threshold of 0.27 means "contact when the predicted probability is at least 27%"; it is not a statement about precision.

- The threshold minimises expected cost on the **nested OOF training predictions** under the illustrative cost model below.
- It is close to the theoretical break-even probability *c / (s·H)* = 1 / (0.3 × 12) = 0.278 for calibrated probabilities.
- On the same training OOF predictions, 0.27 corresponds to recall of about 0.81.
- The test set played no part in the choice.

**Cost-model assumptions.** All three are assumptions of the project team, not measured values or industry benchmarks, and are defined in `src/evaluation.py`:

| Quantity | Assumed value |
|---|---|
| Revenue horizon *H* (months of revenue lost when a customer churns) | 12 |
| Retention success rate *s* (share of contacted churners retained by the offer) | 30% |
| Offer cost *c* (months of the customer's own charges) | 1 |

Only the monthly charge comes from the data.

## 11. Model Evaluation

Three kinds of evidence are kept apart.

**(a) Cross-validation (nested, training set).** See the table in [Section 8](#8-models-evaluated). Logistic Regression: PR-AUC 0.666 ± 0.027, ROC-AUC 0.846 ± 0.015, Brier score 0.136 ± 0.005.

**(b) Out-of-fold predictions at threshold 0.27 (training set, 3,944 customers, pooled).** Precision 0.519, recall 0.806, F1 0.631, ROC-AUC 0.844, PR-AUC 0.658.

**(c) Final internal holdout (986 test customers, scored once, threshold 0.27).** The evaluated model is the Logistic Regression pipeline fitted on the training set only.

| Metric | Test | 95% bootstrap interval |
|---|---|---|
| Precision | 0.516 | 0.469 – 0.563 |
| Recall | 0.810 | 0.764 – 0.859 |
| F1-score | 0.630 | 0.589 – 0.673 |
| ROC-AUC | 0.854 | 0.830 – 0.878 |
| PR-AUC | 0.685 | 0.626 – 0.741 |
| Brier score | 0.132 | 0.121 – 0.143 |
| Accuracy | 0.747 | 0.721 – 0.774 |

Confusion matrix (threshold 0.27): TP 213, FN 50, FP 200, TN 523. The model flags 413 customers (42%), and 52% of them are churners, against a base rate of 27%. The intervals describe sampling uncertainty of one 986-customer sample only.

**Targets.** Recall (0.810) and ROC-AUC (0.854) meet their targets (the recall interval includes values below 0.80). Precision (0.516 versus 0.60) and F1 (0.630 versus 0.65) do not. These are properties of the precision–recall trade-off rather than of the threshold: on the training OOF predictions, the best precision at recall ≥ 0.80 is 0.522, keeping precision ≥ 0.60 limits recall to 0.649, and the maximum F1 at any threshold is 0.637 (0.635 on the 0.01 threshold grid; the six models range from 0.625 to 0.645). No threshold of any of the six models satisfies the precision and F1 targets together (notebook `08`, Section 5.4; `reports/threshold_criteria_check.csv` covers Logistic Regression only). When flagging the same 41.5% of training customers, Random Forest, HistGradientBoosting, XGBoost and LightGBM also flag 91–94% of Logistic Regression's false positives, which points to the information in the ten features rather than to the choice of algorithm. These are training-set (OOF) analyses, not test-set results.

## 12. Business Interpretation

The figures below are **illustrative cost estimates** on the 986 test customers under the assumptions listed in [Section 10](#10-threshold-selection). They measure expected revenue at risk plus offer spend, not profit, and they are not realised savings.

| Strategy | Customers contacted | Churners reached | Expected total cost |
|---|---|---|---|
| Do nothing | 0 | 0 of 263 | 236,717 |
| Contact everyone | 986 | 263 of 263 | 230,637 |
| Model, threshold 0.27 | 413 | 213 of 263 | 209,177 |

Under these assumptions the model strategy has the lowest expected cost, 11.6% below doing nothing. The ordering depends on the assumptions: in the sensitivity grid of notebook `09` the model strategy is cheapest in 11 of 20 combinations of success rate and offer cost, and "do nothing" is cheapest when offers are expensive or ineffective. Nothing in this project shows that contacting the flagged customers would prevent churn; that requires a randomised pilot with a control group.

Segment results in notebook `09` (for example, recall of 0.91 for month-to-month customers and 1 of 23 one-year churners flagged) are descriptive findings on the test set. They were examined after the model and threshold were fixed and are not validated targeting rules. Any actions suggested for particular segments are hypotheses to test.

## 13. Reproducibility Notes

- `RANDOM_STATE = 34` is used for the split, the cross-validation splitters, the models, permutation importance and the bootstrap.
- Notebooks `00`–`09` were executed in a fresh kernel by the authors; their saved outputs come from that run.
- The deployed model `models/churn_pipeline.pkl` is refitted on all 4,930 labelled customers with the hyper-parameters and threshold fixed on the training set. It therefore has no unseen data left, and **its performance on `data_test.csv` must not be reported as a test result**. The test figures above belong to the training-only pipeline.

## 14. Project Limitations

- **No temporal information.** The data contain no timestamps, observation window or churn dates, and `tenure` is measured together with the churn label. The project therefore cannot establish that the model predicts churn *before* it occurs. Prospective prediction requires temporal validation on data collected with a defined prediction date and outcome window. The model should be read as a classifier of churn status from historical customer attributes.
- **No customer identifier.** Identical feature profiles may or may not be the same customer.
- **Internal validation only.** One split of one dataset; no external validation. The test set was scored by earlier pipeline versions before the final evaluation.
- **Targets not met.** Precision and F1 targets were not achieved ([Section 11](#11-model-evaluation)).
- **Limited features.** Ten attributes, with no usage, complaint, payment or contract-end data. Six algorithms reach the same ceiling (PR-AUC about 0.665). Whether additional behavioural data would help has not been tested.
- **Preprocessing selection not fully nested** for the tree and boosting models ([Section 7](#7-methodology)).
- **Weak detection among committed customers.** The model rarely flags churners on one- or two-year contracts.
- **Business figures** rest on assumed parameters and measure revenue at risk, not profit.
- **Data origin and licence.** The source company, period and licence of the dataset are undocumented. No `LICENSE` file is included because the redistribution terms of the data have not been verified.

## 15. Repository Structure

```text
.
├── README.md
├── notes.md                         Original business-understanding notes (Indonesian)
├── pyproject.toml                   Makes src/ installable (pip install -e .)
├── requirements.txt                 Full environment for notebooks and tests
├── requirements-inference.txt       Minimal environment for loading the model
├── requirements-dev.txt             requirements.txt plus pytest and nbconvert
├── data/
│   ├── raw/
│   │   └── data_telco_customer_churn.csv
│   └── processed/
│       ├── data_cleaned.csv
│       ├── data_train.csv
│       ├── data_test.csv
│       └── split_metadata.json
├── notebooks/
│   ├── 00_data_understanding_and_preprocessing.ipynb
│   ├── 01_logistic_regression.ipynb
│   ├── 02_decision_tree.ipynb
│   ├── 03_random_forest.ipynb
│   ├── 04_hist_gradient_boosting.ipynb
│   ├── 05_xgboost.ipynb
│   ├── 06_lightgbm.ipynb
│   ├── 07_smote_comparison.ipynb
│   ├── 08_model_comparison_and_selection.ipynb
│   └── 09_final_evaluation_and_business_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── evaluation.py
│   ├── feature_engineering.py
│   ├── inference.py
│   └── visualization.py
├── models/
│   ├── churn_pipeline.pkl
│   └── model_metadata.json
├── reports/
│   ├── experiment_comparison.csv
│   ├── model_selection.json
│   ├── threshold_criteria_check.csv
│   ├── cv_results/                  Saved nested-CV results of notebooks 01-07
│   └── figures/                     Figures exported by the notebooks
└── tests/
```

Links: [notebooks](notebooks/), [model metadata](models/model_metadata.json), [planning notes](notes.md).

## 16. Installation and Environment Setup

Supported Python: **3.13 or 3.14** (the model and notebooks were produced with Python 3.14.5; the test suite was also run on 3.13). The scikit-learn version must be exactly the one in `requirements.txt`, because pickled scikit-learn models are not guaranteed to load across versions.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt  # notebooks, tests and inference
pip install -e .                     # makes the src package importable from anywhere
```

For model inference only, `pip install -r requirements-inference.txt` and `pip install -e .` are sufficient.

## 17. Running the Project

Run the notebooks in order from the repository root (each locates the project root itself):

1. `00` (independent). 2. `01`–`07` (independent of each other; each saves results to `reports/cv_results/`). 3. `08` (reads `01`–`07`). 4. `09` (reads the selection and **rewrites `models/`**).

Notebook `09` overwrites files in `models/` and `reports/`; work on a copy if the committed results should be kept. Notebooks `00`–`09` need about 7 minutes on a 16-core machine.

Run the tests with:

```bash
python -m pytest
```

The tests use synthetic customers; the inference tests load `models/churn_pipeline.pkl` in a fresh Python process.

## 18. Model Inference Example

```python
import pandas as pd
from src.inference import load_model, predict_churn

model = load_model()   # models/churn_pipeline.pkl (unpickling can execute code: trust only this repository's file)

customers = pd.DataFrame([
    {"Dependents": "No", "tenure": 2, "OnlineSecurity": "No", "OnlineBackup": "No",
     "InternetService": "Fiber optic", "DeviceProtection": "No", "TechSupport": "No",
     "Contract": "Month-to-month", "PaperlessBilling": "Yes", "MonthlyCharges": 85.5},
    {"Dependents": "Yes", "tenure": 60, "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
     "InternetService": "DSL", "DeviceProtection": "Yes", "TechSupport": "Yes",
     "Contract": "Two year", "PaperlessBilling": "No", "MonthlyCharges": 70.0},
])
print(predict_churn(model, customers).round(3))
#    churn_probability  flagged
# 0              0.824        1
# 1              0.010        0
```

The example customers are invented. `flagged` is 1 when the probability is at least 0.27, which is what `model.predict` returns. `predict_churn` ignores extra columns and column order, and raises `ValueError` for missing columns, missing values or unseen categories. `load_model()` locates the repository root from the current directory; from elsewhere, pass the path: `load_model("path/to/models/churn_pipeline.pkl")`.

The artifact is a scikit-learn `FixedThresholdClassifier` (threshold 0.27) wrapping the preprocessing and Logistic Regression pipeline. It references `encode_addons` and `_log_tenure_name` in `src/data_preprocessing.py`, so `src` must be importable (`pip install -e .`, or run from the repository root); those two function names must not be renamed. The probabilities are estimates of churn status given the attributes, not forecasts for a future date.

## 19. Dependencies

| Package | Version | Used for |
|---|---|---|
| numpy, pandas, scipy | 2.4.6, 3.0.5, 1.18.1 | Data handling |
| scikit-learn | 1.9.1 | Pipelines, models, validation (must match the pickle) |
| joblib | 1.6.0 | Saved cross-validation results |
| imbalanced-learn | 0.14.2 | SMOTENC and oversampling (notebook `07`) |
| xgboost, lightgbm | 3.4.1, 4.7.0 | Boosting models (notebooks `05`, `06`) |
| pyarrow | 20 or later | Loading the saved `reports/cv_results/*.joblib` files |
| matplotlib, seaborn | 3.10 or later, 0.13 or later | Figures |
| ipykernel | 7 or later | Running the notebooks |
| pytest, nbformat, nbconvert | development only | Tests and command-line notebook execution |

## 20. Acknowledgements

The dataset and the business brief were provided by the Module 3 course. Pickle files can execute arbitrary code when loaded; load `models/churn_pipeline.pkl` and `reports/cv_results/*.joblib` only from this repository.
