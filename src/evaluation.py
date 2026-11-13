"""Cross-validation, nested CV, metrics, business cost model, selection rule and result storage."""
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix, f1_score,
                             log_loss, make_scorer, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate

from src.data_preprocessing import N_SPLITS, RANDOM_STATE, build_pipeline, find_project_root, get_cv

# ----------------------------------------------------------------------------------- metrics
SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}
METRIC_LABELS = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1",
                 "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "brier": "Brier score", "log_loss": "Log loss"}
REPORT_METRICS = ["pr_auc", "roc_auc", "brier", "accuracy", "precision", "recall", "f1"]


def fold_metrics(y_true, proba, threshold=0.5) -> dict:
    """Threshold-independent (PR-AUC, ROC-AUC, Brier, log loss) and threshold-dependent metrics."""
    pred = (np.asarray(proba) >= threshold).astype(int)
    return {"accuracy": accuracy_score(y_true, pred), "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred), "f1": f1_score(y_true, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, proba), "pr_auc": average_precision_score(y_true, proba),
            "brier": brier_score_loss(y_true, proba), "log_loss": log_loss(y_true, proba)}


def cv_scores(estimator, X, y, scoring=None, cv=None) -> dict:
    """cross_validate on the shared folds, with training-fold scores for overfitting diagnostics."""
    return cross_validate(estimator, X, y, cv=get_cv() if cv is None else cv, scoring=SCORING if scoring is None else scoring,
                          return_train_score=True, n_jobs=-1)


def summarise_cv(results: dict) -> dict:
    """Mean/std of validation scores and mean training score for each metric in ``results``."""
    out = {}
    for key in results:
        if key.startswith("test_"):
            m = key[len("test_"):]
            out[f"{METRIC_LABELS[m]} mean"] = results[key].mean()
            out[f"{METRIC_LABELS[m]} std"] = results[key].std()
            out[f"{METRIC_LABELS[m]} train"] = results[f"train_{m}"].mean()
    return out


def mean_std_table(summary: pd.DataFrame, metrics=("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")):
    """Format 'mean ± std' strings for display."""
    return pd.DataFrame({METRIC_LABELS[m]: summary[f"{METRIC_LABELS[m]} mean"].map("{:.3f}".format) + " ± "
                         + summary[f"{METRIC_LABELS[m]} std"].map("{:.3f}".format) for m in metrics})


# ----------------------------------------------------------------------------------- nested CV
def run_nested_cv(estimator, grid, X, y, prep_config, sampler=None, compute_importance=False, scoring="average_precision"):
    """Nested cross-validation on the training set.

    Outer loop: the shared 5-fold splitter. For each outer fold, a GridSearchCV with its own inner
    5-fold CV is run on the outer-training part ONLY; the selected configuration (refitted on that part)
    predicts the outer validation fold, whose labels played no role in tuning. The concatenated
    outer-fold probabilities are honest out-of-fold (OOF) predictions for every training customer.
    """
    oof = np.full(len(y), np.nan)
    fold_rows, chosen, importance_rows = [], [], []
    start = time.perf_counter()
    for fold, (tr_idx, va_idx) in enumerate(get_cv().split(X, y), start=1):
        X_tr, X_va, y_tr, y_va = X.iloc[tr_idx], X.iloc[va_idx], y.iloc[tr_idx], y.iloc[va_idx]
        inner_cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        search = GridSearchCV(build_pipeline(estimator, sampler=sampler, **prep_config), grid, scoring=scoring,
                              cv=inner_cv, n_jobs=-1)
        search.fit(X_tr, y_tr)
        proba = search.predict_proba(X_va)[:, 1]
        oof[va_idx] = proba
        fold_rows.append({"fold": fold, **fold_metrics(y_va, proba)})
        chosen.append({k.replace("model__", ""): v for k, v in search.best_params_.items()})
        if compute_importance:
            result = permutation_importance(search.best_estimator_, X_va, y_va, scoring="average_precision",
                                            n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
            importance_rows += [{"fold": fold, "feature": f, "importance": v}
                                for f, v in zip(X.columns, result.importances_mean)]
    if np.isnan(oof).any():
        raise RuntimeError("Every training row must receive exactly one OOF prediction.")
    return {"oof": oof, "folds": pd.DataFrame(fold_rows).set_index("fold"), "params": chosen,
            "importance": pd.DataFrame(importance_rows), "seconds": time.perf_counter() - start}


def final_search(estimator, grid, X, y, prep_config, sampler=None, scoring="average_precision") -> GridSearchCV:
    """The same grid search, run once on the full training set (its refit pipeline is the final candidate)."""
    search = GridSearchCV(build_pipeline(estimator, sampler=sampler, **prep_config), grid, scoring=scoring, cv=get_cv(),
                          n_jobs=-1, return_train_score=True)
    return search.fit(X, y)


def nested_summary_row(nested: dict) -> dict:
    """'mean ± std' over the outer folds for the standard report metrics."""
    f = nested["folds"]
    return {METRIC_LABELS[m]: f"{f[m].mean():.3f} ± {f[m].std(ddof=0):.3f}" for m in REPORT_METRICS}


# ----------------------------------------------------------------------------------- result storage
def results_dir(root=None) -> Path:
    path = (find_project_root() if root is None else Path(root)) / "reports" / "cv_results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_results(slug: str, payload: dict, root=None) -> Path:
    """Persist a notebook's training-stage results for the comparison notebook (08)."""
    path = results_dir(root) / f"{slug}.joblib"
    joblib.dump(payload, path)
    return path


def load_results(slug: str, root=None) -> dict:
    path = results_dir(root) / f"{slug}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run the notebook that produces '{slug}' first.")
    return joblib.load(path)


# ----------------------------------------------------------------------------------- model selection
def apply_selection_rule(fold_scores: pd.DataFrame, interpretability_order, margin=0.02, min_folds=4):
    """Pre-specified rule: the most interpretable model is selected unless the model with the highest mean
    nested PR-AUC beats it by more than ``margin`` AND in at least ``min_folds`` of the outer folds."""
    means = fold_scores.mean()
    reference = means.idxmax()
    rows, selected = [], None
    for name in interpretability_order:
        diff = fold_scores[reference] - fold_scores[name]
        clearly_worse = bool(diff.mean() > margin and (diff > 0).sum() >= min_folds)
        rows.append({"model": name, "mean nested PR-AUC": means[name], f"gap to best ({reference})": diff.mean(),
                     "folds where best is higher": int((diff > 0).sum()), "clearly worse than best?": clearly_worse})
        if selected is None and not clearly_worse:
            selected = name
    return selected, reference, pd.DataFrame(rows).set_index("model")


# ----------------------------------------------------------------------------------- business cost model
H_MONTHS = 12        # ILLUSTRATIVE ASSUMPTION: months of revenue lost when a customer churns
SUCCESS_RATE = 0.30  # ILLUSTRATIVE ASSUMPTION: share of contacted churners retained by the offer
OFFER_MONTHS = 1.0   # ILLUSTRATIVE ASSUMPTION: offer cost in months of the customer's own charges
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.01), 2)


def business_outcome(y_true, y_pred, monthly, horizon=H_MONTHS, success=SUCCESS_RATE, offer_months=OFFER_MONTHS):
    """Expected revenue-at-risk accounting for one contact strategy (illustrative assumptions, not profit)."""
    y_true, y_pred, monthly = np.asarray(y_true), np.asarray(y_pred), np.asarray(monthly, dtype=float)
    revenue_horizon = horizon * monthly
    contacted, churner = y_pred == 1, y_true == 1
    offer_spend = (offer_months * monthly)[contacted].sum()
    revenue_at_risk = revenue_horizon[churner].sum()
    revenue_retained = success * revenue_horizon[churner & contacted].sum()
    revenue_lost = revenue_at_risk - revenue_retained
    return {"customers contacted": int(contacted.sum()), "offer spend": offer_spend,
            "revenue at risk (churners)": revenue_at_risk, "expected revenue retained": revenue_retained,
            "expected revenue lost": revenue_lost, "total expected cost": offer_spend + revenue_lost}


def threshold_table(y_true, proba, monthly, thresholds=THRESHOLDS, **cost_kwargs) -> pd.DataFrame:
    """Classification metrics, confusion counts and expected cost for each threshold."""
    rows = []
    for t in thresholds:
        pred = (np.asarray(proba) >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        rows.append({"threshold": t, "accuracy": accuracy_score(y_true, pred),
                     # precision is undefined (NaN) when nobody is flagged, rather than plotted as 0
                     "precision": precision_score(y_true, pred) if pred.sum() else np.nan,
                     "recall": recall_score(y_true, pred), "f1": f1_score(y_true, pred, zero_division=0),
                     "TP": tp, "FP": fp, "FN": fn, "TN": tn, **business_outcome(y_true, pred, monthly, **cost_kwargs)})
    return pd.DataFrame(rows).set_index("threshold")


def pick_threshold(tbl: pd.DataFrame, column: str, largest=True, mask=None):
    """Threshold that maximises (or minimises) ``column``, optionally among rows satisfying ``mask``."""
    sub = tbl if mask is None else tbl[mask]
    if sub.empty:
        return np.nan
    return sub[column].idxmax() if largest else sub[column].idxmin()


# ----------------------------------------------------------------------------------- final evaluation helpers
def classification_metrics(y_true, pred, proba) -> dict:
    return {"Accuracy": accuracy_score(y_true, pred), "Precision": precision_score(y_true, pred, zero_division=0),
            "Recall": recall_score(y_true, pred), "F1": f1_score(y_true, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_true, proba), "PR-AUC": average_precision_score(y_true, proba),
            "Brier score": brier_score_loss(y_true, proba)}


def bootstrap_metrics(y_true, pred, proba, n_boot=1000, seed=RANDOM_STATE) -> pd.DataFrame:
    """Resample customers with replacement and recompute every metric (sampling uncertainty only)."""
    y_true, pred, proba = np.asarray(y_true), np.asarray(pred), np.asarray(proba)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        rows.append(classification_metrics(y_true[idx], pred[idx], proba[idx]))
    return pd.DataFrame(rows)


def segment_performance(frame: pd.DataFrame, by: str) -> pd.DataFrame:
    """Size, churners, churn rate, share flagged, precision, recall and accuracy per segment.
    ``frame`` needs integer columns 'actual' and 'pred'."""
    def stats_for(g):
        return pd.Series({
            "n": len(g), "churners": int(g["actual"].sum()), "churn rate": g["actual"].mean(),
            "flagged (%)": g["pred"].mean() * 100,
            "precision": precision_score(g["actual"], g["pred"], zero_division=0) if g["pred"].sum() else np.nan,
            "recall": recall_score(g["actual"], g["pred"], zero_division=0) if g["actual"].sum() else np.nan,
            "accuracy": accuracy_score(g["actual"], g["pred"]),
        })
    return frame.groupby(by, observed=True)[["actual", "pred"]].apply(stats_for)
