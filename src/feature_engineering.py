"""Candidate engineered features and feature-analysis statistics (notebook 00).

All candidate features are computed row by row from the customer's own values; nothing is
estimated from the data, so they cannot leak information between cross-validation folds.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.data_preprocessing import ADDON_FEATURES, TENURE_GROUP_BINS, TENURE_GROUP_LABELS, encode_addons


def engineer_features(frame):
    """Add the candidate features evaluated in notebook 00 (Section 9)."""
    out = pd.DataFrame(frame).copy()
    is_month_to_month = out["Contract"].eq("Month-to-month").astype(int)
    out["mtm_fiber"] = is_month_to_month * out["InternetService"].eq("Fiber optic").astype(int)
    out["tenure_x_mtm"] = out["tenure"] * is_month_to_month
    out["n_addon_services"] = encode_addons(out[ADDON_FEATURES]).sum(axis=1)
    out["tenure_group"] = pd.cut(out["tenure"], bins=TENURE_GROUP_BINS, labels=TENURE_GROUP_LABELS).astype(str)
    return out


def vif_table(design: pd.DataFrame) -> pd.Series:
    """Variance Inflation Factor: 1 / (1 - R^2) of each column regressed on all other columns."""
    vifs = {}
    for col in design.columns:
        others = design.drop(columns=col)
        r2 = LinearRegression().fit(others, design[col]).score(others, design[col])
        vifs[col] = np.inf if r2 >= 1 - 1e-10 else 1 / (1 - r2)
    return pd.Series(vifs, name="VIF")


def correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    """Eta: square root of the share of variance in ``values`` explained by the category means (0-1)."""
    overall_mean = values.mean()
    grouped = values.groupby(categories, observed=True)
    between = sum(len(g) * (g.mean() - overall_mean) ** 2 for _, g in grouped)
    total = ((values - overall_mean) ** 2).sum()
    return float(np.sqrt(between / total))
