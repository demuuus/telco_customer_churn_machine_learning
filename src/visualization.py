"""Plotting helpers shared by the notebooks, and export of key figures to reports/figures/."""
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, brier_score_loss, confusion_matrix

from src.data_preprocessing import find_project_root

PALETTE = {0: "#4C72B0", 1: "#DD8452"}  # 0 = No churn (blue), 1 = Churn (orange)
CHURN_LABELS = {0: "No churn", 1: "Churn"}


def set_style():
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 100, "axes.titleweight": "bold", "axes.titlesize": 12})


def save_figure(fig, name: str, root=None) -> Path:
    """Save a figure to reports/figures/<name>.png (used by the presentation)."""
    folder = (find_project_root() if root is None else Path(root)) / "reports" / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def plot_confusion(ax, y_true, pred, title):
    ConfusionMatrixDisplay(confusion_matrix(y_true, pred, labels=[0, 1]), display_labels=["No churn", "Churn"]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")


def plot_reliability(ax, y_true, probas: dict, title):
    """Reliability diagram (decile bins) for one or several sets of predicted probabilities."""
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    for name, p in probas.items():
        frac_pos, mean_pred = calibration_curve(y_true, p, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=f"{name} (Brier {brier_score_loss(y_true, p):.3f})")
    ax.set_title(title)
    ax.set_xlabel("Mean predicted churn probability (decile bins)")
    ax.set_ylabel("Observed churn rate")
    ax.legend(fontsize=8)


def plot_permutation_importance(ax, importance, title):
    """Horizontal bar chart of mean ± std permutation importance over outer folds."""
    summary = importance.groupby("feature")["importance"].agg(["mean", "std"]).sort_values("mean")
    ax.barh(summary.index, summary["mean"], xerr=summary["std"], color="#8DA0CB", capsize=3)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Mean drop in PR-AUC when the feature is shuffled")
    ax.set_ylabel("Feature")
    return summary


def plot_fold_scores(ax, fold_scores, title, highlight=None):
    """One grey line per outer fold across models, plus the mean."""
    names = list(fold_scores.columns)
    for fold in fold_scores.index:
        ax.plot(names, fold_scores.loc[fold], marker="o", color="gray", alpha=0.5, linewidth=1)
    ax.plot(names, fold_scores.mean(), marker="D", color="#DD8452", linewidth=2.5, label="Mean of 5 outer folds")
    if highlight is not None:
        ax.scatter([highlight], [fold_scores[highlight].mean()], s=180, facecolors="none", edgecolors="black",
                   linewidths=2, zorder=5, label=f"Selected: {highlight}")
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel("PR-AUC on the outer validation fold")
    ax.tick_params(axis="x", rotation=15)
    ax.legend()
