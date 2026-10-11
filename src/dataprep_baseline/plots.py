"""Plot helpers (headless Agg backend)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import confusion_matrix, roc_curve  # noqa: E402
from sklearn.preprocessing import label_binarize  # noqa: E402


def plot_confusion_matrices(models, y_test, class_names: list[str], path: Path) -> None:
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.5), squeeze=False)
    labels = list(range(len(class_names)))
    for ax, m in zip(axes[0], models, strict=False):
        cm = confusion_matrix(y_test, m.y_pred, labels=labels)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )
        ax.set(title=m.name, xlabel="Predicted", ylabel="Actual")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc_curves(models, y_test, class_names: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    binary = len(class_names) == 2
    y_bin = None if binary else label_binarize(y_test, classes=list(range(len(class_names))))
    for m in models:
        if binary:
            fpr, tpr, _ = roc_curve(y_test, m.y_proba[:, 1])
        else:
            fpr, tpr, _ = roc_curve(y_bin.ravel(), m.y_proba.ravel())
        ax.plot(fpr, tpr, label=f"{m.name} (AUC={m.test['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set(
        title="ROC curves" + ("" if binary else " (micro-average)"),
        xlabel="False positive rate",
        ylabel="True positive rate",
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_predicted_vs_actual(models, y_test, path: Path) -> None:
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.5), squeeze=False)
    lo, hi = float(np.min(y_test)), float(np.max(y_test))
    for ax, m in zip(axes[0], models, strict=False):
        ax.scatter(y_test, m.y_pred, s=12, alpha=0.6)
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set(title=f"{m.name} (RMSE={m.test['rmse']:.3f})", xlabel="Actual", ylabel="Predicted")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
