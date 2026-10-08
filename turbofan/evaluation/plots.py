"""Sonuç grafikleri."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def plot_predictions(y_true, y_pred, title: str = "") -> Figure:
    """Sol: gerçek vs tahmin. Sağ: hata (tahmin - gerçek) gerçek RUL'a göre."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    err = y_pred - y_true
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].scatter(y_true, y_pred, s=18, alpha=0.7)
    lim = max(y_true.max(), y_pred.max()) * 1.05
    axes[0].plot([0, lim], [0, lim], "r--", label="ideal")
    axes[0].set(xlabel="Gerçek RUL", ylabel="Tahmin RUL", title="Gerçek vs tahmin")
    axes[0].legend()

    axes[1].scatter(y_true, err, s=18, alpha=0.7)
    axes[1].axhline(0, color="r", linestyle="--")
    axes[1].set(xlabel="Gerçek RUL", ylabel="Hata (tahmin - gerçek)", title="Hata dağılımı")

    fig.suptitle(title)
    fig.tight_layout()
    return fig