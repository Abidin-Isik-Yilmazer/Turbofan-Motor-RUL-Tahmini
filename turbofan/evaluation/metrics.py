"""Değerlendirme metrikleri."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Metrics:
    rmse: float
    mae: float
    nasa_score: float

    def __str__(self) -> str:
        return f"RMSE={self.rmse:.2f}  MAE={self.mae:.2f}  NASA score={self.nasa_score:.1f}"


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """C-MAPSS asimetrik skoru. Geç tahmin (d>0, arıza sonrası fark edilir) daha ağır cezalanır."""
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))


def evaluate(y_true, y_pred) -> Metrics:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    return Metrics(
        rmse=float(np.sqrt(np.mean(err**2))),
        mae=float(np.mean(np.abs(err))),
        nasa_score=nasa_score(y_true, y_pred),
    )