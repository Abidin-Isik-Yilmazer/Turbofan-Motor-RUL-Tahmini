"""Model soyutlaması: pipeline hangi algoritmanın kullanıldığını bilmez."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class RULModel(ABC):
    name: str

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RULModel":
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...


class SklearnRULModel(RULModel):
    """Herhangi bir scikit-learn regresörünü RULModel arayüzüne uyarlar (Adapter)."""

    def __init__(self, name: str, estimator: BaseEstimator) -> None:
        self.name = name
        self._estimator = estimator

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SklearnRULModel":
        self._estimator.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(self._estimator.predict(X), 0.0, None)  # RUL negatif olamaz