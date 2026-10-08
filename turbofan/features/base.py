"""Özellik dönüştürücü soyutlaması."""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class FeatureTransformer(ABC):
    """DataFrame -> DataFrame dönüşümü. `fit` yalnızca eğitim verisini görmelidir."""

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "FeatureTransformer":
        ...

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ...

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)