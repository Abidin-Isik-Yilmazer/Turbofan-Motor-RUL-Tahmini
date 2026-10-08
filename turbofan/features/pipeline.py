"""Özellik dönüştürücüleri zincirleyen pipeline."""
from __future__ import annotations

from typing import Sequence

import pandas as pd

from turbofan.data.schema import INDEX_COLS, SETTING_COLS
from turbofan.features.base import FeatureTransformer
from turbofan.features.transformers import (
    ConstantSensorDropper,
    RollingMeanFeatures,
    RollingSlopeFeatures,
    SensorScaler,
)


class FeaturePipeline(FeatureTransformer):
    """Dönüştürücüleri sırayla uygular (Composite deseni)."""

    def __init__(self, steps: Sequence[FeatureTransformer]) -> None:
        self._steps = list(steps)

    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        for step in self._steps:
            df = step.fit_transform(df)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for step in self._steps:
            df = step.transform(df)
        return df

    @staticmethod
    def model_inputs(df: pd.DataFrame) -> pd.DataFrame:
        """Modele girecek kolonlar: indeks ve çalışma ayarı kolonları hariç hepsi."""
        drop = [c for c in INDEX_COLS + SETTING_COLS if c in df.columns]
        return df.drop(columns=drop)


def default_feature_pipeline(window: int = 5, slope_window: int | None = 50) -> FeaturePipeline:
    """Projenin standart özellik zinciri (slope_window=None: eğim özelliği yok)."""
    steps: list[FeatureTransformer] = [
        ConstantSensorDropper(),
        SensorScaler(),
        RollingMeanFeatures(window),
    ]
    if slope_window is not None:
        steps.append(RollingSlopeFeatures(slope_window))
    return FeaturePipeline(steps)