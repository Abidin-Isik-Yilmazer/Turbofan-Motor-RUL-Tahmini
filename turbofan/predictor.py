"""Eğitilmiş özellik zinciri + modeli tek pakette taşıyan, kaydedilebilir tahminci."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from turbofan.data.schema import CYCLE, UNIT
from turbofan.features.pipeline import FeaturePipeline, default_feature_pipeline
from turbofan.features.target import RULTargetBuilder
from turbofan.models.base import RULModel


class RULPredictor:
    def __init__(self, features: FeaturePipeline, model: RULModel) -> None:
        self._features = features
        self._model = model

    @classmethod
    def train(
        cls,
        train_df: pd.DataFrame,
        model: RULModel,
        features: FeaturePipeline | None = None,
        rul_cap: int = 125,
    ) -> "RULPredictor":
        features = features or default_feature_pipeline()
        X = FeaturePipeline.model_inputs(features.fit_transform(train_df))
        y = RULTargetBuilder(rul_cap).build(train_df)
        model.fit(X, y)
        return cls(features, model)

    def predict_last(self, df: pd.DataFrame) -> pd.Series:
        """Her motorun SON kaydı için kalan ömür tahmini (index = motor numarası)."""
        last = (
            self._features.transform(df).sort_values([UNIT, CYCLE]).groupby(UNIT).tail(1)
        )
        pred = self._model.predict(FeaturePipeline.model_inputs(last))
        return pd.Series(pred, index=pd.Index(last[UNIT].to_numpy(), name=UNIT), name="RUL_pred")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "RULPredictor":
        return joblib.load(path)