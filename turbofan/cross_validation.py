"""Motor bazında K-katlı çapraz doğrulama."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from turbofan.data.schema import UNIT
from turbofan.evaluation.metrics import Metrics, evaluate
from turbofan.features.pipeline import FeaturePipeline, default_feature_pipeline
from turbofan.features.target import RULTargetBuilder
from turbofan.models.base import RULModel


@dataclass(frozen=True)
class CVResult:
    fold_metrics: list[Metrics]

    @property
    def rmse_mean(self) -> float:
        return float(np.mean([m.rmse for m in self.fold_metrics]))

    @property
    def rmse_std(self) -> float:
        return float(np.std([m.rmse for m in self.fold_metrics]))

    @property
    def mae_mean(self) -> float:
        return float(np.mean([m.mae for m in self.fold_metrics]))


class CrossValidator:
    def __init__(
        self,
        model_factory: Callable[[], RULModel],
        feature_factory: Callable[[], FeaturePipeline] = default_feature_pipeline,
        rul_cap: int = 125,
        n_splits: int = 5,
        seed: int = 42,
    ) -> None:
        self._model_factory = model_factory
        self._feature_factory = feature_factory
        self._rul_cap = rul_cap
        self._n_splits = n_splits
        self._seed = seed

    def run(self, train: pd.DataFrame) -> CVResult:
        rng = np.random.default_rng(self._seed)
        units = rng.permutation(train[UNIT].unique())
        target = RULTargetBuilder(self._rul_cap)

        results = []
        for val_units in np.array_split(units, self._n_splits):
            mask = train[UNIT].isin(val_units)
            tr, va = train[~mask], train[mask]

            features = self._feature_factory()  # her katta taze özellik zinciri
            X_tr = FeaturePipeline.model_inputs(features.fit_transform(tr))
            X_va = FeaturePipeline.model_inputs(features.transform(va))

            model = self._model_factory()  # her katta taze model
            model.fit(X_tr, target.build(tr))
            results.append(evaluate(target.build(va), model.predict(X_va)))
        return CVResult(results)