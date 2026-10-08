"""Uçtan uca akış: veri -> özellik -> model -> değerlendirme."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from turbofan.data.schema import CYCLE, UNIT
from turbofan.data.source import DataSource
from turbofan.data.split import split_by_unit
from turbofan.evaluation.metrics import Metrics, evaluate
from turbofan.features.pipeline import FeaturePipeline, default_feature_pipeline
from turbofan.features.target import RULTargetBuilder
from turbofan.models.base import RULModel


@dataclass(frozen=True)
class ExperimentResult:
    model_name: str
    val_metrics: Metrics
    test_metrics: Metrics
    y_true: np.ndarray
    y_pred: np.ndarray


class TrainingPipeline:
    def __init__(
        self,
        source: DataSource,
        model: RULModel,
        rul_cap: int = 125,
        window: int = 5,
        val_fraction: float = 0.2,
        seed: int = 42,
    ) -> None:
        self._source = source
        self._model = model
        self._rul_cap = rul_cap
        self._window = window
        self._val_fraction = val_fraction
        self._seed = seed

    def run(self) -> ExperimentResult:
        data = self._source.load()
        train_df, val_df = split_by_unit(data.train, self._val_fraction, self._seed)

        target = RULTargetBuilder(self._rul_cap)
        y_train, y_val = target.build(train_df), target.build(val_df)

        features = self._build_features()
        X_train = FeaturePipeline.model_inputs(features.fit_transform(train_df))
        X_val = FeaturePipeline.model_inputs(features.transform(val_df))

        self._model.fit(X_train, y_train)
        val_metrics = evaluate(y_val, self._model.predict(X_val))

        # Test: her motorun yalnızca son kaydı, RUL_FD00X.txt ile karşılaştırılır.
        test_last = (
            features.transform(data.test).sort_values([UNIT, CYCLE]).groupby(UNIT).tail(1)
        )
        y_true = data.test_rul.loc[test_last[UNIT]].to_numpy()
        y_pred = self._model.predict(FeaturePipeline.model_inputs(test_last))
        test_metrics = evaluate(y_true, y_pred)

        return ExperimentResult(self._model.name, val_metrics, test_metrics, y_true, y_pred)

    def _build_features(self) -> FeaturePipeline:
        return default_feature_pipeline(self._window)