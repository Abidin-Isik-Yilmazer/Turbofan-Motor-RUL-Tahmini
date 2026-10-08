"""Somut özellik dönüştürücüler."""
from __future__ import annotations

import numpy as np
import pandas as pd

from turbofan.data.schema import SENSOR_COLS, UNIT
from turbofan.features.base import FeatureTransformer


class ConstantSensorDropper(FeatureTransformer):
    """Eğitim setinde (neredeyse) sabit olan sensörleri eler."""

    def __init__(self, variance_threshold: float = 1e-8) -> None:
        self._threshold = variance_threshold
        self._dropped: list[str] = []

    @property
    def dropped(self) -> list[str]:
        return list(self._dropped)

    def fit(self, df: pd.DataFrame) -> "ConstantSensorDropper":
        variances = df[SENSOR_COLS].var()
        self._dropped = variances[variances < self._threshold].index.tolist()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=self._dropped)


class SensorScaler(FeatureTransformer):
    """Sensörleri eğitim setinin ortalama/std'sine göre standartlaştırır."""

    def __init__(self) -> None:
        self._cols: list[str] = []
        self._mean: pd.Series | None = None
        self._std: pd.Series | None = None

    def fit(self, df: pd.DataFrame) -> "SensorScaler":
        self._cols = [c for c in SENSOR_COLS if c in df.columns]
        self._mean = df[self._cols].mean()
        self._std = df[self._cols].std(ddof=0).replace(0.0, 1.0)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[self._cols] = (df[self._cols] - self._mean) / self._std
        return out


class RollingMeanFeatures(FeatureTransformer):
    """Her motor için sensörlerin kayan ortalamasını ekler (gürültüyü azaltır).

    Hesap motor bazında (groupby unit) yapılır: bir motorun geçmişi diğerine sızmaz.
    Veri, motor içinde döngü sırasına göre sıralı varsayılır.
    """

    def __init__(self, window: int = 5) -> None:
        self._window = window
        self._cols: list[str] = []

    def fit(self, df: pd.DataFrame) -> "RollingMeanFeatures":
        self._cols = [c for c in SENSOR_COLS if c in df.columns]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        rolled = (
            df.groupby(UNIT)[self._cols]
            .rolling(self._window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
            .add_suffix(f"_mean{self._window}")
        )
        return pd.concat([df, rolled.loc[df.index]], axis=1)


def _slope(y: np.ndarray) -> float:
    """En küçük kareler doğrusunun eğimi (x = 0, 1, 2, ...)."""
    x = np.arange(len(y), dtype=float)
    x -= x.mean()
    return float(np.dot(x, y - y.mean()) / np.dot(x, x))


class RollingSlopeFeatures(FeatureTransformer):
    """Her motor için sensörlerin son `window` döngüdeki eğimini (trendini) ekler.

    Motorun ilk döngüsünde eğim hesaplanamaz (tek nokta), 0 kabul edilir.
    """

    def __init__(self, window: int = 10) -> None:
        self._window = window
        self._cols: list[str] = []

    def fit(self, df: pd.DataFrame) -> "RollingSlopeFeatures":
        self._cols = [c for c in SENSOR_COLS if c in df.columns]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        slopes = (
            df.groupby(UNIT)[self._cols]
            .rolling(self._window, min_periods=2)
            .apply(_slope, raw=True)
            .reset_index(level=0, drop=True)
            .fillna(0.0)
            .add_suffix(f"_slope{self._window}")
        )
        return pd.concat([df, slopes.loc[df.index]], axis=1)