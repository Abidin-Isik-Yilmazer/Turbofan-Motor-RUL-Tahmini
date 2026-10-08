"""Hedef değişken (RUL) üretimi."""
from __future__ import annotations

import pandas as pd

from turbofan.data.schema import CYCLE, UNIT


class RULTargetBuilder:
    """Her satır için kalan ömür = (motorun son döngüsü - mevcut döngü).

    `cap` verilirse piecewise-linear hedef kullanılır: motor sağlıklıyken
    bozulma belirtisi olmadığından RUL, cap değerinde sabit tutulur.
    """

    def __init__(self, cap: int | None = 125) -> None:
        self._cap = cap

    def build(self, train: pd.DataFrame) -> pd.Series:
        last_cycle = train.groupby(UNIT)[CYCLE].transform("max")
        rul = last_cycle - train[CYCLE]
        if self._cap is not None:
            rul = rul.clip(upper=self._cap)
        return rul.rename("RUL")