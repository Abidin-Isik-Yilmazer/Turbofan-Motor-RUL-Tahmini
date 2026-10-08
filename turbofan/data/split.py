"""Eğitim/doğrulama ayrımı."""
from __future__ import annotations

import numpy as np
import pandas as pd

from turbofan.data.schema import UNIT


def split_by_unit(
    df: pd.DataFrame, val_fraction: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Motor bazında ayır: aynı motorun satırları hem train hem val'de olmamalı (sızıntı)."""
    rng = np.random.default_rng(seed)
    units = rng.permutation(df[UNIT].unique())
    n_val = max(1, int(len(units) * val_fraction))
    val_units = set(units[:n_val])
    mask = df[UNIT].isin(val_units)
    return df[~mask].copy(), df[mask].copy()