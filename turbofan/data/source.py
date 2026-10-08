"""Veri kaynakları. Projenin geri kalanı yalnızca DataSource soyutlamasını bilir."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from turbofan.data.schema import ALL_COLS, UNIT


@dataclass(frozen=True)
class EngineDataset:
    """Bir C-MAPSS alt kümesi (örn. FD001).

    train: motorların arızaya kadar tüm kayıtları.
    test: motorların arızadan önce kesilmiş kayıtları.
    test_rul: her test motorunun son kayıttaki gerçek RUL'u (index = unit).
    """

    train: pd.DataFrame
    test: pd.DataFrame
    test_rul: pd.Series


class DataSource(ABC):
    @abstractmethod
    def load(self) -> EngineDataset:
        ...


class CMAPSSFileSource(DataSource):
    """NASA'nın train_FD00X.txt / test_FD00X.txt / RUL_FD00X.txt dosyalarını okur."""

    def __init__(self, data_dir: Path, subset: str = "FD001") -> None:
        self._dir = Path(data_dir)
        self._subset = subset

    def load(self) -> EngineDataset:
        train = self._read(f"train_{self._subset}.txt")
        test = self._read(f"test_{self._subset}.txt")
        rul = pd.read_csv(self._path(f"RUL_{self._subset}.txt"), header=None, names=["RUL"])["RUL"]
        rul.index = pd.Index(sorted(test[UNIT].unique()), name=UNIT)
        return EngineDataset(train=train, test=test, test_rul=rul)

    def _path(self, name: str) -> Path:
        path = self._dir / name
        if not path.exists():
            raise FileNotFoundError(f"{path} bulunamadı.")
        return path

    def _read(self, name: str) -> pd.DataFrame:
        df = pd.read_csv(self._path(name), sep=r"\s+", header=None)
        df.columns = ALL_COLS
        return df