import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from turbofan.cross_validation import CrossValidator
from turbofan.data.schema import ALL_COLS, SENSOR_COLS, UNIT
from turbofan.data.split import split_by_unit
from turbofan.evaluation.metrics import evaluate, nasa_score
from turbofan.features.pipeline import FeaturePipeline
from turbofan.features.target import RULTargetBuilder
from turbofan.features.transformers import (
    ConstantSensorDropper,
    RollingMeanFeatures,
    RollingSlopeFeatures,
    SensorScaler,
)
from turbofan.models.base import SklearnRULModel
from turbofan.predictor import RULPredictor


def make_fleet(n_units: int = 12, seed: int = 0) -> pd.DataFrame:
    """C-MAPSS formatında küçük sentetik filo: sensörler zamanla bozulur, s1 sabit."""
    rng = np.random.default_rng(seed)
    gains = rng.normal(0, 8, len(SENSOR_COLS))
    frames = []
    for unit in range(1, n_units + 1):
        life = int(rng.integers(60, 100))
        t = np.arange(1, life + 1)
        health = (t / life) ** 2
        sensors = 100 + np.outer(health, gains) + rng.normal(0, 1, (life, len(SENSOR_COLS)))
        sensors[:, 0] = 5.0  # s1: sabit sensör
        settings = rng.normal(0, 0.002, (life, 3))
        data = np.column_stack([np.full(life, unit), t, settings, sensors])
        frames.append(pd.DataFrame(data, columns=ALL_COLS))
    df = pd.concat(frames, ignore_index=True)
    df[["unit", "cycle"]] = df[["unit", "cycle"]].astype(int)
    return df


@pytest.fixture(scope="module")
def fleet() -> pd.DataFrame:
    return make_fleet()


# ---------- hedef ----------
def test_target_is_capped_and_zero_at_last_cycle(fleet):
    y = RULTargetBuilder(cap=50).build(fleet)
    assert y.max() == 50 and y.min() == 0
    last_rows = fleet.groupby("unit").tail(1).index
    assert (y.loc[last_rows] == 0).all()


# ---------- özellik dönüştürücüler ----------
def test_constant_sensor_dropper_removes_constant_sensor(fleet):
    dropper = ConstantSensorDropper().fit(fleet)
    assert dropper.dropped == ["s1"]
    assert "s1" not in dropper.transform(fleet).columns


def test_scaler_uses_training_statistics():
    train = pd.DataFrame({"unit": [1, 1], "cycle": [1, 2], "s2": [0.0, 10.0]})
    new = pd.DataFrame({"unit": [1], "cycle": [3], "s2": [15.0]})
    scaler = SensorScaler().fit(train)
    assert scaler.transform(train)["s2"].tolist() == [-1.0, 1.0]
    assert scaler.transform(new)["s2"].iloc[0] == 2.0  # (15-5)/5, test'in kendi std'si değil


def test_rolling_mean_does_not_leak_across_units():
    df = pd.DataFrame({"unit": [1, 1, 2, 2], "cycle": [1, 2, 1, 2],
                       "s2": [1.0, 1.0, 100.0, 100.0]})
    out = RollingMeanFeatures(window=2).fit_transform(df)
    assert out.loc[2, "s2_mean2"] == 100.0  # motor 2'nin ilk satırı motor 1'den etkilenmez


def test_rolling_slope_matches_known_line_and_does_not_leak():
    df = pd.DataFrame({"unit": [1] * 6 + [2] * 6,
                       "cycle": list(range(1, 7)) * 2,
                       "s2": [2.0 * c for c in range(1, 7)] + [50.0] * 6})
    out = RollingSlopeFeatures(window=5).fit_transform(df)
    assert out.loc[0, "s2_slope5"] == 0.0                    # tek nokta: eğim yok
    assert out.loc[5, "s2_slope5"] == pytest.approx(2.0)     # y = 2x doğrusunun eğimi
    assert out.loc[6:, "s2_slope5"].abs().max() < 1e-9       # motor 2 sabit, eğim 0


def test_model_inputs_drops_index_and_setting_columns(fleet):
    out = FeaturePipeline.model_inputs(fleet)
    assert not {"unit", "cycle", "op1", "op2", "op3"} & set(out.columns)


# ---------- metrikler ----------
def test_nasa_score_penalizes_late_more_than_early():
    early = nasa_score(np.array([50.0]), np.array([40.0]))  # 10 döngü erken
    late = nasa_score(np.array([50.0]), np.array([60.0]))   # 10 döngü geç
    assert late > early > 0
    assert nasa_score(np.array([10.0]), np.array([10.0])) == 0


def test_evaluate_perfect_prediction():
    m = evaluate([1, 2, 3], [1, 2, 3])
    assert m.rmse == 0 and m.mae == 0


# ---------- ayrım ----------
def test_split_by_unit_has_no_unit_overlap(fleet):
    train, val = split_by_unit(fleet, val_fraction=0.25, seed=1)
    assert set(train[UNIT]).isdisjoint(set(val[UNIT]))
    assert set(train[UNIT]) | set(val[UNIT]) == set(fleet[UNIT])
    assert val[UNIT].nunique() == 3


# ---------- uçtan uca ----------
def test_cross_validation_runs_and_beats_constant_baseline(fleet):
    cv = CrossValidator(lambda: SklearnRULModel("ridge", Ridge()), n_splits=3).run(fleet)
    assert len(cv.fold_metrics) == 3
    assert cv.rmse_mean < RULTargetBuilder(125).build(fleet).std()


def test_predictor_save_load_roundtrip(fleet, tmp_path):
    predictor = RULPredictor.train(fleet, SklearnRULModel("ridge", Ridge()))
    path = tmp_path / "model.joblib"
    predictor.save(path)

    original = predictor.predict_last(fleet)
    reloaded = RULPredictor.load(path).predict_last(fleet)

    assert list(original.index) == sorted(fleet[UNIT].unique())
    pd.testing.assert_series_equal(original, reloaded)