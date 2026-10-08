from sklearn.ensemble import RandomForestRegressor

from turbofan.cross_validation import CrossValidator
from turbofan.data.source import CMAPSSFileSource
from turbofan.features.pipeline import FeaturePipeline
from turbofan.features.transformers import (
    ConstantSensorDropper,
    RollingMeanFeatures,
    RollingSlopeFeatures,
    SensorScaler,
)
from turbofan.models.base import SklearnRULModel

data = CMAPSSFileSource("data/raw", "FD001").load()


def make_model() -> SklearnRULModel:
    return SklearnRULModel("random_forest", RandomForestRegressor(
        n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=42))


def features(*slope_windows: int):
    """Ortalama(5) + verilen pencerelerin eğimleri."""
    return lambda: FeaturePipeline([
        ConstantSensorDropper(),
        SensorScaler(),
        RollingMeanFeatures(5),
        *[RollingSlopeFeatures(w) for w in slope_windows],
    ])


feature_sets = {
    "egim(20) [referans]": features(20),
    "egim(30)": features(30),
    "egim(40)": features(40),
    "egim(50)": features(50),
    "egim(20) + egim(50)": features(20, 50),
}

for name, feature_factory in feature_sets.items():
    cv = CrossValidator(make_model, feature_factory).run(data.train)
    folds = [round(m.rmse, 1) for m in cv.fold_metrics]
    print(f"{name:24s} RMSE {cv.rmse_mean:.2f} ± {cv.rmse_std:.2f}   MAE {cv.mae_mean:.2f}   katlar: {folds}")