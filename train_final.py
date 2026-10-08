from sklearn.ensemble import RandomForestRegressor

from turbofan.data.source import CMAPSSFileSource
from turbofan.evaluation.metrics import evaluate
from turbofan.models.base import SklearnRULModel
from turbofan.predictor import RULPredictor

MODEL_PATH = "outputs/rul_fd001.joblib"

data = CMAPSSFileSource("data/raw", "FD001").load()

model = SklearnRULModel("random_forest", RandomForestRegressor(
    n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=42))

# 1) Tüm eğitim verisiyle eğit ve diske kaydet
predictor = RULPredictor.train(data.train, model)
predictor.save(MODEL_PATH)
print("Kaydedildi:", MODEL_PATH)

# 2) Dosyadan geri yükle (sanki yeni bir program başlatılmış gibi) ve test et
loaded = RULPredictor.load(MODEL_PATH)
pred = loaded.predict_last(data.test)

print("\nİlk 5 motor için tahmin:")
print(pred.head().round(1))
print("\nTest:", evaluate(data.test_rul.loc[pred.index], pred))