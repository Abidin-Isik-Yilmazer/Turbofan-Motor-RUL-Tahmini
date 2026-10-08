# Engine RUL: Turbofan Motor Kalan Ömür Tahmini

NASA C-MAPSS (FD001) veri setiyle, 21 sensör okumasından bir jet motorunun
arızalanmadan önce **kaç uçuş döngüsü kaldığını** (Remaining Useful Life, RUL) tahmin eder.
Öngörücü bakım (predictive maintenance) örneğidir.

## Kurulum
```
pip install -r requirements.txt
```
NASA C-MAPSS dosyalarını (`train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`)
`data/raw/` klasörüne koyun.

## Kullanım
```
python main.py          # 3 modeli karşılaştırır, doğrulama + test sonuçları, grafik
python train_final.py   # tüm eğitim verisiyle eğitip outputs/ altına kaydeder
pytest -v               # testler (veri dosyası gerektirmez)
```

## Sonuçlar (FD001 test seti, her motorun son kaydı)
| Model | RMSE | MAE | NASA skoru |
|---|---|---|---|
| Hep ortalamayı söyle (baz çizgisi) | 43.07 | 35.90 | 33629 |
| Ridge | 21.06 | 16.80 | 1092 |
| Random Forest | **13.07** | 10.03 | 285 |
| Hist Gradient Boosting | 13.28 | 10.04 | 309 |

Özellik ve pencere seçimleri 5 katlı, motor bazlı çapraz doğrulama ile yapıldı; test seti bu kararlar için kullanılmadı. Çapraz doğrulamada RMSE yolculuğu:
sadece kayan ortalama 18.58 → +eğim(20) 15.47 → +eğim(50) **11.12**.

## Özellikler
1. Sabit sensörlerin elenmesi (6 sensör)
2. Eğitim istatistikleriyle standartlaştırma
3. Kayan ortalama (5 döngü), gürültüyü azaltır
4. Kayan eğim (50 döngü), bozulmanın yönünü ve hızını verir. En büyük kazancı bu getirdi.

Hedef: RUL, 125'te kırpılır (piecewise-linear). Motor sağlıklıyken bozulma belirtisi yoktur.

## Mimari
```
turbofan/
  data/          DataSource (ABC) -> CMAPSSFileSource; şema; motor bazlı ayrım
  features/      FeatureTransformer (ABC) -> dönüştürücüler; FeaturePipeline (Composite);
                 RULTargetBuilder
  models/        RULModel (ABC); SklearnRULModel (Adapter)
  evaluation/    RMSE, MAE, NASA skoru; grafikler
  pipeline.py          TrainingPipeline: akışı yöneten orkestratör
  cross_validation.py  CrossValidator: motor bazlı K-katlı doğrulama
  predictor.py         RULPredictor: özellik zinciri + model, kaydedilebilir
tests/           pytest (sentetik veriyle, veri dosyası gerektirmez)
```
Tasarım kararları:
- **Soyutlamalara bağımlılık:** orkestratörler somut sınıfları değil `DataSource`, `FeatureTransformer`
  ve `RULModel` arayüzlerini bilir. Yeni model eklemek mevcut kodu değiştirmez.
- **Veri sızıntısına karşı:** ayrım motor bazında; `fit` yalnızca eğitim verisinde;
  kayan hesaplar motor bazında (testlerle doğrulanır).
- **Değerlendirme:** test için her motorun yalnızca son kaydı, `RUL_FD001.txt` ile karşılaştırılır.

## Bilinen sınırlar / sonraki adımlar
- Yalnızca FD001 (tek çalışma koşulu). FD002/FD004 için koşul bazlı ölçekleme gerekir.
- Orta RUL aralığında (yaklaşık 50-100 döngü) model hâlâ iyimser kalma eğiliminde.
- Denenebilecekler: LSTM/1D-CNN, hiperparametre arama, RUL tavanı ayarı, ModelFactory.
