import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

from turbofan.data.source import CMAPSSFileSource
from turbofan.evaluation.plots import plot_predictions
from turbofan.models.base import SklearnRULModel
from turbofan.pipeline import TrainingPipeline


def main() -> None:
    source = CMAPSSFileSource("data/raw", "FD001")
    models = [
        SklearnRULModel("ridge", Ridge()),
        SklearnRULModel("random_forest", RandomForestRegressor(
            n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=42)),
        SklearnRULModel("hist_gradient_boosting", HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, random_state=42)),
    ]

    results = {}
    for model in models:
        result = TrainingPipeline(source, model).run()
        results[result.model_name] = result
        print(f"\n== {result.model_name}")
        print("  Doğrulama:", result.val_metrics)
        print("  Test     :", result.test_metrics)

    rf = results["random_forest"]
    plot_predictions(rf.y_true, rf.y_pred, title="Random Forest - FD001 test")
    plt.show()


if __name__ == "__main__":
    main()