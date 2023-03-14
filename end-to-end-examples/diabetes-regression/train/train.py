import argparse

import matplotlib.pyplot as plt
import mlfoundry
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.svm import SVR
from sklearn.metrics import PredictionErrorDisplay

def train(kernel: str, n_quantiles: int):
    X, y = load_diabetes(as_frame=True, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    regressor = SVR(kernel=kernel)
    model = TransformedTargetRegressor(
        regressor=regressor,
        transformer=QuantileTransformer(n_quantiles=n_quantiles, output_distribution="normal"),
    ).fit(X_train, y_train)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    PredictionErrorDisplay.from_predictions(
        y_test,
        y_pred,
        kind="actual_vs_predicted",
        scatter_kwargs={"alpha": 0.5},
    )

    run = mlfoundry.get_client().create_run(
        project_name="diabetes-regression", run_name="SVR-with-QT"
    )

    run.log_params(regressor.get_params())
    run.log_metrics({"score": model.score(X_test, y_test)})
    run.log_plots({"actual_vs_predicted": plt})

    model_version = run.log_model(
        name="diabetes-regression",
        model=model,
        framework="sklearn",
        description="SVC model trained on initial data",
        model_schema={
          "features": [
            {"name": c, "type": "float"} for c in X.columns
          ],
          "prediction": "numeric",
        },
        custom_metrics=[{"name": "mean_square_error", "type": "metric", "value_type": "float"}]
    )
    print(f"Logged model: {model_version.fqn}")
    run.end()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", default="linear", type=str)
    parser.add_argument("--n_quantiles", default=100, type=int)
    args = parser.parse_args()

    train(kernel=args.kernel, n_quantiles=args.n_quantiles)
