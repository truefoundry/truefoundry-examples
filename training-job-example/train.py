import argparse

import matplotlib.pyplot as plt
import mlfoundry
import os
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.svm import SVR

def train(kernel: str, n_quantiles: int):
    # load the dataset
    X, y = load_diabetes(as_frame=True, return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # initialize the model
    regressor = SVR(kernel=kernel)
    model = TransformedTargetRegressor(
        regressor=regressor,
        transformer=QuantileTransformer(n_quantiles=n_quantiles, output_distribution="normal"),
    )
    model.fit(X_train, y_train)

    # get the predictions from the model
    y_pred = model.predict(X_test)

    # create a run in truefoundry’s ml_repo
    run = mlfoundry.get_client().create_run(
        ml_repo=os.environ.get("ML_REPO_NAME"), run_name="SVR-with-QT"
    )

    # log the hyperparameters of the model
    run.log_params(regressor.get_params())

    # log the metrics of the model
    run.log_metrics({"score": model.score(X_test, y_test)})

    # log the model
    model_version = run.log_model(
        name="diabetes-regression",
        model=model,
        framework="sklearn"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", default="linear", type=str)
    parser.add_argument("--n_quantiles", default=100, type=int)
    args = parser.parse_args()

    # run the train function
    train(kernel=args.kernel, n_quantiles=args.n_quantiles)