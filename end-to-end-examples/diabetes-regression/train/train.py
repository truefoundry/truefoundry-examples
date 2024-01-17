import argparse
import os
from joblib import dump
import matplotlib.pyplot as plt
import mlfoundry
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.svm import SVR
from sklearn.metrics import PredictionErrorDisplay

def train(kernel: str, n_quantiles: int, ml_repo, train_size):
    # load the dataset
    X, y = load_diabetes(as_frame=True, return_X_y=True)

    # NOTE:- you can pass these configurations via command line
    # arguments, config file, environment variables.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1-int(train_size), random_state=42
    )

    # initialize the model
    regressor = SVR(kernel=kernel)

    # fit the model
    model = TransformedTargetRegressor(
        regressor=regressor,
        transformer=QuantileTransformer(n_quantiles=n_quantiles, output_distribution="normal"),
    )
    model.fit(X_train, y_train)
    model_path = "./classifier.joblib"
    #save the model
    dump(model, model_path)
    # get the predictions from the model
    y_pred = model.predict(X_test)

    # plot the confusion_matrix
    PredictionErrorDisplay.from_predictions(
        y_test,
        y_pred,
        kind="actual_vs_predicted",
        scatter_kwargs={"alpha": 0.5},
    )

    # create a run, setting the project's name the following run
    # should be associated with via setting the `ml_repo`
    # and the name of the run name via `run_name`
    client = mlfoundry.get_client()
    ml_repo = client.create_ml_repo(ml_repo)
    run = client.create_run(
        ml_repo=ml_repo.name, run_name="SVR-with-QT"
    )

    # log the hyperparameters of the model
    run.log_params(regressor.get_params())

    # log the metrics of the model
    run.log_metrics({"score": model.score(X_test, y_test)})

    # log the associated plots
    run.log_plots({"actual_vs_predicted": plt})

    # log the model
    model_version = run.log_model(
        name="diabetes-regression",
        model_file_or_folder=model_path,
        framework="sklearn",
        description="SVC model trained on initial data",
        custom_metrics=[{"name": "mean_square_error", "type": "metric", "value_type": "float"}]
    )
    print(f"Logged model: {model_version.fqn}")

    # end the run
    run.end()


if __name__ == "__main__":
    # setup the argument parser by instantiating `ArgumentParser` class
    parser = argparse.ArgumentParser()
    # add the following hyperparamters as arguments
    parser.add_argument("--kernel", default="linear", type=str)
    parser.add_argument("--n_quantiles", default=100, type=int)
    parser.add_argument("--ml_repo_name", type=str)
    parser.add_argument("--train_size", type=int)
    parser.add_argument("--max_depth", type=int)
    # get the `Namespace` of the arguments
    args = parser.parse_args()

    # run the train function
    train(kernel=args.kernel, n_quantiles=args.n_quantiles, ml_repo=args.ml_repo_name, train_size=args.train_size)
