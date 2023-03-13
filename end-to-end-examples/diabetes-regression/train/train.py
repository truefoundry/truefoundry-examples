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
    # Load the dataset
    X, y = load_diabetes(as_frame=True, return_X_y=True)

    # NOTE:- You can pass these configurations via command line
    # arguments, config file, environment variables.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize the model
    regressor = SVR(kernel=kernel)

    # Fit the model
    model = TransformedTargetRegressor(
        regressor=regressor,
        transformer=QuantileTransformer(n_quantiles=n_quantiles, output_distribution="normal"),
    )
    model.fit(X_train, y_train)

    # Get the predictions from the model
    y_pred = model.predict(X_test)

    # Plot the confusion_matrix
    PredictionErrorDisplay.from_predictions(
        y_test,
        y_pred,
        kind="actual_vs_predicted",
        scatter_kwargs={"alpha": 0.5},
    )

    # Create a run, setting the project's name the following run
    # should be associated with via setting the `project_name`
    # And the name of the run name via `run_name`
    run = mlfoundry.get_client().create_run(
        project_name="diabetes-regression", run_name="SVR-with-QT"
    )

    # Log the hyperparameters of the model
    run.log_params(regressor.get_params())

    # Log the metrics of the model
    run.log_metrics({"score": model.score(X_test, y_test)})

    # Log the associated plots
    run.log_plots({"actual_vs_predicted": plt})

    # Log the model
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

    # End the run
    run.end()


if __name__ == "__main__":
    # Setup the argument parser by instantiating `ArgumentParser` class
    parser = argparse.ArgumentParser()
    # Add the following hyperparamters as arguments
    parser.add_argument("--kernel", default="linear", type=str)
    parser.add_argument("--n_quantiles", default=100, type=int)
    #Get the `Namespace` of the arguments
    args = parser.parse_args()

    #Run the train function
    train(kernel=args.kernel, n_quantiles=args.n_quantiles)
