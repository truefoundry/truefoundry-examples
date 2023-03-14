import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as Classification
import mlfoundry as mlf
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def experiment_track(model, params, metrics, X_train, X_test):
    # Initialize the mlfoundry client.
    mlf_api = mlf.get_client()
    # Create a run
    mlf_run = mlf_api.create_run(
        project_name="churn-prediction", run_name="churn-train-job"
    )
    # Log the hyperparameters
    mlf_run.log_params(params)
    # Log the metrics
    mlf_run.log_metrics(metrics)
    # Log the train dataset
    mlf_run.log_dataset("train", X_train)
    # Log the test dataset
    mlf_run.log_dataset("test", X_test)
    # Log the model
    model_version = mlf_run.log_model(
        name="churn-model",
        model=model,
        # Specify the framework used (in this case sklearn)
        framework=mlf.ModelFramework.SKLEARN,
        description="churn-prediction-model",
    )
    # Log the plots
    mlf_run.log_plots({"confusion_matrix": plt}, step=1)
    # Return the model's fqn
    return model_version.fqn


def train_model(hyperparams):
    # Load the dataset
    df = pd.read_csv("https://raw.githubusercontent.com/nikp1172/datasets-sample/main/Churn_Modelling.csv")
    X = df.iloc[:, 3:-1].drop(["Geography", "Gender"], axis=1)
    y = df.iloc[:, -1]
    # Create train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Initialize the KNN Classifier
    classifier = Classification(n_neighbors=hyperparams['n_neighbors'], weights=hyperparams['weights'], algorithm=hyperparams['algorithm'], p=hyperparams['power'])
    # Fit the classifier with the training data
    classifier.fit(X_train, y_train)
    # Get the predictions
    y_pred = classifier.predict(X_test)
    # Get the ground truth labels
    labels = list(set(y))

    # Plot the confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(set(y)))
    disp.plot()

    # Get the metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred)
    }

    # Log the experiment
    experiment_track(classifier, classifier.get_params(), metrics, X_train, X_test)


if __name__ == "__main__":
    import argparse

    # Setup the argument parser by instantiating `ArgumentParser` class
    parser = argparse.ArgumentParser()
    # Add the hyperparamters as arguments
    parser.add_argument(
        "--n_neighbors",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--power",
        type=int,
        required=True
    )
    # Get the `Namespace` of the arguments
    args = parser.parse_args()
    hyperparams = vars(args)

    # Train the model
    train_model(hyperparams)
