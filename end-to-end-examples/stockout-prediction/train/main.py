import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as Classification
import mlfoundry as mlf
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from joblib import dump

def experiment_track(model_path, params, metrics, X_train, X_test):
    # initialize the mlfoundry client.
    mlf_api = mlf.get_client()

    # create a ml repo
    mlf_api.create_ml_repo("stockout-prediciton")
    # create a run
    mlf_run = mlf_api.create_run(
        ml_repo=args.ml_repo, run_name="churn-train-job"
    )
    # log the hyperparameters
    # log the metrics
    mlf_run.log_metrics(metrics)
    # log the model
    model_version = mlf_run.log_model(
        name="churn-model",
        model_file_or_folder=model_path,
        # specify the framework used (in this case sklearn)
        framework=mlf.ModelFramework.SKLEARN,
        description="churn-prediction-model",
    )
    # log the plots
    mlf_run.log_plots({"confusion_matrix": plt}, step=1)
    # return the model's fqn
    return model_version.fqn


def train_model(hyperparams):

    df = pd.read_csv("https://raw.githubusercontent.com/nikp1172/datasets-sample/main/Churn_Modelling.csv")
    X = df.iloc[:, 3:-1].drop(["Geography", "Gender"], axis=1)
    y = df.iloc[:, -1]
    # Create train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize the KNN Classifier
    classifier = Classification(
        n_neighbors=hyperparams['n_neighbors'],
        leaf_size=hyperparams['leaf_size'],
    )

    # Fit the classifier with the training data
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    
    # Get the ground truth labels
    labels = list(set(y))

    # Plot the confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(set(y)))
    disp.plot()

    # Get the metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
    }
    #save the model
    model_path = "./classifier.joblib"
    dump(classifier, model_path)
    # Log the experiment
    experiment_track(model_path, classifier.get_params(), metrics, X_train, X_test)


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
        "--ml_repo",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--leaf_size",
        type=int,
        required=True
    )
    args = parser.parse_args()
    hyperparams = vars(args)

    # Train the model
    train_model(hyperparams)
