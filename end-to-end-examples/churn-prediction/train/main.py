import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as Classification
import mlfoundry as mlf
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def experiment_track(model, params, metrics, X_train, X_test):
    mlf_api = mlf.get_client()
    mlf_run = mlf_api.create_run(
        project_name="churn-prediction-3", run_name="churn-train-job"
    )
    mlf_run.log_params(params)
    mlf_run.log_metrics(metrics)
    mlf_run.log_dataset("train", X_train)
    mlf_run.log_dataset("test", X_test)
    model_version = mlf_run.log_model(
        name="churn-model",
        model=model,
        framework=mlf.ModelFramework.SKLEARN,
        description="churn-prediction-model",
    )
    mlf_run.log_plots({"confusion_matrix": plt}, step=1)
    return model_version.fqn


def train_model(hyperparams):
    df = pd.read_csv("https://raw.githubusercontent.com/nikp1172/datasets-sample/main/Churn_Modelling.csv")
    X = df.iloc[:, 3:-1].drop(["Geography", "Gender"], axis=1)
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    classifier = Classification(n_neighbors=hyperparams['n_neighbors'], weights=hyperparams['weights'], algorithm=hyperparams['algorithm'], p=hyperparams['power'])
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    labels = list(set(y))

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(set(y)))
    disp.plot()
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred)
    }

    experiment_track(classifier, classifier.get_params(), metrics, X_train, X_test)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()
    hyperparams = vars(args)

    train_model(hyperparams)
