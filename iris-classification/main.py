import os
import mlfoundry

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--REPO_NAME", required=True, type=str)
args = parser.parse_args()

REPO_NAME = args.REPO_NAME

mlf_client = mlfoundry.get_client()
mlf_client.create_ml_repo(REPO_NAME)

# Load the iris dataset as an example
iris = datasets.load_iris()
X = iris.data
y = iris.target

# create an MLFoundry run
run = mlf_client.create_run(ml_repo=REPO_NAME, run_name='iris-train-job')
run.set_tags({
    'dataset': 'iris'
})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

# store our hyperparams
run.log_params({
    'test_size': 0.7,
    'random_state': 42
})

# Create a logistic regression model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

# Log metrics
# store our hyperparams
run.log_metrics({
    'accuracy': accuracy
})

# Log model
model_version = run.log_model(
    name="iris-model",
    model=model,
    framework="sklearn"
)
