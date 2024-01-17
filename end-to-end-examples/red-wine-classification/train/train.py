import argparse
import time
from joblib import dump
import mlfoundry
from datetime import datetime
from train_model import train_model
from sklearn.metrics import accuracy_score, f1_score
from dataset import get_initial_data

s = time.time()
parser = argparse.ArgumentParser()
parser.add_argument("--ml_repo_name", type=str)
parser.add_argument("--train_size", type=float)
parser.add_argument("--max_depth", type=int)
args = parser.parse_args()
# you can bring data from your own sources
X_train, X_test, y_train, y_test = get_initial_data(test_size=1-int(args.train_size), random_state=42)

# train the model, and get the associated metadata
model, metadata = train_model(X_train, y_train, X_test, y_test)

#save the model
path = dump(model, "classifier.joblib")

# get the training and test data predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# logging the data for experiment tracking
# create a run
run = mlfoundry.get_client().create_run(ml_repo=args.ml_repo_name, run_name=f"train-{datetime.now().strftime('%m-%d-%Y')}")
# log the hyperparameters
# run.log_params(model.get_params())
# log the metrics
run.log_metrics({
    'train/accuracy_score': accuracy_score(y_train, y_pred_train),
    'train/f1': f1_score(y_train, y_pred_train, average='weighted'),
    'test/accuracy_score': accuracy_score(y_test, y_pred_test),
    'test/f1': f1_score(y_test, y_pred_test, average='weighted'),
})

# log the dataset
run.log_dataset(
    dataset_name='train',
    features=X_train,
    predictions=y_pred_train,
    actuals=y_train,
)

# log the dataset
run.log_dataset(
    dataset_name='test',
    features=X_test,
    predictions=y_pred_test,
    actuals=y_test,
)

# log the model
model_version = run.log_model(
    name="Demand ForeCast Training",
    model_file_or_folder=path[0],
    framework="sklearn",
    description="model trained for Demand Forecasting",
    metadata=metadata,
    custom_metrics=[{"name": "log_loss", "type": "metric", "value_type": "float"}],
)

e = time.time()
print(f"Time taken to log models {e-s} ")
print(f"Logged model: {model_version.fqn}")