import os
import time
import mlfoundry
from dataset import append_inference_data, get_initial_data
from datetime import datetime
from train_model import train_model
from sklearn.metrics import accuracy_score, f1_score

s = time.time()
# it is used for getting inference data for Retraining the model
MODEL_FQN, _ = os.getenv("MLF_MODEL_VERSION_FQN").rsplit(":", 1)


# You can bring data from your own sources
X_train, X_test, y_train, y_test = get_initial_data(test_size=0.1, random_state=42)

# fetch inference data for retraining the model on the combined dataset
X_train, y_train = append_inference_data(
    X_train=X_train, y_train=y_train, model_fqn=MODEL_FQN
)

model, metadata = train_model(X_train, y_train, X_test, y_test)
e = time.time()

print(f"################### Time taken to log models {e-s} ")

features = [{"name": column, "type": "float"} for column in X_train.columns]
schema = {"features": features, "prediction": "categorical"}
print(f"Schema: {schema}")

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# logging the data for experiment tracking
# You can push the model to your choice of storage or model registry.
run = mlfoundry.get_client().create_run(ml_repo="red-wine-quality-demo", run_name=f"retrain-{datetime.now().strftime('%m-%d-%Y')}")
run.log_params(model.get_params())
run.log_metrics({
    'train/accuracy_score': accuracy_score(y_train, y_pred_train),
    'train/f1': f1_score(y_train, y_pred_train, average='weighted'),
    'test/accuracy_score': accuracy_score(y_test, y_pred_test),
    'test/f1': f1_score(y_test, y_pred_test, average='weighted'),
})

run.log_dataset(
    dataset_name='train',
    features=X_train,
    predictions=y_pred_train,
    actuals=y_train,
)
run.log_dataset(
    dataset_name='test',
    features=X_test,
    predictions=y_pred_test,
    actuals=y_test,
)

model_version = run.log_model(
    name="red-wine-quality-classifier",
    model=model,
    framework="sklearn",
    description="model trained for red wine quality classification",
    metadata=metadata,
    model_schema=schema,
    custom_metrics=[{"name": "log_loss", "type": "metric", "value_type": "float"}],
)

print(f"Logged model: {model_version.fqn}")
