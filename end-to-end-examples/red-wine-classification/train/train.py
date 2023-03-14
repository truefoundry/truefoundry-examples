import time
import mlfoundry
from datetime import datetime
from train_model import train_model
from sklearn.metrics import accuracy_score, f1_score
from dataset import get_initial_data

s = time.time()

# You can bring data from your own sources
X_train, X_test, y_train, y_test = get_initial_data(test_size=0.1, random_state=42)

# Train the model, and get the associated metadata
model, metadata = train_model(X_train, y_train, X_test, y_test)

# Create a features dictionary and schema
features = [{"name": column, "type": "float"} for column in X_train.columns]
schema = {"features": features, "prediction": "categorical"}
print(f"Schema: {schema}")

# Get the training and test data predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# logging the data for experiment tracking
# You can push the model to your choice of storage or model registry.
# Create a run
run = mlfoundry.get_client().create_run(project_name="red-wine-quality-demo", run_name=f"train-{datetime.now().strftime('%m-%d-%Y')}")
# Log the hyperparameters
run.log_params(model.get_params())
# Log the metrics
run.log_metrics({
    'train/accuracy_score': accuracy_score(y_train, y_pred_train),
    'train/f1': f1_score(y_train, y_pred_train, average='weighted'),
    'test/accuracy_score': accuracy_score(y_test, y_pred_test),
    'test/f1': f1_score(y_test, y_pred_test, average='weighted'),
})

# Log the dataset
run.log_dataset(
    dataset_name='train',
    features=X_train,
    predictions=y_pred_train,
    actuals=y_train,
)

# Log the dataset
run.log_dataset(
    dataset_name='test',
    features=X_test,
    predictions=y_pred_test,
    actuals=y_test,
)

# Log the model
model_version = run.log_model(
    name="red-wine-quality-classifier",
    model=model,
    framework="sklearn",
    description="model trained for red wine quality classification",
    metadata=metadata,
    model_schema=schema,
    custom_metrics=[{"name": "log_loss", "type": "metric", "value_type": "float"}],
)

e = time.time()
print(f"################### Time taken to log models {e-s} ")
print(f"Logged model: {model_version.fqn}")