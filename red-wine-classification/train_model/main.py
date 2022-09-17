import os

import mlfoundry
from data import append_inference_data, get_initial_data
from train import train_model

# if not NONE, it is used for getting inference data for Retraining the model
MODEL_FQN = os.getenv("MLF_MODEL_FQN")

# You can bring data from your own sources
X_train, X_test, y_train, y_test = get_initial_data(test_size=0.1, random_state=42)

# fetch inference data for retraining the model on the combined dataset
if MODEL_FQN:
    X_train, y_train = append_inference_data(
        X_train=X_train, y_train=y_train, model_fqn=MODEL_FQN
    )

model, metadata = train_model(X_train, y_train, X_test, y_test)
features = [{"name": column, "type": "float"} for column in X_train.columns]
schema = {"features": features, "prediction": "categorical"}
print(f"Schema: {schema}")

# You can push the model to your choice of storage or model registry.
run = mlfoundry.get_client().create_run(project_name="red-wine-quality-demo")
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
