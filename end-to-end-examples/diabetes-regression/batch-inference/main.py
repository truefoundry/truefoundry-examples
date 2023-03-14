import mlfoundry as mlf
import os
import uuid
from datetime  import datetime
from sklearn.datasets import load_diabetes

# get the model from `Truefoundry Model Registry`.
MODEL_VERSION_FQN = os.environ["MODEL_VERSION_FQN"]
client = mlf.get_client()
model_version = client.get_model(MODEL_VERSION_FQN)

model_schema = model_version.model_schema

# load the model
model = model_version.load()

# load the dataset
X, y = load_diabetes(as_frame=True, return_X_y=True)

# we are sampling a smaller version of the dataset for demonstration purposes
X_sampled = X.sample(n=6)
y_sampled = y[X_sampled.index]

# make predictions from the model
y_pred = model.predict(X_sampled)

# parse the predictions in `mlfoundry.Prediction` format
predictions = []
for i in range(6):
    row = X_sampled.iloc[i]

    # create a dictionary of features
    features_dict = {}
    for feature in model_schema.features:
        features_dict[feature.name] = float(row[feature.name])

    #append `mlf.Prediction` objects in predictions list
    predictions.append(mlf.Prediction(
        data_id=uuid.uuid4().hex,
        features=features_dict,
        prediction_data={
            "value": float(y_pred[i]),
        },
        actual_value=float(y_sampled.iloc[i]),
        occurred_at=datetime.utcnow(),
        raw_data={"data": "any_data"},
    ))

# log the predictions using the client
client.log_predictions(model_version_fqn=MODEL_VERSION_FQN, predictions=predictions)

