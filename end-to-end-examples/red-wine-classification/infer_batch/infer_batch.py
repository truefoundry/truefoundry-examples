import os
import random
import time
from urllib.parse import urljoin

import mlfoundry as mlf
import pandas as pd
import requests

client = mlf.get_client()
request_url = os.getenv("INFERENCE_SERVER_URL")

DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"


# generate random input samples from the original dataset
def get_input_data(num_samples=random.randint(15, 30)):
    df = pd.read_csv(DATASET_URL, sep=";")
    df = df.sample(n=num_samples)
    y = df.pop("quality").to_list()
    X = df.to_dict("records")
    X = [{key.replace(" ", "_"): value for key, value in row.items()} for row in X]
    print(f"Fetched {num_samples} random data points")
    return X, y


features_list, actuals_list = get_input_data()

# finding predictions from the inference server
predictions_list = requests.post(
    url=urljoin(request_url, "/predict"), json=features_list
).json()

# logging actual values (null value for 12% cases, random value for 25% cases, correct value for rest)
time.sleep(10)
for prediction, actual in zip(predictions_list, actuals_list):
    rand = random.random()
    actual_value = str(actual)
    if rand < 0.12:
        continue
    if rand > 0.75:
        value = random.choice(["3", "4", "5", "6", "7", "8"])
    client.log_actuals(
        model_version_fqn=prediction["model_version_fqn"],
        actuals=[mlf.Actual(data_id=prediction["data_id"], value=actual_value)],
    )
