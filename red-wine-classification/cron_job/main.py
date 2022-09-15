import os
import random
import time
from urllib.parse import urljoin

import mlfoundry as mlf
import requests

client = mlf.get_client()
request_url = os.getenv("INFERENCE_SERVER_URL")


def generate_input_data():
    return {
        "fixed_acidity": random.random() * 13,
        "volatile_acidity": random.random() * 2,
        "citric_acid": random.random(),
        "residual_sugar": random.random() * 4,
        "chlorides": random.random() * 0.1,
        "free_sulfur_dioxide": int(random.random() * 60),
        "total_sulfur_dioxide": int(random.random() * 200),
        "density": 0.99 + random.random() * 0.01,
        "pH": 2.5 + random.random() * 1.23,
        "sulphates": 0.33 + random.random() * 0.5,
        "alcohol": 9.3 + random.random() * 0.6,
    }


# finding predictions on the data
predictions_dict = {}
for i in range(200):
    try:
        prediction = requests.get(
            url=urljoin(request_url, "/predict"), params=generate_input_data()
        ).json()
        predictions_dict[prediction["data_id"]] = prediction
    except Exception as e:
        print(f"Log Prediction Failed with Exception: {repr(e)}")


# logging actual values (null value for 12% cases, random value for 25% cases, correct value for rest
time.sleep(25)
for data_id, prediction in predictions_dict.items():
    rand = random.random()
    value = prediction["value"]
    if rand < 0.12:
        continue
    if rand > 0.75:
        value = random.choice(["3", "4", "5", "6", "7", "8"])
    client.log_actuals(
        model_version_fqn=prediction["model_version_fqn"],
        actuals=[mlf.Actual(data_id=data_id, value=value)],
    )
