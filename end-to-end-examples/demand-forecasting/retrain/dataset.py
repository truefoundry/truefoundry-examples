from datetime import datetime, timedelta, timezone

import mlfoundry as mlf
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"


def append_inference_data(X_train, y_train, model_fqn):
    client = mlf.get_client()
    inference_data = client.get_inference_dataset(
        model_fqn=model_fqn,
        start_time=datetime.now(timezone.utc) - timedelta(days=7),
        actual_value_required=True,
    )

    X_train.append([data.features for data in inference_data], ignore_index=True)
    y_train.append(
        pd.Series([int(data.actual["value"]) for data in inference_data]).astype(str),
        ignore_index=True,
    )
    return X_train, y_train


def get_initial_data(test_size=0.1, random_state=42):
    df = pd.read_csv(DATASET_URL, sep=";")
    y = df.pop("quality").astype(str)
    X = df
    print(f"Num samples: {len(X)}")
    oversample = SMOTE()
    X, y = oversample.fit_resample(X, y)
    print(f"Num samples after Oversampling: {len(X)}")
    if test_size is None:
        X_train, X_test, y_train, y_test = X, None, y, None
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42, stratify=y
        )
    return X_train, X_test, y_train, y_test

