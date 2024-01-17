import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"


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

