from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler


def train_model(X_train, y_train, X_test=None, y_test=None):
    metadata = {}
    pipeline = make_pipeline(MinMaxScaler(), RandomForestClassifier(random_state=42))
    hyperparameters = {
        "randomforestclassifier__max_depth": [None, 2, 5, 10],
        "randomforestclassifier__max_features": [None, "sqrt", "log2"],
        "randomforestclassifier__n_estimators": [25, 50, 100],
    }
    model = GridSearchCV(
        pipeline,
        hyperparameters,
        cv=5,
        refit=True,
        n_jobs=4,
        return_train_score=True,
        verbose=True,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_train)
    print(classification_report(y_true=y_train, y_pred=y_pred))
    metadata["train_report"] = classification_report(
        y_true=y_train, y_pred=y_pred, output_dict=True
    )

    if X_test is not None and y_test is not None:
        y_pred = model.predict(X_test)
        print(classification_report(y_true=y_test, y_pred=y_pred))
        metadata["test_report"] = classification_report(
            y_true=y_test, y_pred=y_pred, output_dict=True
        )

    return model, metadata
