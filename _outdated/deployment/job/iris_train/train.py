import mlfoundry
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report

X, y = load_iris(as_frame=True, return_X_y=True)
X = X.rename(columns={
        "sepal length (cm)": "sepal_length",
        "sepal width (cm)": "sepal_width",
        "petal length (cm)": "petal_length",
        "petal width (cm)": "petal_width",
})

# NOTE:- You can pass these configurations via command line
# arguments, config file, environment variables.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
pipe = Pipeline([("scaler", StandardScaler()), ("svc", SVC())])
pipe.fit(X_train, y_train)
print(classification_report(y_true=y_test, y_pred=pipe.predict(X_test)))

# You can push model to any storage, here we are using Truefoundry's Model Registry
run = mlfoundry.get_client().create_run(project_name="iris-classification")
model_version = run.log_model(
    name="iris-classifier",
    model=model,
    framework="sklearn",
    description="SVC model trained on initial data",
)
print(f"Logged model: {model_version.fqn}")
