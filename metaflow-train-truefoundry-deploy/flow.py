import datetime
import os
import functools
from metaflow import (
    FlowSpec,
    step,
    environment,
    conda_base,
    Parameter,
    card,
    current,
)
from metaflow.cards import Markdown, Table, Image

# BASE_URL = "https://metaflow-demo-public.s3.us-west-2.amazonaws.com/taxi/clean"
# TRAIN_URL = BASE_URL + "/train_sample.parquet"
# TEST_URL = BASE_URL + "/test.parquet"
TRAIN_URL = "train_sample.parquet"
TEST_URL = "test.parquet"
SAMPLE_FRAC = 0.01
WORKSPACE_FQN = "tfy-ctl-euwe1-devtest:tfy-demo"


# See: https://github.com/Netflix/metaflow/issues/24#issuecomment-571976372
def pip(libraries):
    def decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            import subprocess
            import sys

            for library, version in libraries.items():
                print('Pip Install:', library, version)
                subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', library + '==' + version])
                print('Pip Install Done')
            return function(*args, **kwargs)

        return wrapper

    return decorator


@conda_base(
    libraries={
        "scikit-learn": "1.1.2",
        "pandas": "1.4.2",
        "pyarrow": "9.0.0",
        "matplotlib": "3.5.0",
    }
)
class FareRegressionFlow(FlowSpec):
    train_data_url = Parameter("train_url", default=TRAIN_URL)
    test_data_url = Parameter("test_url", default=TEST_URL)

    FEATURES = [
        "pickup_year",
        "pickup_dow",
        "pickup_hour",
        "abs_distance",
        "pickup_longitude",
        "dropoff_longitude",
    ]

    @card
    @step
    def start(self):
        import pandas as pd

        train_df = pd.read_parquet(self.train_data_url)
        train_df = train_df.sample(frac=SAMPLE_FRAC, random_state=42)
        self.X_train = train_df.loc[:, self.FEATURES]
        self.y_train = train_df.loc[:, "fare_amount"]
        print("Training set includes %d data points" % len(train_df))
        self.next(self.model)

    @card
    @step
    def model(self):
        print("Fitting a model")
        self.rf_model = make_random_forest()
        self.rf_model.fit(self.X_train, self.y_train)
        self.next(self.evaluate)

    @card
    @environment(vars={
        "TFY_HOST": os.getenv("TFY_HOST"),
        "TFY_API_KEY": os.getenv("TFY_API_KEY"),
    })
    @pip(libraries={"mlfoundry": "0.6.1"})
    @step
    def evaluate(self):
        from sklearn.metrics import mean_squared_error as mse
        import numpy as np
        import pandas as pd
        import mlfoundry

        print("Starting eval on the model")

        test_df = pd.read_parquet(self.test_data_url)
        test_df = test_df.sample(frac=SAMPLE_FRAC, random_state=42)
        self.X_test = test_df.loc[:, self.FEATURES]
        self.y_test = test_df.loc[:, "fare_amount"]
        n_rows = self.y_test.shape[0]

        self.y_pred = self.rf_model.predict(self.X_test)
        self.y_baseline_pred = np.repeat(self.y_test.mean(), n_rows)
        self.model_rmse = mse(self.y_test, self.y_pred)
        self.baseline_rmse = mse(self.y_test, self.y_baseline_pred)

        print(f"Model RMSE: {self.model_rmse}")
        print(f"Baseline RMSE: {self.baseline_rmse}")

        client = mlfoundry.get_client()
        run_name = datetime.datetime.now().strftime('run-%Y-%m-%d-%H-%M-%S')
        print("Creating run", run_name)
        with client.create_run(
            ml_flow="cj-metaflow-tf", run_name=run_name
        ) as run:
            run.log_metrics({"model_rmse": self.model_rmse, "baseline_rmse": self.baseline_rmse}, step=0)
            model_version = run.log_model(name="rf_model", model=self.rf_model, framework="sklearn", step=0)
            self.model_version_fqn = model_version.fqn
        
        # E.g. This is what a model_version_fqn will look like
        # self.model_version_fqn = "model:truefoundry/user-truefoundry/cj-metaflow-tf/rf_model:1"
        
        print("Logged model fqn is", self.model_version_fqn)
        self.next(self.create_report)

    @card(type="blank")
    @step
    def create_report(self):
        self.plot = plot(self.y_test, self.y_pred)
        current.card.append(Markdown("# Model Report"))
        current.card.append(
            Table(
                [
                    ["Random Forest", float(self.model_rmse)],
                    ["Baseline", float(self.baseline_rmse)],
                ],
                headers=["Model", "RMSE"],
            )
        )
        current.card.append(Image(self.plot, label="Correct vs. Predicted Fare"))
        self.next(self.deploy_to_truefoundry)

    @environment(vars={
        "TFY_HOST": os.getenv("TFY_HOST"),
        "TFY_API_KEY": os.getenv("TFY_API_KEY"),
    })
    @pip(libraries={"servicefoundry": "0.6.6"})
    @step
    def deploy_to_truefoundry(self):
        deploy_to_truefoundry(
            tfy_host=os.getenv("TFY_HOST"), 
            tfy_api_key=os.getenv("TFY_API_KEY"), 
            model_version_fqn=self.model_version_fqn,
            workspace_fqn=WORKSPACE_FQN
        )
        self.next(self.end)

    @step
    def end(self):
        pass


def make_random_forest():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.compose import ColumnTransformer

    ct_pipe = ColumnTransformer(
        transformers=[
            ("encoder", OneHotEncoder(categories="auto", sparse=False), ["pickup_dow"]),
            (
                "std_scaler",
                StandardScaler(),
                ["abs_distance", "pickup_longitude", "dropoff_longitude"],
            ),
        ]
    )
    return Pipeline(
        [
            ("ct", ct_pipe),
            (
                "forest_reg",
                RandomForestRegressor(
                    n_estimators=10, n_jobs=-1, random_state=3, max_features=8
                ),
            ),
        ]
    )


def plot(correct, predicted):
    import matplotlib.pyplot as plt
    from io import BytesIO
    import numpy

    MAX_FARE = 100
    line = numpy.arange(0, MAX_FARE, MAX_FARE / 1000)
    plt.rcParams.update({"font.size": 22})
    plt.scatter(x=correct, y=predicted, alpha=0.01, linewidth=0.5)
    plt.plot(line, line, linewidth=2, color="black")
    plt.xlabel("Correct fare")
    plt.ylabel("Predicted fare")
    plt.xlim([0, MAX_FARE])
    plt.ylim([0, MAX_FARE])
    fig = plt.gcf()
    fig.set_size_inches(18, 10)
    buf = BytesIO()
    fig.savefig(buf)
    return buf.getvalue()


def deploy_to_truefoundry(
    tfy_host, 
    tfy_api_key, 
    model_version_fqn,
    workspace_fqn
):
    import logging
    from servicefoundry import Build, PythonBuild, Resources, Service
    logging.basicConfig(level=logging.INFO)

    service = Service(
        name="cj-metaflow-tf-taxi-rf",
        image=Build(
            build_spec=PythonBuild(
                python_version="3.9",
                requirements_path="predict_requirements.txt",
                command="uvicorn app:app --port 8080 --host 0.0.0.0",
            ),
        ),
        env={
            "TFY_HOST": tfy_host,
            "TFY_API_KEY": tfy_api_key,
            "MODEL_VERSION_FQN": model_version_fqn,
        },
        ports=[{"port": 8080}],
        resources=Resources(
            cpu_request=0.5,
            cpu_limit=1.5,
            memory_request=1500,
            memory_limit=2500,
        ),
    )

    service.deploy(workspace_fqn=workspace_fqn)


if __name__ == '__main__':
    FareRegressionFlow()
