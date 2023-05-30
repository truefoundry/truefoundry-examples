import argparse

import mlfoundry
from autogluon.tabular import TabularDataset, TabularPredictor

client = mlfoundry.get_client()


def train(
    run: mlfoundry.MlFoundryRun,
    rf_n_estimators: int,
    train_data_uri: str,
    test_data_uri: str,
):
    train_data = TabularDataset(train_data_uri)
    subsample_size = 500  # subsample subset of data for faster demo, try setting this to much larger values
    train_data = train_data.sample(n=subsample_size, random_state=0)
    print(train_data.head())

    label = "occupation"
    print("Summary of occupation column: \n", train_data["occupation"].describe())

    new_data = TabularDataset(test_data_uri)
    test_data = new_data[
        5000:
    ].copy()  # this should be separate data in your applications
    test_data_nolabel = test_data.drop(columns=[label])  # delete label column
    metric = "accuracy"

    predictor = TabularPredictor(label=label, eval_metric=metric).fit(
        train_data,
        hyperparameters={
            # "GBM": {"num_boost_round": 20},
            "RF": {"n_estimators": rf_n_estimators},
        },
    )
    run.log_artifact(name="model", artifact_paths=[(predictor.path,)])
    y_pred = predictor.predict(test_data_nolabel)
    print("Predictions:  ", list(y_pred)[:5])
    perf = predictor.evaluate(test_data, auxiliary_metrics=False)

    run.log_metrics(perf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--ml_repo", required=True, type=str)
    parser.add_argument("--train_data_uri", required=True, type=str)
    parser.add_argument("--test_data_uri", required=True, type=str)
    parser.add_argument("--rf_n_estimators", required=True, type=int)

    args = parser.parse_args()

    client.create_ml_repo(args.ml_repo)
    with client.create_run(args.ml_repo) as run:
        run.log_params(args)
        train(
            run=run,
            rf_n_estimators=args.rf_n_estimators,
            train_data_uri=args.train_data_uri,
            test_data_uri=args.test_data_uri,
        )
