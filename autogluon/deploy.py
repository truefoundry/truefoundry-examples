import logging, argparse
from servicefoundry import Build, Job, PythonBuild, Param, Port, LocalSource

logging.basicConfig(level=logging.INFO, force=True)
# parsing the arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--workspace_fqn", type=str, required=True, help="fqn of the workspace to deploy to"
)
args = parser.parse_args()

# defining the job specifications
job = Job(
    name="autogluon-train",
    image=Build(
        build_spec=PythonBuild(
            command="python train.py --ml_repo {{ml_repo}} --train_data_uri {{train_data_uri}} "
            "--test_data_uri {{test_data_uri}} --rf_n_estimators {{rf_n_estimators}}",
            requirements_path="requirements.txt",
        ),
        build_source=LocalSource(local_build=False),
    ),
    params=[
        Param(name="ml_repo", param_type="ml_repo"),
        Param(
            name="train_data_uri",
            default="https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv ",
        ),
        Param(
            name="test_data_uri",
            default="https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv",
        ),
        Param(
            name="rf_n_estimators",
            default="2",
        ),
    ],
)
job.deploy(workspace_fqn=args.workspace_fqn)
