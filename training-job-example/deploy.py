import logging, os, argparse
from servicefoundry import Build, Job, PythonBuild, Param, Port, LocalSource

# parsing the arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--workspace_fqn", type=str, required=True, help="fqn of the workspace to deploy to"
)
parser.add_argument(
    "--ml_repo", type=str, required=True, help="name of the ml-repo where you want to save the training metadata"
)
args = parser.parse_args()

# defining the job specifications
job = Job(
    name="diabetes-train-job",
    image=Build(
        build_spec=PythonBuild(
            command="python train.py --kernel {{kernel}} --n_quantiles {{n_quantiles}}",
            requirements_path="requirements.txt",
        ),
        build_source=LocalSource(local_build=False)
    ),
    params=[
            Param(name="n_quantiles", default='100'),
            Param(name="kernel", default='linear', description="kernel for svm"),
        ],
    env={
        "ML_REPO_NAME": args.ml_repo    
    }
)
deployment = job.deploy(workspace_fqn=args.workspace_fqn)