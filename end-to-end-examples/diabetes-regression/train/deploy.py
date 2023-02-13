import logging
import os
import argparse
from servicefoundry import Build, Job, PythonBuild, Param

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', force=True)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--workspace_fqn", type=str, required=True, help="fqn of the workspace to deploy to"
)
args = parser.parse_args()

job = Job(
    name="diabetes-reg-train",
    image=Build(
        build_spec=PythonBuild(
            command="python train.py --kernel {{kernel}} --n_quantiles {{n_quantiles}}",
            requirements_path="requirements.txt",
        )
    ),
    params=[
            Param(name="n_quantiles", default='100'),
            Param(name="kernel", default='linear', description="kernel for svm"),
        ],
    env={
        "TFY_API_KEY": os.environ['TFY_API_KEY']
    }
)
deployment = job.deploy(workspace_fqn=args.workspace_fqn)