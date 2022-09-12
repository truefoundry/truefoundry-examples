# Replace `<YOUR_SECRET_FQN>` with the actual value.
import logging
import argparse
from servicefoundry import Build, Job, PythonBuild

parser = argparse.ArgumentParser()
parser.add_argument("--workspace_fqn", type=str, required=True, help="fqn of the workspace to deploy to")
args = parser.parse_args()
logging.basicConfig(level=logging.INFO)

# First we define how to build our code into a Docker image
image = Build(
    build_spec=PythonBuild(
        command="python train.py",
        requirements_path="requirements.txt",
    )
)
job = Job(
    name="iris-train-job",
    image=image,
    env={"MLF_API_KEY": "tfy-secret://<YOUR_SECRET_FQN>"}
)
job.deploy(workspace_fqn=args.workspace_fqn)
