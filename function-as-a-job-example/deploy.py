import argparse
import os
import logging
from utils import generate_params
from module import normal
from servicefoundry import Job, Build, PythonBuild, Resources, Param

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--workspace_fqn", required=True, type=str)
args = parser.parse_args()

# First we define how to build our code into a Docker image
image = Build(
    build_spec=PythonBuild(
        command="python main.py normal {{loc}} {{scale}}",
        requirements_path="requirements.txt",
    )
)

params = generate_params(normal)

job = Job(
    name="function-job-with-params",
    image=image,
    params=normal,
)
job.deploy(workspace_fqn=args.workspace_fqn)