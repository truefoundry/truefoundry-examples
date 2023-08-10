import argparse
import os
import logging
from utils import generate_params
from module1 import normal
from servicefoundry import Job, Build, PythonBuild, Resources, Param

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--workspace_fqn", required=True, type=str)
args = parser.parse_args()

# First we define how to build our code into a Docker image
image = Build(
    build_spec=PythonBuild(
        command="python main.py normal --loc {{loc}} --scale {{scale}}",
        requirements_path="requirements.txt",
    )
)

job = Job(
    name="function-job-with-params",
    image=image,
    params=generate_params(normal),
    resources=Resources(
        cpu_request=0.25,
        cpu_limit=0.5,
        memory_request=512,
        memory_limit=512,
        ephemeral_storage_limit=512,
        ephemeral_storage_request=512,
    ),
)
job.deploy(workspace_fqn=args.workspace_fqn)
