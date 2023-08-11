import argparse
import logging

from servicefoundry import Job, Build, PythonBuild, Resources

from module1 import normal
from utils import generate_params

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--workspace_fqn", required=True, type=str)
args = parser.parse_args()

params = generate_params(normal)

print(params)


image = Build(
    build_spec=PythonBuild(
        command=f"python -u main.py normal {params.command_argument}",
        requirements_path="requirements.txt",
    )
)

job = Job(
    name="function-job-with-params",
    image=image,
    params=params.params,
    resources=Resources(
        cpu_request=0.1,
        cpu_limit=0.1,
        memory_request=256,
        memory_limit=256,
        ephemeral_storage_limit=100,
        ephemeral_storage_request=100,
    ),
)
job.deploy(workspace_fqn=args.workspace_fqn)
