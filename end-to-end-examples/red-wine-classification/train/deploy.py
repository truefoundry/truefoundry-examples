import argparse
import logging
import os

from servicefoundry import Build, Job, PythonBuild, Resources
from servicefoundry.internal.experimental import trigger_job

logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser()
parser.add_argument(
    "--workspace_fqn", type=str, required=True, help="fqn of the workspace to deploy to"
)
args = parser.parse_args()

# servicefoundry uses this specification to automatically create a Dockerfile and build an image,
job_run_command = "python train.py"

python_build = PythonBuild(
    python_version="3.9",
    command=job_run_command,
)
env = {
    "TFY_API_KEY": os.getenv('TFY_API_KEY'),
}
job = Job(
    name="red-wine-train",
    image=Build(build_spec=python_build),
    env=env,
    resources=Resources(
        cpu_request=1, cpu_limit=1.5, memory_request=1000, memory_limit=1500
    ),
)
deployed_job = job.deploy(workspace_fqn=args.workspace_fqn)

# Run/Trigger the deployed job
trigger_job(deployment_fqn=deployed_job.fqn, command=job_run_command)
