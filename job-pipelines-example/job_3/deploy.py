import argparse
import logging
import os
from servicefoundry import Build, PythonBuild, Resources, Job, Param, VolumeMount, LocalSource

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--workspace_fqn", type=str, required=True, help="fqn of the workspace to deploy to"
)
args = parser.parse_args()

# Define the job
job = Job(
    name="sample-job-3",
    image=Build(
        build_spec=PythonBuild(command="python main.py --param1 {{param1}} --param2 {{param2}}") ,
        build_source=LocalSource(local_build=False)
    ),
    params=[
        Param(name="param1", default='p1', description="Value for param1"),
        Param(name="param2", default='p2', description="Value for param2"),
    ],

    # This is the number of jobs that you want to run in parallel (limit). If more jobs are triggered they will be queued.
    concurrency_limit=1,

    resources=Resources(
        cpu_request=0.1, cpu_limit=0.1, memory_request=300, memory_limit=500,
    )
)

job.deploy(workspace_fqn=args.workspace_fqn, wait=False)