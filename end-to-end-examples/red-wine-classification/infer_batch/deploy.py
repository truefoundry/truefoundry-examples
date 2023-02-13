import argparse
import logging
import os
from servicefoundry import Build, Job, PythonBuild, Schedule

parser = argparse.ArgumentParser()
parser.add_argument(
    "--workspace_fqn", type=str, required=True, help="fqn of workspace to deploy"
)
parser.add_argument(
    "--inference_server_url",
    type=str,
    required=True,
    help="url of the inference server",
)
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)

job = Job(
    name="red-wine-batch",
    image=Build(
        build_spec=PythonBuild(command="python infer_batch.py"),
    ),
    env={
        "INFERENCE_SERVER_URL": args.inference_server_url,
        "TFY_API_KEY": os.getenv('TFY_API_KEY')
    },
    trigger=Schedule(schedule="*/10 * * * *"),
)
job.deploy(workspace_fqn=args.workspace_fqn)
