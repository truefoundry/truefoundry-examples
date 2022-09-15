import argparse
import logging

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
    name="red-wine-data-cron",
    image=Build(
        build_spec=PythonBuild(command="python main.py"),
    ),
    env={
        "INFERENCE_SERVER_URL": args.inference_server_url,
        "MLF_HOST": "tfy-secret://user-truefoundry:red-wine-quality-sg:MLF_HOST",
        "MLF_API_KEY": "tfy-secret://user-truefoundry:red-wine-quality-sg:MLF_API_KEY",
    },
    trigger=Schedule(schedule="*/20 * * * *"),
)
job.deploy(workspace_fqn=args.workspace_fqn)
