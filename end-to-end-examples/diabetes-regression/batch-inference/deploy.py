import argparse
import logging
import os
from servicefoundry import Build, Job, PythonBuild, Schedule

parser = argparse.ArgumentParser()
parser.add_argument(
    "--workspace_fqn", type=str, required=True, help="fqn of workspace to deploy"
)
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)

job = Job(
    name="diabetes-reg-batch",
    image=Build(
        build_spec=PythonBuild(command="python main.py"),
    ),
    env={
        "MODEL_VERSION_FQN": os.environ['MODEL_VERSION_FQN'],
        "TFY_API_KEY": os.environ['TFY_API_KEY']
    },
    trigger=Schedule(schedule="*/10 * * * *"),
)
job.deploy(workspace_fqn=args.workspace_fqn)
