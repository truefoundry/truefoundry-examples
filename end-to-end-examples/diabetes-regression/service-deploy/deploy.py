import logging
import os
import argparse
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', force=True)

from servicefoundry import Build, Service, PythonBuild

parser = argparse.ArgumentParser()
parser.add_argument(
    "--workspace_fqn", type=str, required=True, help="fqn of the workspace to deploy to"
)
args = parser.parse_args()


service = Service(
    name="diabetes-reg",
    image=Build(
        build_spec=PythonBuild(
            command="uvicorn main:app --port 8000 --host 0.0.0.0",
        )
    ),
    ports=[{"port": 8000}],
    env={
        "TFY_API_KEY": os.environ['TFY_API_KEY'],
        "MODEL_VERSION_FQN": os.environ['MODEL_VERSION_FQN']
    },
    replicas=1
)
deployment = service.deploy(workspace_fqn=args.workspace_fqn)
