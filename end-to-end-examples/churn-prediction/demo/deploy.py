import argparse
import logging
import os

from servicefoundry import Build, PythonBuild, Resources, Service

logging.basicConfig(level=logging.INFO)

# parsing the input arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--workspace_fqn",
    type=str,
    required=True,
    help="fqn of workspace where you want to deploy",
)
args = parser.parse_args()

# creating a service object and defining all the configurations
service = Service(
    name="churn-prediction-demo",
    image=Build(
        build_spec=PythonBuild(
            command="python demo.py",
            python_version="3.9",
        ),
    ),
    env={
        "MODEL_DEPLOYED_URL": os.environ['MODEL_DEPLOYED_URLpi'],
    },
    ports=[{"port": 8080}], #In public cloud deployment TrueFoundry exposes port 8501
    resources=Resources(
        cpu_request=0.5, cpu_limit=0.5, memory_limit=2500, memory_request=1500
    ),
    replicas=1
)
service.deploy(workspace_fqn=args.workspace_fqn)
