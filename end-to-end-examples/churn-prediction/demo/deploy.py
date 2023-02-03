import argparse
import logging

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

parser.add_argument(
    "--model_deployed_url",
    type=str,
    required=True,
    help="end point of the trained model that would be used for inference",
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
        "MODEL_DEPLOYED_URL": args.model_deployed_url,
    },
    ports=[{"port": 8080}], #In public cloud deployment TrueFoundry exposes port 8501
    resources=Resources(
        cpu_request=0.5, cpu_limit=0.5, memory_limit=2500, memory_request=1500
    ),
    replicas=1
)
service.deploy(workspace_fqn=args.workspace_fqn)
