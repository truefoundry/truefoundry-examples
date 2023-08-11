import argparse
import logging
import os

from servicefoundry import ModelDeployment, Resources, TruefoundryModelRegistry, Endpoint

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

model_deployment = ModelDeployment(
    name=f"churn-prediction",
    model_source=TruefoundryModelRegistry(model_version_fqn=os.environ['MODEL_VERSION_FQN']),
    resources=Resources(cpu_request=0.2, cpu_limit=0.5, memory_request=500, memory_limit=1000),
    endpoint=Endpoint(host="<Enter the target host for your model>")
)
)
model_deployment.deploy(workspace_fqn=args.workspace_fqn)
