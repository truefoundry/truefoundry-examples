import argparse
import logging

from servicefoundry import ModelDeployment, Resources, TruefoundryModelRegistry

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
    "--model_version_fqn",
    type=str,
    required=True,
    help="end point of the trained model that would be used for inference",
)

args = parser.parse_args()

model_deployment = ModelDeployment(
    name=f"churn-prediction",
    model_source=TruefoundryModelRegistry(model_version_fqn= args.model_version_fqn),
    resources=Resources(cpu_request=0.2, cpu_limit=0.5, memory_request=500, memory_limit=1000)
)
model_deployment.deploy(workspace_fqn=args.workspace_fqn)
