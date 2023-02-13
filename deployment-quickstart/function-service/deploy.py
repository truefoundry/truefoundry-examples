import logging
import argparse
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', force=True)

from functions import normal, uniform, multiply
from servicefoundry.function_service import FunctionService

parser = argparse.ArgumentParser()
parser.add_argument(
    "--workspace_fqn", type=str, required=True, help="fqn of the workspace to deploy to"
)
args = parser.parse_args()


service = FunctionService(name="fn-service")
service.register_function(normal)
service.register_function(uniform)
service.register_function(multiply)
deployment = service.deploy(workspace_fqn=args.workspace_fqn)
