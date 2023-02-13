import logging
import argparse
from servicefoundry import Build, Service, PythonBuild, Resources

parser = argparse.ArgumentParser()
parser.add_argument(
    "--workspace_fqn", type=str, required=True, help="fqn of the workspace to deploy to"
)
args = parser.parse_args()
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', force=True)

service = Service(
    name="fastapi-service",
    image=Build(build_spec=PythonBuild(command="uvicorn service:app --host 0.0.0.0 --port 8080")),
    resources=Resources(cpu_request=0.5, cpu_limit=1.0),
    ports=[{"port": 8080}]
)
service.deploy(workspace_fqn=args.workspace_fqn)
