import argparse
import logging

from servicefoundry import (
    Build,
    GPUType,
    NodeSelector,
    Port,
    PythonBuild,
    Resources,
    Service,
)

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--workspace_fqn", required=True, type=str)
parser.add_argument("--host", required=True, type=str)
args = parser.parse_args()

service = Service(
    name="diabetespredicition",
    image=Build(
        build_spec=PythonBuild(
            command="uvicorn app:app --port 8000 --host 0.0.0.0",
            requirements_path="requirements.txt",
        )
    ),
    ports=[
        Port(
            port=8000,
            host=args.host,
        )
    ],
    resources=Resources(
        cpu_request=0.2,
        cpu_limit=0.3,
        memory_request=300,
        memory_limit=400,
    ),
)
service.deploy(workspace_fqn=args.workspace_fqn)
