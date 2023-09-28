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
    name="llama2-70b-4bit",
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
        cpu_request=4,
        cpu_limit=4,
        memory_request=35000,
        memory_limit=35000,
        ephemeral_storage_request=35000,
        ephemeral_storage_limit=35000,
        gpu_count=1,
        node=NodeSelector(gpu_type=GPUType.A100_40GB),
    ),
    env={"UVICORN_WEB_CONCURRENCY": "1", "ENVIRONMENT": "dev"},
)
service.deploy(workspace_fqn=args.workspace_fqn)
