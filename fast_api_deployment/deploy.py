import argparse
import logging

from servicefoundry import Build, Port, PythonBuild, Resources, Service

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
        cpu_request=8,
        cpu_limit=8,
        memory_request=70000,
        memory_limit=70000,
        ephemeral_storage_request=70000,
        ephemeral_storage_limit=70000,
    ),
    env={"UVICORN_WEB_CONCURRENCY": "1", "ENVIRONMENT": "dev"},
)
service.deploy(workspace_fqn=args.workspace_fqn)
