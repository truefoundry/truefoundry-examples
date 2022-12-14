import logging

from servicefoundry import Build, PythonBuild, Service, Resources

logging.basicConfig(level=logging.INFO)
service = Service(
    name="fastapi",
    image=Build(
        build_spec=PythonBuild(
            command="uvicorn main:app --port 8000 --host 0.0.0.0",
        ),
    ),
    ports=[{"port": 8000}],
    resources=Resources(memory_limit=1500, memory_request=1000),
)
# service.deploy(workspace_fqn="YOUR_WORKSPACE_FQN")
service.deploy(workspace_fqn="v1:local:srihari-ws")
