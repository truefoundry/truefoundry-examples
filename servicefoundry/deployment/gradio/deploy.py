import logging

from servicefoundry import Build, PythonBuild, Service, Resources

logging.basicConfig(level=logging.INFO)
service = Service(
    name="gradio",
    image=Build(
        build_spec=PythonBuild(
            command="python main.py",
        ),
    ),
    ports=[{"port": 8080}],
    resources=Resources(memory_limit=1500, memory_request=1000),
)
# deployment = service.deploy(workspace_fqn="YOUR_WORKSPACE_FQN")
deployment = service.deploy(workspace_fqn="v1:local:my-ws-2")
