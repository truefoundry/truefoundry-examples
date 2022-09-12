import logging

from servicefoundry import Build, Service, DockerFileBuild, Resources

logging.basicConfig(level=logging.INFO)

service = Service(
    name="docker-svc",
    image=Build(build_spec=DockerFileBuild()),
    ports=[{"port": 8080}],
    resources=Resources(memory_limit=1500, memory_request=1000),
)
# service.deploy(workspace_fqn="YOUR_WORKSPACE_FQN")
service.deploy(workspace_fqn="v1:local:my-ws-2")
