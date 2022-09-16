import logging

from servicefoundry.function_service import FunctionService, remote
from servicefoundry import Resources

from inference import Model

logging.basicConfig(level=logging.INFO)

service = FunctionService(
    name="t5-small",
    resources=Resources(memory_request=1000, memory_limit=1500),
)

deployble_model_class = remote(Model, init_kwargs={"model_fqn": "t5-small"})
service.register_class(deployble_model_class)

service.deploy(workspace_fqn="v1:local:my-ws-2")
