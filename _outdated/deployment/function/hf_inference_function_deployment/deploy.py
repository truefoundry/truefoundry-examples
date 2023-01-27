import logging

from servicefoundry.function_service import FunctionService
from servicefoundry import Resources

from inference import Model

logging.basicConfig(level=logging.INFO)

service = FunctionService(
    name="t5-small",
    resources=Resources(memory_request=1000, memory_limit=1500),
)

service.register_class(Model, init_kwargs={"model_fqn": "t5-small"}, name="t5-small")

service.deploy(workspace_fqn="v1:local:my-ws-2")
