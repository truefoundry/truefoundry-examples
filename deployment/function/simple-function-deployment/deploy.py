import logging

from servicefoundry.function_service import FunctionService

from module import normal, uniform

logging.basicConfig(level=logging.INFO)

service = FunctionService(name="func-service")
service.register_function(normal)
service.register_function(uniform)

service.deploy(workspace_fqn="v1:local:my-ws-2")
