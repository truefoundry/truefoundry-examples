import argparse
import logging
from servicefoundry import Service, Build, PythonBuild, Resources, Port, AppProtocol

logging.basicConfig(level=logging.INFO, format=logging.BASIC_FORMAT)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--workspace_fqn",
    type=str,
    required=True,
    help="FQN of the workspace to deploy to",
)
parser.add_argument(
    "--replicas",
    type=int,
    required=False,
    default=1,
    help="How many replicas to run",
)
args = parser.parse_args()

service = Service(
    name="grpc-py-helloworld",
    image=Build(
        build_spec=PythonBuild(
            python_version="3.9",
            requirements_path="requirements.txt",
            command="python greeter_server_with_reflection.py"
        )
    ),
    resources=Resources(
        cpu_request=0.2,
        cpu_limit=0.5,
        memory_request=500,
        memory_limit=500,
    ),
    ports=[
        Port(
            port=50051,
            app_protocol=AppProtocol.grpc,
            # Note: Your cluster should allow subdomain based routing (*.yoursite.com) for gRPC to work correctly via public internet.
            # A host matching the wildcard base domain for the cluster can be explicitly configured by passing in `host`
            # E.g. if the cluster's base domain urls contain `*.tfy-ctl-euwe1-production.truefoundry.com`, then we can pass something like
            # host="grpc-py-helloworld-tfy-demo.tfy-ctl-euwe1-production.truefoundry.com"
        ),
    ],
    replicas=args.replicas,
)

service.deploy(workspace_fqn=args.workspace_fqn)
