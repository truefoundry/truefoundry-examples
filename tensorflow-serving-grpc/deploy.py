import argparse
import json
import logging
from servicefoundry import Service, Build, DockerFileBuild, Resources, Port, BasicAuthCreds, AppProtocol

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
    name="mobilenet-v3-small-tf",
    image=Build(build_spec=DockerFileBuild()),
    resources=Resources(
        cpu_request=1, 
        cpu_limit=1, 
        memory_request=500, 
        memory_limit=500,
    ),
    ports=[
        Port(
            port=9000,
            app_protocol=AppProtocol.grpc,
            # Note: Your cluster should allow subdomain based routing (*.yoursite.com) for gRPC to work correctly via public internet.
            # A host matching the wildcard base domain for the cluster can be explicitly configured by passing in `host`
        ),
    ],
    replicas=args.replicas,
)

service.deploy(workspace_fqn=args.workspace_fqn)
