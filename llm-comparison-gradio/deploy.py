import argparse
import logging

from servicefoundry import Build, LocalSource, PythonBuild, Service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s", force=True)

parser = argparse.ArgumentParser()
parser.add_argument("--workspace_fqn", type=str, required=True, help="fqn of the workspace to deploy to")
args = parser.parse_args()

service = Service(
    name="llm-comparison-demo",
    image=Build(
        build_spec=PythonBuild(
            command="python main.py",
        ),
        build_source=LocalSource(local_build=False),
    ),
    ports=[{"port": 8080, "host": "llm-comparison-demo.demo2.truefoundry.tech"}],
    replicas=1,
)
deployment = service.deploy(workspace_fqn=args.workspace_fqn)
