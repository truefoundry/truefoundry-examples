import logging
import argparse
from servicefoundry import Build, Job, PythonBuild, Resources

parser = argparse.ArgumentParser()
parser.add_argument(
    "--workspace_fqn", type=str, required=True, help="fqn of the workspace to deploy to"
)
args = parser.parse_args()


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', force=True)

job = Job(
    name="counter-job",
    image=Build(build_spec=PythonBuild(command="python run.py")),
    resources=Resources(cpu_request=0.5, cpu_limit=1.0)
)
job.deploy(workspace_fqn=args.workspace_fqn)
