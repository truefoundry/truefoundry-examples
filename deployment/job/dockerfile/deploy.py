# Replace `<YOUR_SECRET_FQN>` with the actual value.
import logging
import argparse
from servicefoundry import Build, Job, DockerFileBuild

parser = argparse.ArgumentParser()
parser.add_argument("--workspace_fqn", type=str, required=True, help="fqn of the workspace to deploy to")
args = parser.parse_args()
logging.basicConfig(level=logging.INFO)

# This time around we use `DockerFileBuild` as the build spec. By default it looks for
# ./Dockerfile
job = Job(
    name="iris-train-job",
    image=Build(build_spec=DockerFileBuild()),
    env={"MLF_API_KEY": "tfy-secret://<YOUR_SECRET_FQN>"}
)
job.deploy(workspace_fqn=args.workspace_fqn)
