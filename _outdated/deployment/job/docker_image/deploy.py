# Replace `<YOUR_DOCKERHUB_USERNAME>`, `<YOUR_SECRET_FQN>` with the actual values.
import logging
import argparse
from servicefoundry import Build, Job, Image

parser = argparse.ArgumentParser()
parser.add_argument("--workspace_fqn", type=str, required=True, help="fqn of the workspace to deploy to")
args = parser.parse_args()
logging.basicConfig(level=logging.INFO)

# This time around we use `Image` directly and give it a image_uri
job = Job(
    name="iris-train-job",
    image=Image(
        type="image",
        image_uri="<YOUR_DOCKERHUB_USERNAME>/tf-job-docker-image:latest"
    ),
    env={"MLF_API_KEY": "tfy-secret://<YOUR_SECRET_FQN>"}
)
job.deploy(workspace_fqn=args.workspace_fqn)
