import argparse
import logging

from servicefoundry import Build, PythonBuild, Resources, Job, DockerFileBuild 

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--workspace_fqn", type=str, required=True, help="fqn of the workspace to deploy to"
)
args = parser.parse_args()



# We now define our Service, we provide it a name, which script to run, environment variables and resources
job = Job(
    name="job-comp-tes1t",
    image=Build(build_spec=PythonBuild(
        python_version="3.9",
        command="python train.py",)
    ),
    env={
        # These will automatically map the secret value to the environment variable.
        "TFY_HOST": "https://app.devtest.truefoundry.tech",
        "TFY_API_KEY": "tfy-secret://user-truefoundry:housing-sg:TFY_API_KEY",
    },
    resources=Resources(
        cpu_request=1, cpu_limit=1, memory_request=4000, memory_limit=4000, 
    ),
)
# Finally, we call deploy to push it to Truefoundry platform
job.deploy(workspace_fqn=args.workspace_fqn)
