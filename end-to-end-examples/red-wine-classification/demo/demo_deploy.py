import argparse
import logging

from servicefoundry import Build, PythonBuild, Resources, Service

logging.basicConfig(level=logging.INFO)

# parsing the input arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--workspace_fqn",
    type=str,
    required=True,
    help="fqn of workspace where you want to deploy",
)

parser.add_argument(
    "--inference_server_url",
    type=str,
    required=True,
    help="end point of the trained model that would be used for inference",
)

args = parser.parse_args()

# creating a service object and defining all the configurations
service = Service(
    name="red-wine-demo",
    image=Build(
        build_spec=PythonBuild(
            command="streamlit run demo.py",
            python_version="3.9",
        ),
    ),
    env={
        # These will automatically map the secret value to the environment variable.
        "INFERENCE_SERVER_URL": args.inference_server_url,
        "TFY_HOST": "tfy-secret://user-truefoundry:red-wine-sg:TFY_HOST",
        "TFY_API_KEY": "tfy-secret://user-truefoundry:red-wine-sg:TFY_API_KEY",
    },
    ports=[{"port": 8501}], #In public cloud deployment TrueFoundry exposes port 8501
    resources=Resources(
        cpu_request=0.5, cpu_limit=.5, memory_limit=2500, memory_request=1500
    ),
)
service.deploy(workspace_fqn=args.workspace_fqn)
