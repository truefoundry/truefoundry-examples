import os
import argparse
import logging

from servicefoundry import Build, PythonBuild, Resources, Service, Autoscaling, CPUUtilizationMetric

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
        "TFY_API_KEY": os.getenv('TFY_API_KEY'),
        "INFERENCE_SERVER_URL": args.inference_server_url,
    },
    ports=[{"port": 8501, "host": "my-host-1234.tfy-ctl-euwe1-production.production.truefoundry.com"}], #In public cloud deployment TrueFoundry exposes port 8501
    resources=Resources(
        cpu_request=0.5, cpu_limit=0.5, memory_limit=2500, memory_request=1500
    ),
    replicas = Autoscaling(min_replicas=1, max_replicas=2, metrics=CPUUtilizationMetric(value=26))
)
service.deploy(workspace_fqn=args.workspace_fqn)
