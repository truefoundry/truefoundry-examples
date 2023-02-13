import argparse
import logging
from servicefoundry import (
        Service, Build, PythonBuild, Port, 
        Resources, GPU, GPUType 
)
# from servicefoundry import CUDAVersion

parser = argparse.ArgumentParser()
parser.add_argument(
    "--workspace_fqn",
    type=str,
    required=True,
    help="FQN of the workspace to deploy to",
)
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format=logging.BASIC_FORMAT)

service = Service(
    name="gpt-j-api",
    image=Build(
        build_spec=PythonBuild(
             python_version="3.10",
             # CUDA is not required for this example because pytorch ships its own libcuda.so
             # but it can be added like following:
             # cuda_version=CUDAVersion.CUDA_11_3_CUDNN8,
             requirements_path="requirements.txt",
             command="bash start.sh"
        ),
    ),
    ports=[Port(port=8080)],
    resources=Resources(
        cpu_request=3.5,
        cpu_limit=3.5,
        memory_request=14500,
        memory_limit=14500,
        ephemeral_storage_request=50000,
        ephemeral_storage_limit=50000,
        gpu=GPU(type=GPUType.T4)
    )
)

service.deploy(workspace_fqn=args.workspace_fqn, wait=False)
