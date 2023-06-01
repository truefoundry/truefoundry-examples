import argparse
import logging
import os
from servicefoundry import Build, PythonBuild, Resources, Job, Param, VolumeMount, LocalSource

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--workspace_fqn", type=str, required=True, help="fqn of the workspace to deploy to"
)
args = parser.parse_args()

# We now define our Service, we provide it a name, which script to run, environment variables and resources
job = Job(
    name="churn-prediction-train",
    image=Build(
        build_spec=PythonBuild(command="python main.py --n_neighbors {{n_neighbors}} --weights {{weights}} --algorithm {{algorithm}} --power {{power}} --leaf_size {{leaf_size}} --metric {{metric}} --n_jobs {{n_jobs}}") ,
        build_source=LocalSource(local_build=False)
    ),
    env={
        "TFY_API_KEY": "djE6dHJ1ZWZvdW5kcnk6bmlraGlsOjlkMDcxNw=="
    },
    params=[
        Param(name="n_neighbors", default='5', description="Number of neighbors to use by default"),
        Param(name="weights", default='uniform', description="Weight function used in prediction.  Possible values: uniform, distance"),
        Param(name="algorithm", default='auto', description="Algorithm used to compute the nearest neighbors: possible values: 'auto', 'ball_tree', 'kd_tree', 'brute'"),
        Param(name="power", default='2', description="Power parameter for the Minkowski metric. When p = 1, this is manhattan_dist, and euclidean_dist p = 2"),
        Param(name="leaf_size", default='30', description="Leaf size passed to BallTree or KDTree"),
        Param(name="metric", default='minkowski', description="The distance metric to use for the tree. The default metric is minkowski,"),
        Param(name="n_jobs", default='1', description="The number of parallel jobs to run for neighbors search"),
    ],
    resources=Resources(
        cpu_request=1, cpu_limit=1, memory_request=2000, memory_limit=2000,
    ),
)
# Finally, we call deploy to push it to Truefoundry platform
job.deploy(workspace_fqn=args.workspace_fqn)