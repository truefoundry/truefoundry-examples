
import random

import mlfoundry as mlf
client = mlf.get_client()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "--run_name", type=str, required=True, help="name of run"
)
args = parser.parse_args()

run = client.create_run(project_name="tr-fe1", run_name=args.run_name)

run.log_metrics({"datadrift": random.random(), "f1_score": random.random()})