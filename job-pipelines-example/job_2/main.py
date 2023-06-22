from servicefoundry import trigger_job
import mlfoundry as mlf
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "--param1", type=str, required=True
)
parser.add_argument(
    "--param2", type=str, required=True
)
parser.add_argument(
    "--artifact_version_fqn", type=str, required=True
)
args = parser.parse_args()

print("Running Job2 with param1: ", args.param1, " and param2: ", args.param2)

client = mlf.get_client()
artifact_version = client.get_artifact(args.artifact_version_fqn).download(".")
with open("files/artifact.txt", "r") as f:
    print("Reading Contents from job1")
    print(f.read())


print("Job Run Complete. Triggering Job3")

# these output params can be filenames/paths, params required for the next step to
output_param1 = args.param1 + "-from-job2"
output_param2 = args.param2 + "-from-job2"

# trigger_job is a helper function that triggers the next job in the pipeline
job_run_result = trigger_job(
    application_fqn=os.environ["APPLICATION_FQN_JOB3"],
    params={"param1": output_param1, "param2": output_param2},
)

# the job run name can be considered as the unique identifier for the job run
job3_run_name = job_run_result.jobRunName

# You can save the job2_run_name in some a run of mlfoundry OR in a db table
# Your db table for runs can look something like this:
# <Other columns>| job1_run_name | job2_run_name | job3_run_name
# <Other >       | job-1-1234    | job2-6785     | job3-545433   
print(f"Job3 triggered with run name: {job3_run_name}")

