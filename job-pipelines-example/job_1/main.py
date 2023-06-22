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
args = parser.parse_args()

print("Running Job1 with param1: ", args.param1, " and param2: ", args.param2)

# Lets save a artifact (maybe the output of the job)
# you may use boto3 library and save it to s3 if required
client = mlf.get_client()

# create a sample artifact
with open("artifact.txt", "w") as f:
    f.write("hello-world")
    f.write(f"Param1: {args.param1}")

# lets log it to the repo
client.create_ml_repo("my-test-ml-repo")
run = client.create_run(
    ml_repo="my-test-ml-repo", run_name="test-run"
)
artifact_version = run.log_artifact(
    name="hello-world-file",
    artifact_paths=[('artifact.txt', '.')]
)

print("Job Run Complete. Triggering Job2")

# these output params can be filenames/paths, params required for the next step to
output_param1 = args.param1 + "-from-job1"
output_param2 = args.param2 + "-from-job1"

# trigger_job is a helper function that triggers the next job in the pipeline
job_run_result = trigger_job(
    application_fqn=os.environ["APPLICATION_FQN_JOB2"],
    params={
        "param1": output_param1,
        "param2": output_param2,
        "artifact_version_fqn": artifact_version.fqn
    },
)

# the job run name can be considered as the unique identifier for the job run
job2_run_name = job_run_result.jobRunName

# You can save the job2_run_name in some a run of mlfoundry OR in a db table
# Your db table for runs can look something like this:
# <Other columns>| job1_run_name | job2_run_name | job3_run_name
# <Other >       | job-1-1234    | job2-6785     | job3-545433           
print(f"Job2 triggered with run name: {job2_run_name}")

