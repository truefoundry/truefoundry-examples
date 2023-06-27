import argparse
import time
parser = argparse.ArgumentParser()
parser.add_argument(
    "--param1", type=str, required=True
)
parser.add_argument(
    "--param2", type=str, required=True
)
args = parser.parse_args()
time.sleep(100)
print("Running Job2 with param1: ", args.param1, " and param2: ", args.param2)

print("Job3 Run Complete.")
## write whatever code you want to run here
## save the final results in s3/db wherever you want