import argparse
from urllib.parse import urljoin

import requests

parser = argparse.ArgumentParser()
parser.add_argument("--host", required=True, type=str)
args = parser.parse_args()

response = requests.post(
    urljoin(args.host, "/normal"), json={"loc": 0, "scale": 1, "size": [12, 1]}
)
print(response.json())

response = requests.post(
    urljoin(args.host, "/uniform"), json={"low": 0, "high": 1, "size": [12, 1]}
)
print(response.json())
