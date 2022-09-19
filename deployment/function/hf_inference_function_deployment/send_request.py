import argparse
from urllib.parse import urljoin

import requests

parser = argparse.ArgumentParser()
parser.add_argument("--host", required=True, type=str)
args = parser.parse_args()

response = requests.post(
    urljoin(args.host, "/Model/infer"),
    json={"input_text": "translate English to German: Hello world."},
)
print(response.json())
