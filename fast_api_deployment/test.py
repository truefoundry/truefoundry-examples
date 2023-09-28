import requests

url = "https://test-llama-70b-4bit-mitanshu-llm.demo2.truefoundry.tech/generate"

payload = {
    "inputs": "My name is Olivier and I",
    "parameters": {
        "max_new_tokens": 50,
        "repetition_penalty": 1.03,
        "return_full_text": False,
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.95,
    },
}
headers = {"Content-Type": "application/json", "Accept": "application/json"}

response = requests.post(url, json=payload, headers=headers)

print(response.json())
