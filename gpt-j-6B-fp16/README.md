Run GPT-J-6B on Truefoundry with GPUs
---
This example downloads and runs [GPT-J-6B](https://huggingface.co/EleutherAI/gpt-j-6B/tree/float16) model on a single Nvidia T4 GPU and 15 GB RAM in FP16 mode using huggingface `transformers` and `accelerate`

Note on resources
---
Although it is possible to load the model using 14.5GB of RAM, to provide some buffer and generate larger texts, `memory_limit` can be increased in `deploy.py`, however that would also mean a larger instance would be provisioned which would cost more.


Deploying the example
---

1. Install servicefoundry

```shell
pip install "servicefoundry>=0.9.0,<0.10.0"
```

2. Login

> Optionally, you can pass in a host using `--host` for hosted truefoundry platform

```shell
sfy login --relogin
```

3. Deploy with your Workspace FQN

E.g.
```shell
python deploy.py --workspace_fqn "tfy-ctl-euwe1-devtest:tfy-demo"
```

Testing
---

Note:
- First request can time out because it takes a while to pull the model from Huggingface Hub
- Second request can time out because it can take a while to load the model and transfer to the GPU
- Generation results might not be that great in fp16 mode. Difficult to evaluate completely


```shell
curl -X 'POST' \
  '<YOUR ENDPOINT HERE>/generate' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "Hey Vsauce, Michael here. Today we are ",
  "temperature": 0.9,
  "max_length": 100
}'
```

can produce something like


```json
{
  "outputs": [
    {
      "generated_text": "\ngoing to be talking about why you shouldn't \nbe afraid of the dark. You know, \nI've mentioned it before, but a lot of \npeople have asked, well, why do we \nhave to be scared of the dark? Why do we \nhave to be so afraid of the dark? Why do \nwe have to go to such great lengths to \nmake ourselves more comfortable when"
    }
  ],
  "time": {
    "load": 314.46373830799985,
    "generate": 5.832131454000773
  }
}
```
