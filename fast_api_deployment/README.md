# Llama 2 70b 4bit
This folder contains script to deploy llama 2 70b Chat in 4bit quantize mode as a FastAPI service

## Deploy model as a Fast API service
To deploy the model as a fast api service run following commands
* First Install servicefoundry
```
pip install servicefoundry
```
* Then login using cli
```
sfy login --host <host>
```
* Then Run the below command to deploy the service
```
python deploy.py --workspace_fqn <workspace-fqn> --host <your-host>
```
