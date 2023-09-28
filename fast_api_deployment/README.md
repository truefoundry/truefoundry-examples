# Model deployment using Fast API as a Service
This folder contains script to deploy any text generation model as a FastAPI service

## Model selection
 * Change the `model_name_or_path` variable in app.py with the model id of the model that you want to deploy
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
