# Red Wine Classification Problem

## Demo Video

[![Wine Classification demo](https://i.ytimg.com/an_webp/ZnjsA78RuI4/mqdefault_6s.webp?du=3000&sqp=CKKe65kG&rs=AOn4CLBaOtFI7tmhZgQp3TOXzO714IDWug)](https://www.youtube.com/watch?v=ZnjsA78RuI4 "Demo-Problem-Statement-Iris-Deployment-Monitoring")

## Description of the Problem

The aim of the problem is to predict the quality of red-wine as a number between 0-10 with input features like pH, density etc of red-wine.

## Dataset Description

Here is the Link to the dataset:  https://www.kaggle.com/code/sevilcoskun/red-wine-quality-classification/data

![img.png](../assets/img.png)

Size of dataset: 100.95 kb 
Number of Unique Rows: 1599
Total Columns: 12 (11 input features, one the target quality value)

## Model Trained

The model trained is a RandomForestClassifier. 
The model is trained using sklearn and we have used GridSearchCV for hyperparameter tuning.

## Instructions

<details>
<summary><b><font size="5">Setting up servicefoundry</font></b></summary>

Install and setup servicefoundry on your computer.

```commandline
pip install servicefoundry
sfy login
```
</details>

<details>
<summary><b><font size="5">Training model</font></b></summary>

To deploy using the python API run the following command in terminal

1. Change working directory to train folder
    ```commandline
    cd train
    ```
2. Create [workspace](https://docs.truefoundry.com/documentation/deploy/concepts/workspace) and [API Key](https://app.truefoundry.com/settings) on the TrueFoundry platform 
3. Replace the value against `TFY_API_KEY` in the `deploy.py` file with the API Key found in [Settings Tab](https://app.truefoundry.com/settings) of your TrueFoundry account
4. Copy the workspace fqn of the workspace that you want to use from the [Workspaces Tab](https://app.truefoundry.com/workspaces) of TrueFoundry <i>[(Instructions here)](https://docs.truefoundry.com/documentation/deploy/concepts/workspace#copy-workspace-fqn-fully-qualified-name)</i>
5. To deploy using python script:
   ```commandline
   python deploy.py --workspace_fqn <YOUR_WORKSPACE_FQN>
   ```
   To deploy using CLI:
   
   > Replace values for `TFY_API_KEY` and in `servicefoundry.yaml`
   
   ```commandline
   sfy deploy -f servicefoundry.yaml --workspace-fqn <YOUR_WORKSPACE_FQN>
   ```
6. Click on the dashboard link in the terminal
7. Click on the <b>"Trigger Job"</b> on the dashboard to run the training job
</details>

<details>
<summary><b><font size="5">Deploying realtime inference</font></b></summary>

Note: <i>It is necessary to train a model before being able to deploy it as a service</i>
1. Change working directory to infer_realtime folder
    ```commandline
    cd infer_realtime
    ```
2. Create [workspace](https://docs.truefoundry.com/documentation/deploy/concepts/workspace) and [API Key](https://app.truefoundry.com/settings) on the TrueFoundry platform 
3. Replace the value against `TFY_API_KEY` in the `deploy.py` file with the API Key found in [Settings Tab](https://app.truefoundry.com/settings) of your TrueFoundry account

4. Copy the workspace fqn of the workspace that you want to use from the [Workspaces Tab](https://app.truefoundry.com/workspaces) of TrueFoundry <i>[(Instructions here)](https://docs.truefoundry.com/documentation/deploy/concepts/workspace#copy-workspace-fqn-fully-qualified-name)</i>

5. Find the model_version_fqn of the model that you want to deploy from

   * Go to  [Experiments Tracking tab of TrueFoundry](https://app.truefoundry.com/mlfoundry)
   * Click on the project name that you trained (<i>red-wine-quality-demo by default</i>)
   * Click on models tab
   * Click on the model name to the model trained to open the tab showing different versions of the model
   * Copy the FQN of the latest version of the model


6. To deploy using python script:
   ```commandline
   python deploy.py --workspace_fqn <YOUR_WORKSPACE_FQN> --model_version_fqn <YOUR_MODEL_VERSION_FQN>
   ```
   To deploy using CLI:
   
   > Replace values for `TFY_API_KEY` and `MLF_MODEL_VERSION_FQN` in `servicefoundry.yaml`
   
   ```commandline
   sfy deploy -f servicefoundry.yaml --workspace-fqn <YOUR_WORKSPACE_FQN>
   ```
7. Click on the dashboard link in the terminal to open the service deployment page with FastAPI EndPoint

</details>

<details>
<summary><b><font size="5">Querying the deployed model</font></b></summary>

This can either be done via the [fastapi endpoint](https://red-wine-prediction-tfy-demo.tfy-ctl-euwe1-develop.develop.truefoundry.tech) directly via browser.

You can also query with python script:

```python
import requests

# Replace the below url with your own endpoint
request_url = "https://red-wine-prediction-tfy-demo.tfy-ctl-euwe1-develop.develop.truefoundry.tech"

features_list = [
    {
     'fixed_acidity': 7.5,
     'volatile_acidity': 0.42,
     'citric_acid': 0.32,
     'residual_sugar': 2.7,
     'chlorides': 0.067,
     'free_sulfur_dioxide': 7.0,
     'total_sulfur_dioxide': 25.0,
     'density': 0.99628,
     'pH': 3.24,
     'sulphates': 0.44,
     'alcohol': 10.4
    }
]

predictions_list = requests.post(
    url=urljoin(request_url, "/predict"), json=features_list
).json()
print(predictions_list)
```
</details>

<details>
<summary><b><font size="5">Running batch inference</font></b></summary>

Note: <i>It is necessary to deploy live inference model before being able to deploy it as a service</i>

1. Change working directory to infer_batch folder
    ```commandline
    cd infer_batch
    ```
2. Create [workspace](https://docs.truefoundry.com/documentation/deploy/concepts/workspace) and [API Key](https://app.truefoundry.com/settings) on the TrueFoundry platform 
3. Replace the value against `TFY_API_KEY` in the `deploy.py` file with the API Key found in [Settings Tab](https://app.truefoundry.com/settings) of your TrueFoundry account

4. Copy the workspace fqn of the workspace that you want to use from the [Workspaces Tab](https://app.truefoundry.com/workspaces) of TrueFoundry <i>[(Instructions here)](https://docs.truefoundry.com/documentation/deploy/concepts/workspace#copy-workspace-fqn-fully-qualified-name)</i>

5. Copy the inference_server_url from:
   * Go to  [Deployments Tab of TrueFoundry](https://app.truefoundry.com/deployments)
   * Open the service that was deployment as live inference model <i>("red-wine-fastapi" by default)</i>
   * Copy the Endpoint link and paste it as the value against `INFERENCE_SERVER_URL` in `deploy.py`
6. To deploy using python script:
   ```commandline
   python deploy.py --workspace_fqn <YOUR_WORKSPACE_FQN> --inference_server_url <YOUR_INFERENCE_SERVER_URL>
   ```
   To deploy using CLI:

   > Replace values for `TFY_API_KEY` and `INFERENCE_SERVER_URL` in `servicefoundry.yaml`

   ```commandline
   sfy deploy -f servicefoundry.yaml --workspace-fqn <YOUR_WORKSPACE_FQN>
   ```
7. Click on the dashboard link in the terminal
8. Click on the <b>"Trigger Job"</b> on the dashboard to run the batch inference job

</details>

<details>
<summary><b><font size="5">Retraining Model </font></b></summary>

Note: <i>It is necessary to train a model before being able to deploy it as a service</i>
1. Change working directory to retrain folder
    ```commandline
    cd retrain
    ```
2. Create [workspace](https://docs.truefoundry.com/documentation/deploy/concepts/workspace) and [API Key](https://app.truefoundry.com/settings) on the TrueFoundry platform 
3. Replace the value against `TFY_API_KEY` in the `deploy.py` file with the API Key found in [Settings Tab](https://app.truefoundry.com/settings) of your TrueFoundry account

4. Copy the workspace fqn of the workspace that you want to use from the [Workspaces Tab](https://app.truefoundry.com/workspaces) of TrueFoundry <i>[(Instructions here)](https://docs.truefoundry.com/documentation/deploy/concepts/workspace#copy-workspace-fqn-fully-qualified-name)</i>

5. Find the model_version_fqn of the model that you want to deploy from

   * Go to experiments [tracking tab of TrueFoundry](https://app.truefoundry.com/mlfoundry)
   * Click on the project name that you trained (<i>red-wine-quality-demo by default</i>)
   * Click on models tab
   * Click on the model name to the model trained to open the tab showing different versions of the model
   * Copy the FQN of the latest version of the model and paste it as the value against `MLF_MODEL_VERSION_FQN` in `deploy.py`


6. To deploy using python script:
   ```commandline
   python deploy.py --workspace_fqn <YOUR_WORKSPACE_FQN> --model_version_fqn <YOUR_MODEL_VERSION_FQN>
   ```
   To deploy using CLI:
   
   > Replace values for `TFY_API_KEY` and `MLF_MODEL_VERSION_FQN` in `servicefoundry.yaml`
   
   ```commandline
   sfy deploy -f servicefoundry.yaml --workspace-fqn <YOUR_WORKSPACE_FQN>
   ```
7. Click on the <b>"Trigger Job"</b> on the dashboard to run model retraining job



</details>

<details>
<summary><b><font size="5">Deploying Demo </font></b></summary>

Note: <i>It is necessary to deploy live inference model before being able to deploy a demo</i>

1. Change working directory to infer_batch folder
    ```commandline
    cd demo
    ```

2. Copy the workspace fqn of the workspace that you want to use from the [Workspaces Tab](https://app.truefoundry.com/workspaces) of TrueFoundry <i>[(Instructions here)](https://docs.truefoundry.com/documentation/deploy/concepts/workspace#copy-workspace-fqn-fully-qualified-name)</i>

3. Copy the inference_server_url from:

    * Go to  [Deployments Tab of TrueFoundry](https://app.truefoundry.com/deployments)
    * Open the service that was deployment as live inference model <i>("red-wine-fastapi" by default)</i>
    * Copy the Endpoint link and paste it as the value against `INFERENCE_SERVER_URL` in `deploy.py`

4. To deploy using python script:
   ```commandline
   python deploy.py --workspace_fqn <YOUR_WORKSPACE_FQN> --inference_server_url <YOUR_INFERENCE_SERVER_URL>
   ```
   To deploy using CLI:

   > Replace values for `INFERENCE_SERVER_URL` in `servicefoundry.yaml`

   ```commandline
   sfy deploy -f servicefoundry.yaml --workspace-fqn <YOUR_WORKSPACE_FQN>
   ```

5. Click on the dashboard link in the terminal

6. Click on the <b>"Endpoint"</b> link on the dashboard to open the streamlit demo


</details>