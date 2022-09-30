# Red Wine Classification Problem

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

## Querying the Deployed Model

This can either be done via the [fastapi endpoint](https://red-wine-prediction-tfy-demo.tfy-ctl-euwe1-develop.develop.truefoundry.tech) directly via browser.

You can also query with python script:

```python
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
```

## Steps to run

Install and setup servicefoundry on your computer.

```commandline
pip install servicefoundry
servicefoundry use server https://app.develop.truefoundry,tech
servicefoundry login
```

Each file can be run either using the python script or using cli.

#### Deploy using python script
```commandline
python deploy_{scriptname}.py --workspace_fqn <your workspace fqn>
```
You might need to pass other command line arguments also. [depending upon the the deploy script]

#### Deploy using CLI

For deployment via CLI. The application specifications are defined in a yaml file
```commandline
servicefoundry deploy --file <yaml_file_name>.yaml --workspace_fqn <your_workspace_fqn>
```