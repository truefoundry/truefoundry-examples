# Steps to run


1. Login to truefoundry
```
pip install servicefoundry
servicefoundry login --host <enter your host here>
```
2. Deploy the jobs in reverse order of their execution

* Deploy Job3
    ```
    cd job_3
    python deploy.py --workspace_fqn <paste workspace fqn here>
    ```
* Deploy job_2 (copy the Application FQN of job_3 from UI or from logs)
    ```
    cd ../job_2
    APPLICATION_FQN_JOB3=<paste application fqn> python deploy.py --workspace_fqn <paste workspace fqn>
    ```
* Deploy job_1 (copy the Application FQN of job_2 from UI or from logs)
    ```
    cd ../job_1
    APPLICATION_FQN_JOB2=<paste application fqn> python deploy.py --workspace_fqn <paste workspace fqn>
    ```
3. Trigger job runs. Run the following python script
    ```python
    from servicefoundry import trigger_job
    trigger_job(
        application_fqn=<application fqn of first job>,
        params={
            "param1": "hello",
            "param2": "world
        }
    )
    ```

For reference, check the following doc: https://docs.truefoundry.com/docs/viewing-job-runs-via-sdk-and-cli
