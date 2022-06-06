This example trains and logs a sklearn model for iris plant detection
with mlfoundry and then deploys it using servicefoundry

To deploy this example in your own account,

1. Create an account at https://app.truefoundry.com
2. Clone this repository and cd to this directory
3. Run the train.ipynb jupyter notebook to train a sklearn model and log
   it
4. Note down the run id and update it in `servicefoundry.yaml` as
    ```yaml
    Component.spec.container.env:
      - name: TFY_RUN_ID
        value: <run id here>
    ```
5. Install servicefoundry cli and create a workspace (one time setup)
    ```shell
    pip install servicefoundry
    sfy login
    sfy create workspace workspace-1
    ```
6. Deploy it!
    ```shell
    sfy deploy
    ```

> Note: You can choose to create a workspace with a different name, just
> make sure to update `servicefoundry.yaml`
